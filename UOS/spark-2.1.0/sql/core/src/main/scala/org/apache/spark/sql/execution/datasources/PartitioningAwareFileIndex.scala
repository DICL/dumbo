/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.datasources

import java.io.FileNotFoundException

import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._
import org.apache.hadoop.mapred.{FileInputFormat, JobConf}

import org.apache.spark.internal.Logging
import org.apache.spark.metrics.source.HiveCatalogMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{expressions, InternalRow}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.util.SerializableConfiguration

/**
 * An abstract class that represents [[FileIndex]]s that are aware of partitioned tables.
 * It provides the necessary methods to parse partition data based on a set of files.
 *
 * @param parameters as set of options to control partition discovery
 * @param userPartitionSchema an optional partition schema that will be use to provide types for
 *                            the discovered partitions
 */
abstract class PartitioningAwareFileIndex(
    sparkSession: SparkSession,
    parameters: Map[String, String],
    userPartitionSchema: Option[StructType],
    fileStatusCache: FileStatusCache = NoopCache) extends FileIndex with Logging {
  import PartitioningAwareFileIndex.BASE_PATH_PARAM

  /** Returns the specification of the partitions inferred from the data. */
  def partitionSpec(): PartitionSpec

  override def partitionSchema: StructType = partitionSpec().partitionColumns

  protected val hadoopConf = sparkSession.sessionState.newHadoopConfWithOptions(parameters)

  protected def leafFiles: mutable.LinkedHashMap[Path, FileStatus]

  protected def leafDirToChildrenFiles: Map[Path, Array[FileStatus]]

  override def listFiles(filters: Seq[Expression]): Seq[PartitionDirectory] = {
    val selectedPartitions = if (partitionSpec().partitionColumns.isEmpty) {
      PartitionDirectory(InternalRow.empty, allFiles().filter(f => isDataPath(f.getPath))) :: Nil
    } else {
      prunePartitions(filters, partitionSpec()).map {
        case PartitionPath(values, path) =>
          val files: Seq[FileStatus] = leafDirToChildrenFiles.get(path) match {
            case Some(existingDir) =>
              // Directory has children files in it, return them
              existingDir.filter(f => isDataPath(f.getPath))

            case None =>
              // Directory does not exist, or has no children files
              Nil
          }
          PartitionDirectory(values, files)
      }
    }
    logTrace("Selected files after partition pruning:\n\t" + selectedPartitions.mkString("\n\t"))
    selectedPartitions
  }

  /** Returns the list of files that will be read when scanning this relation. */
  override def inputFiles: Array[String] =
    allFiles().map(_.getPath.toUri.toString).toArray

  override def sizeInBytes: Long = allFiles().map(_.getLen).sum

  def allFiles(): Seq[FileStatus] = {
    if (partitionSpec().partitionColumns.isEmpty) {
      // For each of the root input paths, get the list of files inside them
      rootPaths.flatMap { path =>
        // Make the path qualified (consistent with listLeafFiles and listLeafFilesInParallel).
        val fs = path.getFileSystem(hadoopConf)
        val qualifiedPathPre = fs.makeQualified(path)
        val qualifiedPath: Path = if (qualifiedPathPre.isRoot && !qualifiedPathPre.isAbsolute) {
          // SPARK-17613: Always append `Path.SEPARATOR` to the end of parent directories,
          // because the `leafFile.getParent` would have returned an absolute path with the
          // separator at the end.
          new Path(qualifiedPathPre, Path.SEPARATOR)
        } else {
          qualifiedPathPre
        }

        // There are three cases possible with each path
        // 1. The path is a directory and has children files in it. Then it must be present in
        //    leafDirToChildrenFiles as those children files will have been found as leaf files.
        //    Find its children files from leafDirToChildrenFiles and include them.
        // 2. The path is a file, then it will be present in leafFiles. Include this path.
        // 3. The path is a directory, but has no children files. Do not include this path.

        leafDirToChildrenFiles.get(qualifiedPath)
          .orElse { leafFiles.get(qualifiedPath).map(Array(_)) }
          .getOrElse(Array.empty)
      }
    } else {
      leafFiles.values.toSeq
    }
  }

  protected def inferPartitioning(): PartitionSpec = {
    // We use leaf dirs containing data files to discover the schema.
    val leafDirs = leafDirToChildrenFiles.filter { case (_, files) =>
      files.exists(f => isDataPath(f.getPath))
    }.keys.toSeq
    userPartitionSchema match {
      case Some(userProvidedSchema) if userProvidedSchema.nonEmpty =>
        val spec = PartitioningUtils.parsePartitions(
          leafDirs,
          typeInference = false,
          basePaths = basePaths)

        // Without auto inference, all of value in the `row` should be null or in StringType,
        // we need to cast into the data type that user specified.
        def castPartitionValuesToUserSchema(row: InternalRow) = {
          InternalRow((0 until row.numFields).map { i =>
            Cast(
              Literal.create(row.getUTF8String(i), StringType),
              userProvidedSchema.fields(i).dataType).eval()
          }: _*)
        }

        PartitionSpec(userProvidedSchema, spec.partitions.map { part =>
          part.copy(values = castPartitionValuesToUserSchema(part.values))
        })
      case _ =>
        PartitioningUtils.parsePartitions(
          leafDirs,
          typeInference = sparkSession.sessionState.conf.partitionColumnTypeInferenceEnabled,
          basePaths = basePaths)
    }
  }

  private def prunePartitions(
      predicates: Seq[Expression],
      partitionSpec: PartitionSpec): Seq[PartitionPath] = {
    val PartitionSpec(partitionColumns, partitions) = partitionSpec
    val partitionColumnNames = partitionColumns.map(_.name).toSet
    val partitionPruningPredicates = predicates.filter {
      _.references.map(_.name).toSet.subsetOf(partitionColumnNames)
    }

    if (partitionPruningPredicates.nonEmpty) {
      val predicate = partitionPruningPredicates.reduce(expressions.And)

      val boundPredicate = InterpretedPredicate.create(predicate.transform {
        case a: AttributeReference =>
          val index = partitionColumns.indexWhere(a.name == _.name)
          BoundReference(index, partitionColumns(index).dataType, nullable = true)
      })

      val selected = partitions.filter {
        case PartitionPath(values, _) => boundPredicate(values)
      }
      logInfo {
        val total = partitions.length
        val selectedSize = selected.length
        val percentPruned = (1 - selectedSize.toDouble / total.toDouble) * 100
        s"Selected $selectedSize partitions out of $total, pruned $percentPruned% partitions."
      }

      selected
    } else {
      partitions
    }
  }

  /**
   * Contains a set of paths that are considered as the base dirs of the input datasets.
   * The partitioning discovery logic will make sure it will stop when it reaches any
   * base path.
   *
   * By default, the paths of the dataset provided by users will be base paths.
   * Below are three typical examples,
   * Case 1) `spark.read.parquet("/path/something=true/")`: the base path will be
   * `/path/something=true/`, and the returned DataFrame will not contain a column of `something`.
   * Case 2) `spark.read.parquet("/path/something=true/a.parquet")`: the base path will be
   * still `/path/something=true/`, and the returned DataFrame will also not contain a column of
   * `something`.
   * Case 3) `spark.read.parquet("/path/")`: the base path will be `/path/`, and the returned
   * DataFrame will have the column of `something`.
   *
   * Users also can override the basePath by setting `basePath` in the options to pass the new base
   * path to the data source.
   * For example, `spark.read.option("basePath", "/path/").parquet("/path/something=true/")`,
   * and the returned DataFrame will have the column of `something`.
   */
  private def basePaths: Set[Path] = {
    parameters.get(BASE_PATH_PARAM).map(new Path(_)) match {
      case Some(userDefinedBasePath) =>
        val fs = userDefinedBasePath.getFileSystem(hadoopConf)
        if (!fs.isDirectory(userDefinedBasePath)) {
          throw new IllegalArgumentException(s"Option '$BASE_PATH_PARAM' must be a directory")
        }
        Set(fs.makeQualified(userDefinedBasePath))

      case None =>
        rootPaths.map { path =>
          // Make the path qualified (consistent with listLeafFiles and listLeafFilesInParallel).
          val qualifiedPath = path.getFileSystem(hadoopConf).makeQualified(path)
          if (leafFiles.contains(qualifiedPath)) qualifiedPath.getParent else qualifiedPath }.toSet
    }
  }

  // SPARK-15895: Metadata files (e.g. Parquet summary files) and temporary files should not be
  // counted as data files, so that they shouldn't participate partition discovery.
  private def isDataPath(path: Path): Boolean = {
    val name = path.getName
    !((name.startsWith("_") && !name.contains("=")) || name.startsWith("."))
  }

  /**
   * List leaf files of given paths. This method will submit a Spark job to do parallel
   * listing whenever there is a path having more files than the parallel partition discovery
   * discovery threshold.
   *
   * This is publicly visible for testing.
   */
  def listLeafFiles(paths: Seq[Path]): mutable.LinkedHashSet[FileStatus] = {
    val output = mutable.LinkedHashSet[FileStatus]()
    val pathsToFetch = mutable.ArrayBuffer[Path]()
    for (path <- paths) {
      fileStatusCache.getLeafFiles(path) match {
        case Some(files) =>
          HiveCatalogMetrics.incrementFileCacheHits(files.length)
          output ++= files
        case None =>
          pathsToFetch += path
      }
    }
    val filter = FileInputFormat.getInputPathFilter(new JobConf(hadoopConf, this.getClass))
    val discovered = PartitioningAwareFileIndex.bulkListLeafFiles(
      pathsToFetch, hadoopConf, filter, sparkSession)
    discovered.foreach { case (path, leafFiles) =>
      HiveCatalogMetrics.incrementFilesDiscovered(leafFiles.size)
      fileStatusCache.putLeafFiles(path, leafFiles.toArray)
      output ++= leafFiles
    }
    output
  }
}

object PartitioningAwareFileIndex extends Logging {
  val BASE_PATH_PARAM = "basePath"

  /** A serializable variant of HDFS's BlockLocation. */
  private case class SerializableBlockLocation(
      names: Array[String],
      hosts: Array[String],
      offset: Long,
      length: Long)

  /** A serializable variant of HDFS's FileStatus. */
  private case class SerializableFileStatus(
      path: String,
      length: Long,
      isDir: Boolean,
      blockReplication: Short,
      blockSize: Long,
      modificationTime: Long,
      accessTime: Long,
      blockLocations: Array[SerializableBlockLocation])

  /**
   * Lists a collection of paths recursively. Picks the listing strategy adaptively depending
   * on the number of paths to list.
   *
   * This may only be called on the driver.
   *
   * @return for each input path, the set of discovered files for the path
   */
  private def bulkListLeafFiles(
      paths: Seq[Path],
      hadoopConf: Configuration,
      filter: PathFilter,
      sparkSession: SparkSession): Seq[(Path, Seq[FileStatus])] = {

    // Short-circuits parallel listing when serial listing is likely to be faster.
    if (paths.size < sparkSession.sessionState.conf.parallelPartitionDiscoveryThreshold) {
      return paths.map { path =>
        (path, listLeafFiles(path, hadoopConf, filter, Some(sparkSession)))
      }
    }

    logInfo(s"Listing leaf files and directories in parallel under: ${paths.mkString(", ")}")
    HiveCatalogMetrics.incrementParallelListingJobCount(1)

    val sparkContext = sparkSession.sparkContext
    val serializableConfiguration = new SerializableConfiguration(hadoopConf)
    val serializedPaths = paths.map(_.toString)

    // Set the number of parallelism to prevent following file listing from generating many tasks
    // in case of large #defaultParallelism.
    val numParallelism = Math.min(paths.size, 10000)

    val statusMap = sparkContext
      .parallelize(serializedPaths, numParallelism)
      .mapPartitions { pathStrings =>
        val hadoopConf = serializableConfiguration.value
        pathStrings.map(new Path(_)).toSeq.map { path =>
          (path, listLeafFiles(path, hadoopConf, filter, None))
        }.iterator
      }.map { case (path, statuses) =>
        val serializableStatuses = statuses.map { status =>
          // Turn FileStatus into SerializableFileStatus so we can send it back to the driver
          val blockLocations = status match {
            case f: LocatedFileStatus =>
              f.getBlockLocations.map { loc =>
                SerializableBlockLocation(
                  loc.getNames,
                  loc.getHosts,
                  loc.getOffset,
                  loc.getLength)
              }

            case _ =>
              Array.empty[SerializableBlockLocation]
          }

          SerializableFileStatus(
            status.getPath.toString,
            status.getLen,
            status.isDirectory,
            status.getReplication,
            status.getBlockSize,
            status.getModificationTime,
            status.getAccessTime,
            blockLocations)
        }
        (path.toString, serializableStatuses)
      }.collect()

    // turn SerializableFileStatus back to Status
    statusMap.map { case (path, serializableStatuses) =>
      val statuses = serializableStatuses.map { f =>
        val blockLocations = f.blockLocations.map { loc =>
          new BlockLocation(loc.names, loc.hosts, loc.offset, loc.length)
        }
        new LocatedFileStatus(
          new FileStatus(
            f.length, f.isDir, f.blockReplication, f.blockSize, f.modificationTime,
            new Path(f.path)),
          blockLocations)
      }
      (new Path(path), statuses)
    }
  }

  /**
   * Lists a single filesystem path recursively. If a SparkSession object is specified, this
   * function may launch Spark jobs to parallelize listing.
   *
   * If sessionOpt is None, this may be called on executors.
   *
   * @return all children of path that match the specified filter.
   */
  private def listLeafFiles(
      path: Path,
      hadoopConf: Configuration,
      filter: PathFilter,
      sessionOpt: Option[SparkSession]): Seq[FileStatus] = {
    logTrace(s"Listing $path")
    val fs = path.getFileSystem(hadoopConf)
    val name = path.getName.toLowerCase
    if (shouldFilterOut(name)) {
      Seq.empty[FileStatus]
    } else {
      // [SPARK-17599] Prevent InMemoryFileIndex from failing if path doesn't exist
      // Note that statuses only include FileStatus for the files and dirs directly under path,
      // and does not include anything else recursively.
      val statuses = try fs.listStatus(path) catch {
        case _: FileNotFoundException =>
          logWarning(s"The directory $path was not found. Was it deleted very recently?")
          Array.empty[FileStatus]
      }

      val allLeafStatuses = {
        val (dirs, topLevelFiles) = statuses.partition(_.isDirectory)
        val nestedFiles: Seq[FileStatus] = sessionOpt match {
          case Some(session) =>
            bulkListLeafFiles(dirs.map(_.getPath), hadoopConf, filter, session).flatMap(_._2)
          case _ =>
            dirs.flatMap(dir => listLeafFiles(dir.getPath, hadoopConf, filter, sessionOpt))
        }
        val allFiles = topLevelFiles ++ nestedFiles
        if (filter != null) allFiles.filter(f => filter.accept(f.getPath)) else allFiles
      }

      allLeafStatuses.filterNot(status => shouldFilterOut(status.getPath.getName)).map {
        case f: LocatedFileStatus =>
          f

        // NOTE:
        //
        // - Although S3/S3A/S3N file system can be quite slow for remote file metadata
        //   operations, calling `getFileBlockLocations` does no harm here since these file system
        //   implementations don't actually issue RPC for this method.
        //
        // - Here we are calling `getFileBlockLocations` in a sequential manner, but it should not
        //   be a big deal since we always use to `listLeafFilesInParallel` when the number of
        //   paths exceeds threshold.
        case f =>
          // The other constructor of LocatedFileStatus will call FileStatus.getPermission(),
          // which is very slow on some file system (RawLocalFileSystem, which is launch a
          // subprocess and parse the stdout).
          val locations = fs.getFileBlockLocations(f, 0, f.getLen)
          val lfs = new LocatedFileStatus(f.getLen, f.isDirectory, f.getReplication, f.getBlockSize,
            f.getModificationTime, 0, null, null, null, null, f.getPath, locations)
          if (f.isSymlink) {
            lfs.setSymlink(f.getSymlink)
          }
          lfs
      }
    }
  }

  /** Checks if we should filter out this path name. */
  def shouldFilterOut(pathName: String): Boolean = {
    // We filter everything that starts with _ and ., except _common_metadata and _metadata
    // because Parquet needs to find those metadata files from leaf files returned by this method.
    // We should refactor this logic to not mix metadata files with data files.
    ((pathName.startsWith("_") && !pathName.contains("=")) || pathName.startsWith(".")) &&
      !pathName.startsWith("_common_metadata") && !pathName.startsWith("_metadata")
  }
}
