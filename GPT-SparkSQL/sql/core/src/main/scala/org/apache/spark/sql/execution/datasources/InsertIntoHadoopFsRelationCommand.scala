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

import java.io.{BufferedReader, FileWriter, IOException, InputStreamReader}
import java.util
import java.util.{Date, UUID}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext, TaskAttemptID, TaskID, TaskType}
import org.apache.parquet.hadoop.{ParquetFileReader, ParquetFileWriter}
import org.apache.spark.{SparkContext, SparkException, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.internal.io.FileCommitProtocol.TaskCommitMessage
import org.apache.spark.internal.io.{FileCommitProtocol, SparkHadoopWriterUtils}
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.catalog.{BucketSpec, CatalogTable, CatalogTablePartition, ExternalCatalogUtils}
import org.apache.spark.sql.catalyst.catalog.CatalogTypes.TablePartitionSpec
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeSet, Cast, Concat, Expression, Literal, ScalaUDF, SortOrder, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.command._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.catalyst.util.{CaseInsensitiveMap, DateTimeUtils}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution, SortExec}
import org.apache.spark.sql.execution.datasources.FileFormatWriter.{logDebug, logError, logInfo, _}
import org.apache.spark.sql.types.StringType
import org.apache.spark.util.{SerializableConfiguration, Utils}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * A command for writing data to a [[HadoopFsRelation]].  Supports both overwriting and appending.
 * Writing to dynamic partitions is also supported.
 *
 * @param staticPartitions partial partitioning spec for write. This defines the scope of partition
 *                         overwrites: when the spec is empty, all partitions are overwritten.
 *                         When it covers a prefix of the partition keys, only partitions matching
 *                         the prefix are overwritten.
 * @param ifPartitionNotExists If true, only write if the partition does not exist.
 *                             Only valid for static partitions.
 */
case class InsertIntoHadoopFsRelationCommand(
    outputPath: Path,
    staticPartitions: TablePartitionSpec,
    ifPartitionNotExists: Boolean,
    partitionColumns: Seq[String],
    bucketSpec: Option[BucketSpec],
    fileFormat: FileFormat,
    options: Map[String, String],
    query: LogicalPlan,
    mode: SaveMode,
    catalogTable: Option[CatalogTable],
    fileIndex: Option[FileIndex])
  extends RunnableCommand {

  import org.apache.spark.sql.catalyst.catalog.ExternalCatalogUtils.escapePathName

  override protected def innerChildren: Seq[LogicalPlan] = query :: Nil

  override def run(sparkSession: SparkSession): Seq[Row] = {
    // Most formats don't do well with duplicate columns, so lets not allow that
    if (query.schema.fieldNames.length != query.schema.fieldNames.distinct.length) {
      val duplicateColumns = query.schema.fieldNames.groupBy(identity).collect {
        case (x, ys) if ys.length > 1 => "\"" + x + "\""
      }.mkString(", ")
      throw new AnalysisException(s"Duplicate column(s): $duplicateColumns found, " +
        "cannot save to file.")
    }

    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val fs = outputPath.getFileSystem(hadoopConf)
    val qualifiedOutputPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)

    val partitionsTrackedByCatalog = sparkSession.sessionState.conf.manageFilesourcePartitions &&
      catalogTable.isDefined &&
      catalogTable.get.partitionColumnNames.nonEmpty &&
      catalogTable.get.tracksPartitionsInCatalog

    var initialMatchingPartitions: Seq[TablePartitionSpec] = Nil
    var customPartitionLocations: Map[TablePartitionSpec, String] = Map.empty
    var matchingPartitions: Seq[CatalogTablePartition] = Seq.empty

    // When partitions are tracked by the catalog, compute all custom partition locations that
    // may be relevant to the insertion job.
    if (partitionsTrackedByCatalog) {
      matchingPartitions = sparkSession.sessionState.catalog.listPartitions(
        catalogTable.get.identifier, Some(staticPartitions))
      initialMatchingPartitions = matchingPartitions.map(_.spec)
      customPartitionLocations = getCustomPartitionLocations(
        fs, catalogTable.get, qualifiedOutputPath, matchingPartitions)
    }

    val pathExists = fs.exists(qualifiedOutputPath)
    // If we are appending data to an existing dir.
    val isAppend = pathExists && (mode == SaveMode.Append)

    val committer = FileCommitProtocol.instantiate(
      sparkSession.sessionState.conf.fileCommitProtocolClass,
      jobId = java.util.UUID.randomUUID().toString,
      outputPath = outputPath.toString,
      isAppend = isAppend)

    val doInsertion = (mode, pathExists) match {
      case (SaveMode.ErrorIfExists, true) =>
        throw new AnalysisException(s"path $qualifiedOutputPath already exists.")
      case (SaveMode.Overwrite, true) =>
        if (ifPartitionNotExists && matchingPartitions.nonEmpty) {
          false
        } else {
          deleteMatchingPartitions(fs, qualifiedOutputPath, customPartitionLocations, committer)
          true
        }
      case (SaveMode.Append, _) | (SaveMode.Overwrite, _) | (SaveMode.ErrorIfExists, false) =>
        true
      case (SaveMode.Ignore, exists) =>
        !exists
      case (s, exists) =>
        throw new IllegalStateException(s"unsupported save mode $s ($exists)")
    }

    if (doInsertion) {

      // Callback for updating metastore partition metadata after the insertion job completes.
      def refreshPartitionsCallback(updatedPartitions: Seq[TablePartitionSpec]): Unit = {
        if (partitionsTrackedByCatalog) {
          val newPartitions = updatedPartitions.toSet -- initialMatchingPartitions
          if (newPartitions.nonEmpty) {
            AlterTableAddPartitionCommand(
              catalogTable.get.identifier, newPartitions.toSeq.map(p => (p, None)),
              ifNotExists = true).run(sparkSession)
          }
          if (mode == SaveMode.Overwrite) {
            val deletedPartitions = initialMatchingPartitions.toSet -- updatedPartitions
            if (deletedPartitions.nonEmpty) {
              AlterTableDropPartitionCommand(
                catalogTable.get.identifier, deletedPartitions.toSeq,
                ifExists = true, purge = false,
                retainData = true /* already deleted */).run(sparkSession)
            }
          }
        }
      }

      FileFormatWriter.write(
        sparkSession = sparkSession,
        queryExecution = Dataset.ofRows(sparkSession, query).queryExecution,
        fileFormat = fileFormat,
        committer = committer,
        outputSpec = FileFormatWriter.OutputSpec(qualifiedOutputPath.toString, customPartitionLocations),
        hadoopConf = hadoopConf,
        partitionColumnNames = partitionColumns,
        bucketSpec = bucketSpec,
        refreshFunction = refreshPartitionsCallback,
        options = options)

      // refresh cached files in FileIndex
      fileIndex.foreach(_.refresh())
      // refresh data cache if table is cached
      sparkSession.catalog.refreshByPath(outputPath.toString)
    } else {
      logInfo("Skipping insertion into a relation that already exists.")
    }

    // Do merging GPT partitioned file if necessary
    var isGPTPartitioning = false
    var tblName = "none"
    var isNeedSort = false
    var DBName = "none"

    options.foreach(x =>
      if (x._1 == "GPT" && x._2 == "true") {
        isGPTPartitioning = true
      }
      else if (x._1 == "TableName") {
        tblName = x._2
      }
      else if (x._1 == "Sort" && x._2 == "true") {
        isNeedSort = true
      }
      else if (x._1 == "DBName") {
        DBName = x._2
      }
    )

    /*
    class GPTMerging(pairs: String,
                     conf: org.apache.hadoop.conf.Configuration) extends Serializable with Logging {

      val dfs: org.apache.hadoop.fs.FileSystem = FileSystem.get(conf)
      val tokens = pairs.split("|")
      var outputPath = tokens(0)
      var partitions = new scala.collection.mutable.ListBuffer[String]
      for(i <- 1 to tokens.length-1) {
        partitions += tokens(i)
      }

      def writeAsCSV(): Unit = {

        val path: Path = new Path(outputPath)
        if (dfs.exists(path)) {
          dfs.delete(path, true)
        }

        var numRowScanned = 0

        val partitionPath: Array[Path] = partitions.map(f => new Path(f)).toArray[Path]

        val dataOutputStream = dfs.create(path)
        val bw: java.io.BufferedWriter = new java.io.BufferedWriter(
          new java.io.OutputStreamWriter(dataOutputStream, "UTF-8"))

        partitions.foreach {
          r =>
            val filePath = new org.apache.hadoop.fs.Path(r)
            val status = dfs.getContentSummary(filePath)
            val fileLength = status.getLength
            if (fileLength > 0) {
              val dataInputStream = dfs.open(filePath)
              val br: java.io.BufferedReader = new java.io.BufferedReader(
                new java.io.InputStreamReader(dataInputStream, "UTF-8")
              )

              var readLine: Option[String] = Option("")
              var isEnd = false
              while (!isEnd) {
                readLine = Option(br.readLine())

                if (!readLine.isDefined) {
                  isEnd = true
                } else {
                  bw.write(readLine.get)
                  bw.newLine()
                  numRowScanned += 1
                }
              }
              br.close()
            }
        }

        bw.close
        logInfo(s"Merging GPT partitions FINISHED [num rows: " + numRowScanned + "]")
      }
    }
    */

    // CURRENT MERGING TASK (SERIAL VERSION)
    def mergeAsCSV(outputPath: String, sparkSession: SparkSession,
                   partitions: List[String]): Int = {

      val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
      val dfs = FileSystem.get(hadoopConf)

      val path: Path = new Path(outputPath) with Serializable
      if (dfs.exists(path)) {
        dfs.delete(path, true)
      }

      var numRowScanned = 0

      val partitionPath: Array[Path] = partitions.map(f => new Path(f)).toArray[Path]

      val dataOutputStream = dfs.create(path)
      val bw: java.io.BufferedWriter = new java.io.BufferedWriter(
        new java.io.OutputStreamWriter(dataOutputStream, "UTF-8") with Serializable) with Serializable

      partitions.foreach {
        r =>
          val filePath = new org.apache.hadoop.fs.Path(r)
          val status = dfs.getContentSummary(filePath)
          val fileLength = status.getLength
          if (fileLength > 0) {
            val dataInputStream = dfs.open(filePath)
            val br: java.io.BufferedReader = new java.io.BufferedReader(
              new java.io.InputStreamReader(dataInputStream, "UTF-8")
            )

            var readLine: Option[String] = Option("")
            var isEnd = false
            while (!isEnd) {
              readLine = Option(br.readLine())
              if (!readLine.isDefined) {
                isEnd = true
              } else {
                // logInfo(s"readLine: " + readLine.get)
                bw.write(readLine.get)
                bw.newLine()
                numRowScanned += 1
              }
            }
            br.close()
          }
      }

      bw.close
      logInfo(s"Merging GPT partitions FINISHED [num rows: " + numRowScanned + "]")
      numRowScanned
    }

    if (isGPTPartitioning) {

      logInfo(s"Do merging GPT partitioned files!")
      logInfo(s"Destination: " + qualifiedOutputPath.toString)
      sparkSession.sessionState.conf.setConfString("GPTMergerJob", "true")

      val iscatalogTableDefined = catalogTable.isDefined
      if (iscatalogTableDefined) {
        val isBucketSpecDefined = catalogTable.get.bucketSpec.isDefined
        if (isBucketSpecDefined) {
          catalogTable.get.bucketSpec.get.setGPTPartitioned()
        }
      }

      val isSerialMerger = if (sparkSession.sparkContext.conf.get("spark.GPT.ParallelMerge") == "true") false else true

      if (isSerialMerger) {
        var tableName = ""

        val tmpPathStr = "/GPT_tmp/"
        val tmpPath = new Path(tmpPathStr)

        if (fs.exists(tmpPath)) {
          fs.delete(tmpPath, true)
        }
        fs.mkdirs(tmpPath)

        val status = fs.listStatus {
          new org.apache.hadoop.fs.Path(qualifiedOutputPath.toString)
        }

        var partitionMap = scala.collection.mutable.Map[String, scala.collection.mutable.ListBuffer[String]]()

        var isCSV = false
        var isParquet = false

        // hdfs://10.150.20.24:8021/tpcds/test_CB/GPT-0-test_text-0-00000.csv
        for (partition <- status) {
          var partitionFile = partition.getPath.toString()

          if (partitionFile.contains("GPT")) {

            // logInfo(s"test! : " + partitionFile)

            if (partitionFile.contains("csv") || partitionFile.contains("CSV")) {
              isCSV = true
              isParquet = false
            } else {
              isCSV = false
              isParquet = true
            }

            val tokens = partitionFile.split("-")
            val partitionIdentifier = tokens(3)
            val partitionID = tokens(4)
            val partitionKey = partitionIdentifier + "-" + partitionID

            if (tableName.isEmpty) {
              tableName = tokens(2)
            }

            if (partitionMap.contains(partitionKey)) {
              partitionMap(partitionKey) += partitionFile
            } else {
              var partitionList = scala.collection.mutable.ListBuffer[String]()
              partitionList += partitionFile
              partitionMap(partitionKey) = partitionList
            }
          }
        }

        var partitionOutputMap = scala.collection.mutable.Map[String, String]()

        partitionMap.foreach { m =>
          var savePath = tmpPathStr + ("GPT-" + tableName + "-" + m._1)
          partitionOutputMap(m._1) = savePath
        }

        /*
        val mergerTmpSrc = "/Merger_Tmp"
        val mergerPath: Path = new Path(mergerTmpSrc) with Serializable
        if (fs.exists(mergerPath)) {
          fs.delete(mergerPath, true)
        }
        fs.mkdirs(mergerPath)

        partitionMap.foreach { p =>
          val mergePath = ("GPT-" + tableName + "-" + p._1)
          var inputFilePaths = ""
          p._2.foreach( c => inputFilePaths += (c + "|"))
          var mergerTaskInfo = mergePath + "|" + inputFilePaths
          val infoFilePath = mergerTmpSrc + "/" + mergePath
          val bw = new java.io.BufferedWriter(new java.io.OutputStreamWriter(fs.create(new Path(infoFilePath)), "UTF-8"))
          bw.write(mergerTaskInfo)
          bw.close()
        }
        */

        // make GPT file: GPT-{table name}-{partition index}-{partition ID}

        var totalRows = 0

        if (isCSV) {

          partitionMap.foreach { m =>
            logInfo(s"Merging GPT partitions [CSV]: " + m._1.toString)
            totalRows += mergeAsCSV(partitionOutputMap(m._1), sparkSession, m._2.toList)
          }


        } else {
          var parquetDFs = scala.collection.mutable.ListBuffer[org.apache.spark.sql.DataFrame]()
          val hadoopConf = sparkSession.sparkContext.hadoopConfiguration

          partitionMap.foreach { m =>
            logInfo(s"Merging GPT partitions [Parquet]: " + m._1.toString)

            var parquetFile = ParquetFileReader.open(hadoopConf, new Path(m._2(0)))
            val fileMetaData = parquetFile.getFooter.getFileMetaData.getSchema()

            val writer = new ParquetFileWriter(hadoopConf, fileMetaData,
              new Path(partitionOutputMap(m._1)))

            writer.start()
            m._2.foreach { m =>
              parquetFile = ParquetFileReader.open(hadoopConf, new Path(m))
              writer.appendFile(hadoopConf, new Path(m))
              parquetFile.close()
            }
            writer.end(new util.HashMap[String, String]())
          }

          logInfo(s"Merging GPT Partitions: FINISHED!")
        }

        // finalyzing
        // moving /GPT_tmp/* to /{qualifiedOutputPath}
        fs.delete(qualifiedOutputPath, true)
        fs.rename(tmpPath, qualifiedOutputPath)
      }
    } else {
      sparkSession.sessionState.conf.setConfString("GPTMergerJob", "false")
    }
    Seq.empty[Row]
  }

  def fileRemover(fs: FileSystem, partitions: scala.collection.mutable.ListBuffer[String]): Unit = {
    val partitionPath: Array[Path] = partitions.map(f => new Path(f)).toArray[Path]
    partitionPath.foreach(f => fs.delete(f, true))
  }

  /**
   * Deletes all partition files that match the specified static prefix. Partitions with custom
   * locations are also cleared based on the custom locations map given to this class.
   */
  private def deleteMatchingPartitions(
      fs: FileSystem,
      qualifiedOutputPath: Path,
      customPartitionLocations: Map[TablePartitionSpec, String],
      committer: FileCommitProtocol): Unit = {
    val staticPartitionPrefix = if (staticPartitions.nonEmpty) {
      "/" + partitionColumns.flatMap { col =>
        staticPartitions.get(col) match {
          case Some(value) =>
            Some(escapePathName(col) + "=" + escapePathName(value))
          case None =>
            None
        }
      }.mkString("/")
    } else {
      ""
    }
    // first clear the path determined by the static partition keys (e.g. /table/foo=1)
    val staticPrefixPath = qualifiedOutputPath.suffix(staticPartitionPrefix)
    if (fs.exists(staticPrefixPath) && !committer.deleteWithJob(fs, staticPrefixPath, true)) {
      throw new IOException(s"Unable to clear output " +
        s"directory $staticPrefixPath prior to writing to it")
    }
    // now clear all custom partition locations (e.g. /custom/dir/where/foo=2/bar=4)
    for ((spec, customLoc) <- customPartitionLocations) {
      assert(
        (staticPartitions.toSet -- spec).isEmpty,
        "Custom partition location did not match static partitioning keys")
      val path = new Path(customLoc)
      if (fs.exists(path) && !committer.deleteWithJob(fs, path, true)) {
        throw new IOException(s"Unable to clear partition " +
          s"directory $path prior to writing to it")
      }
    }
  }

  /**
   * Given a set of input partitions, returns those that have locations that differ from the
   * Hive default (e.g. /k1=v1/k2=v2). These partitions were manually assigned locations by
   * the user.
   *
   * @return a mapping from partition specs to their custom locations
   */
  private def getCustomPartitionLocations(
      fs: FileSystem,
      table: CatalogTable,
      qualifiedOutputPath: Path,
      partitions: Seq[CatalogTablePartition]): Map[TablePartitionSpec, String] = {
    partitions.flatMap { p =>
      val defaultLocation = qualifiedOutputPath.suffix(
        "/" + PartitioningUtils.getPathFragment(p.spec, table.partitionSchema)).toString
      val catalogLocation = new Path(p.location).makeQualified(
        fs.getUri, fs.getWorkingDirectory).toString
      if (catalogLocation != defaultLocation) {
        Some(p.spec -> catalogLocation)
      } else {
        None
      }
    }.toMap
  }
}
