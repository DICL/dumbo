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

import org.apache.hadoop.fs.Path

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.catalog.CatalogTable
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.StructType


/**
 * A [[FileIndex]] for a metastore catalog table.
 *
 * @param sparkSession a [[SparkSession]]
 * @param table the metadata of the table
 * @param sizeInBytes the table's data size in bytes
 */
class CatalogFileIndex(
    sparkSession: SparkSession,
    val table: CatalogTable,
    override val sizeInBytes: Long) extends FileIndex {

  protected val hadoopConf = sparkSession.sessionState.newHadoopConf

  private val fileStatusCache = FileStatusCache.newCache(sparkSession)

  assert(table.identifier.database.isDefined,
    "The table identifier must be qualified in CatalogFileIndex")

  private val baseLocation = table.storage.locationUri

  override def partitionSchema: StructType = table.partitionSchema

  override def rootPaths: Seq[Path] = baseLocation.map(new Path(_)).toSeq

  override def listFiles(filters: Seq[Expression]): Seq[PartitionDirectory] = {
    filterPartitions(filters).listFiles(Nil)
  }

  override def refresh(): Unit = fileStatusCache.invalidateAll()

  /**
   * Returns a [[InMemoryFileIndex]] for this table restricted to the subset of partitions
   * specified by the given partition-pruning filters.
   *
   * @param filters partition-pruning filters
   */
  def filterPartitions(filters: Seq[Expression]): InMemoryFileIndex = {
    if (table.partitionColumnNames.nonEmpty) {
      val selectedPartitions = sparkSession.sessionState.catalog.listPartitionsByFilter(
        table.identifier, filters)
      val partitions = selectedPartitions.map { p =>
        val path = new Path(p.location)
        val fs = path.getFileSystem(hadoopConf)
        PartitionPath(
          p.toRow(partitionSchema), path.makeQualified(fs.getUri, fs.getWorkingDirectory))
      }
      val partitionSpec = PartitionSpec(partitionSchema, partitions)
      new PrunedInMemoryFileIndex(
        sparkSession, new Path(baseLocation.get), fileStatusCache, partitionSpec)
    } else {
      new InMemoryFileIndex(sparkSession, rootPaths, table.storage.properties, None)
    }
  }

  override def inputFiles: Array[String] = filterPartitions(Nil).inputFiles

  // `CatalogFileIndex` may be a member of `HadoopFsRelation`, `HadoopFsRelation` may be a member
  // of `LogicalRelation`, and `LogicalRelation` may be used as the cache key. So we need to
  // implement `equals` and `hashCode` here, to make it work with cache lookup.
  override def equals(o: Any): Boolean = o match {
    case other: CatalogFileIndex => this.table.identifier == other.table.identifier
    case _ => false
  }

  override def hashCode(): Int = table.identifier.hashCode()
}

/**
 * An override of the standard HDFS listing based catalog, that overrides the partition spec with
 * the information from the metastore.
 *
 * @param tableBasePath The default base path of the Hive metastore table
 * @param partitionSpec The partition specifications from Hive metastore
 */
private class PrunedInMemoryFileIndex(
    sparkSession: SparkSession,
    tableBasePath: Path,
    fileStatusCache: FileStatusCache,
    override val partitionSpec: PartitionSpec)
  extends InMemoryFileIndex(
    sparkSession,
    partitionSpec.partitions.map(_.path),
    Map.empty,
    Some(partitionSpec.partitionColumns),
    fileStatusCache)
