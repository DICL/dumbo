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

package org.apache.spark.sql.execution.command

import scala.util.control.NonFatal

import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{AnalysisException, Dataset, Row, SparkSession}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis.EliminateSubqueryAliases
import org.apache.spark.sql.catalyst.catalog.{CatalogRelation, CatalogTable}
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.internal.SessionState


/**
 * Analyzes the given table to generate statistics, which will be used in query optimizations.
 */
case class AnalyzeTableCommand(
    tableIdent: TableIdentifier,
    noscan: Boolean = true) extends RunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    val sessionState = sparkSession.sessionState
    val db = tableIdent.database.getOrElse(sessionState.catalog.getCurrentDatabase)
    val tableIdentWithDB = TableIdentifier(tableIdent.table, Some(db))
    val relation = EliminateSubqueryAliases(sessionState.catalog.lookupRelation(tableIdentWithDB))

    relation match {
      case relation: CatalogRelation =>
        updateTableStats(relation.catalogTable,
          AnalyzeTableCommand.calculateTotalSize(sessionState, relation.catalogTable))

      // data source tables have been converted into LogicalRelations
      case logicalRel: LogicalRelation if logicalRel.catalogTable.isDefined =>
        updateTableStats(logicalRel.catalogTable.get,
          AnalyzeTableCommand.calculateTotalSize(sessionState, logicalRel.catalogTable.get))

      case otherRelation =>
        throw new AnalysisException("ANALYZE TABLE is not supported for " +
          s"${otherRelation.nodeName}.")
    }

    def updateTableStats(catalogTable: CatalogTable, newTotalSize: Long): Unit = {
      val oldTotalSize = catalogTable.stats.map(_.sizeInBytes.toLong).getOrElse(0L)
      val oldRowCount = catalogTable.stats.flatMap(_.rowCount.map(_.toLong)).getOrElse(-1L)
      var newStats: Option[Statistics] = None
      if (newTotalSize > 0 && newTotalSize != oldTotalSize) {
        newStats = Some(Statistics(sizeInBytes = newTotalSize))
      }
      // We only set rowCount when noscan is false, because otherwise:
      // 1. when total size is not changed, we don't need to alter the table;
      // 2. when total size is changed, `oldRowCount` becomes invalid.
      // This is to make sure that we only record the right statistics.
      if (!noscan) {
        val newRowCount = Dataset.ofRows(sparkSession, relation).count()
        if (newRowCount >= 0 && newRowCount != oldRowCount) {
          newStats = if (newStats.isDefined) {
            newStats.map(_.copy(rowCount = Some(BigInt(newRowCount))))
          } else {
            Some(Statistics(sizeInBytes = oldTotalSize, rowCount = Some(BigInt(newRowCount))))
          }
        }
      }
      // Update the metastore if the above statistics of the table are different from those
      // recorded in the metastore.
      if (newStats.isDefined) {
        sessionState.catalog.alterTable(catalogTable.copy(stats = newStats))
        // Refresh the cached data source table in the catalog.
        sessionState.catalog.refreshTable(tableIdentWithDB)
      }
    }

    Seq.empty[Row]
  }
}

object AnalyzeTableCommand extends Logging {

  def calculateTotalSize(sessionState: SessionState, catalogTable: CatalogTable): Long = {
    // This method is mainly based on
    // org.apache.hadoop.hive.ql.stats.StatsUtils.getFileSizeForTable(HiveConf, Table)
    // in Hive 0.13 (except that we do not use fs.getContentSummary).
    // TODO: Generalize statistics collection.
    // TODO: Why fs.getContentSummary returns wrong size on Jenkins?
    // Can we use fs.getContentSummary in future?
    // Seems fs.getContentSummary returns wrong table size on Jenkins. So we use
    // countFileSize to count the table size.
    val stagingDir = sessionState.conf.getConfString("hive.exec.stagingdir", ".hive-staging")

    def calculateTableSize(fs: FileSystem, path: Path): Long = {
      val fileStatus = fs.getFileStatus(path)
      val size = if (fileStatus.isDirectory) {
        fs.listStatus(path)
          .map { status =>
            if (!status.getPath.getName.startsWith(stagingDir)) {
              calculateTableSize(fs, status.getPath)
            } else {
              0L
            }
          }.sum
      } else {
        fileStatus.getLen
      }

      size
    }

    catalogTable.storage.locationUri.map { p =>
      val path = new Path(p)
      try {
        val fs = path.getFileSystem(sessionState.newHadoopConf())
        calculateTableSize(fs, path)
      } catch {
        case NonFatal(e) =>
          logWarning(
            s"Failed to get the size of table ${catalogTable.identifier.table} in the " +
              s"database ${catalogTable.identifier.database} because of ${e.toString}", e)
          0L
      }
    }.getOrElse(0L)
  }
}
