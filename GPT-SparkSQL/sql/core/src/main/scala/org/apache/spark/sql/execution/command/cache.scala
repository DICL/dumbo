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

import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

case class CacheTableCommand(
    tableIdent: TableIdentifier,
    plan: Option[LogicalPlan],
    isLazy: Boolean) extends RunnableCommand {
  require(plan.isEmpty || tableIdent.database.isEmpty,
    "Database name is not allowed in CACHE TABLE AS SELECT")

  override protected def innerChildren: Seq[QueryPlan[_]] = {
    plan.toSeq
  }

  override def run(sparkSession: SparkSession): Seq[Row] = {

    plan.foreach { logicalPlan =>
      Dataset.ofRows(sparkSession, logicalPlan).createTempView(tableIdent.quotedString)
    }
    sparkSession.sharedState.cacheManager.setTableIndentForCachingGPT(tableIdent)
    sparkSession.catalog.cacheTable(tableIdent.quotedString)

    if(sparkSession.conf.get("GPTTableCaching") == "true") {
      sparkSession.sharedState.cacheManager.makeGPTCacheStatusDefault()
    }

    if (!isLazy && sparkSession.conf.get("GPTTableCaching") == "false") {
      // Performs eager caching
      sparkSession.table(tableIdent).count()
    }

    Seq.empty[Row]
  }

    /*

    // FOR CACHING GPT PARTITIONED TABLE
    logInfo("Run CacheTableCommand for table: " + tableIdent.quotedString + "(plan argument: " + plan.isDefined + ")")
    if(plan.isDefined) {
      logInfo("\t\t plan: " + plan.get.numberedTreeString)
    }

    val logicalPlan = sparkSession.table(tableIdent.quotedString).logicalPlan

    logInfo("call catalog.cacheTable: " + logicalPlan.nodeName)
    logInfo("\t\t plan: " + logicalPlan.numberedTreeString)

    val optimizedPlan: LogicalPlan = sparkSession.sessionState.optimizer.execute(logicalPlan)

    def findScanOP(p: SparkPlan): SparkPlan = {

      var curChildOP = p

      while (!curChildOP.isInstanceOf[FileSourceScanExec] && !curChildOP.children.isEmpty) {
        curChildOP = curChildOP.children(0)
      }
      curChildOP
    }

    def isGPTPartitiondTable(p: FileSourceScanExec): Boolean = {
      val planTree = p.numberedTreeString
      logDebug(s"isGPTPartitiondTable: " + planTree)
      var isGPTPartitionedTable = false

      val token = planTree.split("GPT:")(1).split(",")
      val GPTConf = token(0).trim
      if (GPTConf == "true") {
        isGPTPartitionedTable = true
      }
      isGPTPartitionedTable
    }

    val planner = sparkSession.sessionState.planner

    val sparkPlan: SparkPlan = {
      SparkSession.setActiveSession(sparkSession)
      // TODO: We use next(), i.e. take the first plan returned by the planner, here for now,
      //       but we will implement to choose the best plan.
      planner.plan(ReturnAnswer(optimizedPlan)).next()
    }

    val ScanOP = findScanOP(sparkPlan)
    var GPTPartitionedTable = false
    if(ScanOP.isInstanceOf[FileSourceScanExec] &&
    isGPTPartitiondTable(ScanOP.asInstanceOf[FileSourceScanExec])) {

      ScanOP.asInstanceOf[FileSourceScanExec].setAsOuterJoinTable()
      GPTPartitionedTable = true
      logInfo("Caching GPT Partitioned table!")
    }

    if (GPTPartitionedTable) {

      // Caching GPT Partitioned table needs logic
      // For each partitioning column used for GPT partitioning
      // we have to make corresponding full table scan plan
      // including outer join handling
      val tblDF = sparkSession.table(tableIdent.quotedString)
      val tblExecPlan = sparkSession.sessionState.executePlan(tblDF.logicalPlan).executedPlan

      val scanOP = findScanOP(tblExecPlan).asInstanceOf[FileSourceScanExec]

      // "00 SubqueryAlias web_returns"
      val tblName = tblDF.logicalPlan.numberedTreeString.split(" ")(2).split("\n")(0)
      val partitioningColumns = scanOP.relation.bucketSpec.get.bucketColumnNames
      logInfo("Caching table [" + tblName + "] with partitioning columns: " + partitioningColumns)

      var colIdx = 0

      partitioningColumns.reverse.foreach{ c =>
        sparkSession.sharedState.cacheManager.joinColToIdxMap += (c -> colIdx)
        colIdx += 1
      }

      scanOP.setAsOuterJoinTable()

      partitioningColumns.foreach{ col =>

        val tblName = tableIdent.table
        val scanSQL = s"select * from $tblName where $col is null or $col is not null"
        val logicalPlan = sparkSession.sql(scanSQL).logicalPlan
        val execPlan = findScanOP(sparkSession.sql(scanSQL).queryExecution.executedPlan)

        val bitIdx = sparkSession.sharedState.cacheManager.joinColToIdxMap.get(col).get
        val newTblName = tblName + "_" + bitIdx
        logInfo("Execution Plan for creating DF for table [" + tblName + "]\n" + execPlan.numberedTreeString)
        sparkSession.sharedState.cacheManager.cacheQueryForGPT(execPlan, logicalPlan, sparkSession, Option(newTblName))

      }

      // sparkSession.sharedState.cacheManager.cacheQueryForGPT()

      // scan plan for web_returns_0 : for wr_item_sk
      // scan plan for web_returns_1 : for wr_returned_date_sk

      /*
        def cacheQuery(
      query: Dataset[_],
      tableName: Option[String] = None,
      storageLevel: StorageLevel = MEMORY_AND_DISK): Unit = writeLock {

        val planToCache = query.logicalPlan
        if (lookupCachedData(planToCache).nonEmpty) {
          logWarning("Asked to cache already cached data.")
        } else {
          val sparkSession = query.sparkSession
          cachedData.add(CachedData(
            planToCache,
            InMemoryRelation(
              sparkSession.sessionState.conf.useCompression,
              sparkSession.sessionState.conf.columnBatchSize,
              storageLevel,
              sparkSession.sessionState.executePlan(planToCache).executedPlan,
              tableName)))
          }
      }
      **/

      // sparkSession.sharedState.cacheManager.cacheQuery(sparkSession.table(tableName), Some(tableName))
      // sparkSession.catalog.cacheTable(tableIdent.quotedString)

      if (!isLazy) {
        // Performs eager caching
        sparkSession.table(tableIdent).count()
      }


      // rule
      // to make table name for each partitioning column
      // we use column index of each partitioning column
      // e.g.,  ss_sold_date_sk ==> idx 1
      //        ss_item_sk      ==> idx 0
      // then, store_sales partitioned by ss_sold_date_sk has a table name as store_sales_1
      // similarly, store_sales_0 for the table partitioned by ss_item_sk
      // reading such cached table carefully considers this!
      // e.g., decides which cached table to scan during runtime!

    } else {

      plan.foreach { logicalPlan => {
        Dataset.ofRows(sparkSession, logicalPlan).createTempView(tableIdent.quotedString)
        logInfo("createTempView for table [" + tableIdent.quotedString + "] and its plan: " + logicalPlan.numberedTreeString)
        }
      }

      sparkSession.catalog.cacheTable(tableIdent.quotedString)

      if (!isLazy) {
        // Performs eager caching
        sparkSession.table(tableIdent).count()
      }
    }

    Seq.empty[Row]
  }
  */
}


case class UncacheTableCommand(
    tableIdent: TableIdentifier,
    ifExists: Boolean) extends RunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    val tableId = tableIdent.quotedString
    try {
      sparkSession.catalog.uncacheTable(tableId)
    } catch {
      case _: NoSuchTableException if ifExists => // don't throw
    }
    Seq.empty[Row]
  }
}

/**
 * Clear all cached data from the in-memory cache.
 */
case object ClearCacheCommand extends RunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {
    sparkSession.catalog.clearCache()
    Seq.empty[Row]
  }
}
