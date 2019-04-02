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

package org.apache.spark.sql.execution

import java.util.concurrent.locks.ReentrantReadWriteLock

import scala.collection.JavaConverters._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.SubqueryExpression
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, ReturnAnswer}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.execution.columnar.InMemoryRelation
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.storage.StorageLevel
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK

import scala.collection.mutable

/** Holds a cached logical plan and its data */
case class CachedData(plan: LogicalPlan, cachedRepresentation: InMemoryRelation)


/**
 * Provides support in a SQLContext for caching query results and automatically using these cached
 * results when subsequent queries are executed.  Data is cached using byte buffers stored in an
 * InMemoryRelation.  This relation is automatically substituted query plans that return the
 * `sameResult` as the originally cached query.
 *
 * Internal to Spark SQL.
 */
class CacheManager extends Logging {

  @transient
  private val cachedData = new java.util.LinkedList[CachedData]

  // for GPT implementation
  val cachedTableForGPT = new mutable.HashSet[String]()
  val cachedDataForGPT = new mutable.HashMap[(String,String), CachedData]()
  val cachedPlanForGPT = new mutable.HashMap[(String,String), LogicalPlan]()
  val cachedInMemRelationForGPT = new mutable.HashMap[(String,String), InMemoryRelation]()

  var joinColToIdxMap : collection.mutable.HashMap[String, Int] = new collection.mutable.HashMap[String, Int]()
  var joinColToTblNameMap : collection.mutable.HashMap[String, String] = new collection.mutable.HashMap[String, String]()

  var bitIdxForRetrievingLogicalPlan = -1
  def setBitIdxForRetrievingLogicalPlan(idx: Int) = {bitIdxForRetrievingLogicalPlan = idx}
  def getBitIdxForRetrievingLogicalPlan() : Int = {bitIdxForRetrievingLogicalPlan}

  var TableIndentForCachingGPT : TableIdentifier = null
  def setTableIndentForCachingGPT(tblIdent : TableIdentifier) = {TableIndentForCachingGPT = tblIdent}
  def getTableIndentForCachingGPT() : TableIdentifier = {TableIndentForCachingGPT}

  var isTaskForCachingGPTable = false
  def setTaskForCachingGPTable() = {isTaskForCachingGPTable = true}
  def getTaskForCachingGPTable() : Boolean = {isTaskForCachingGPTable}

  var TableNameForCachingGPT = "None"
  def setTableNameForCachingGPT(name: String) = {TableNameForCachingGPT = name}
  def getTableNameForCachingGPT() : String = {TableNameForCachingGPT}

  var requestedCacheForGPTTable = "None"
  def setrequestedCacheForGPTTable(name: String) = {requestedCacheForGPTTable = name}
  def getrequestedCacheForGPTTable() : String = {requestedCacheForGPTTable}


  def makeGPTCacheStatusDefault() : Unit = {
    isTaskForCachingGPTable = false
    TableIndentForCachingGPT = null
    requestedCacheForGPTTable = "None"

    // bitIdxForRetrievingLogicalPlan = -1
    // TableNameForCachingGPT = "None"
  }

  def findScanOP(p: SparkPlan): SparkPlan = {

    var curChildOP = p

    while (!curChildOP.isInstanceOf[FileSourceScanExec] && !curChildOP.children.isEmpty) {
      curChildOP = curChildOP.children(0)
    }
    curChildOP
  }

  def isScanOpWithGPTPartitionedTable(ScanOP: SparkPlan) : Boolean = {
    if(ScanOP.isInstanceOf[FileSourceScanExec] &&
      isGPTPartitiondTable(ScanOP.asInstanceOf[FileSourceScanExec])) {
      true
    } else {
      false
    }
  }

  def isGPTPartitiondTable(p: FileSourceScanExec): Boolean = {
    val planTree = p.numberedTreeString
    // logDebug(s"isGPTPartitiondTable: " + planTree)
    var isGPTPartitionedTable = false

    val token = planTree.split("GPT:")(1).split(",")
    val GPTConf = token(0).trim
    if (GPTConf == "true") {
      isGPTPartitionedTable = true
    }
    isGPTPartitionedTable
  }

  @transient
  private val cacheLock = new ReentrantReadWriteLock

  /** Acquires a read lock on the cache for the duration of `f`. */
  private def readLock[A](f: => A): A = {
    val lock = cacheLock.readLock()
    lock.lock()
    try f finally {
      lock.unlock()
    }
  }

  /** Acquires a write lock on the cache for the duration of `f`. */
  private def writeLock[A](f: => A): A = {
    val lock = cacheLock.writeLock()
    lock.lock()
    try f finally {
      lock.unlock()
    }
  }

  /** Clears all cached tables. */
  def clearCache(): Unit = writeLock {
    cachedData.asScala.foreach(_.cachedRepresentation.cachedColumnBuffers.unpersist())
    cachedData.clear()
  }

  /** Checks if the cache is empty. */
  def isEmpty: Boolean = readLock {
    cachedData.isEmpty
  }

  /**
   * Caches the data produced by the logical representation of the given [[Dataset]].
   * Unlike `RDD.cache()`, the default storage level is set to be `MEMORY_AND_DISK` because
   * recomputing the in-memory columnar representation of the underlying table is expensive.
   */
  def cacheQuery(
      query: Dataset[_],
      tableName: Option[String] = None,
      storageLevel: StorageLevel = MEMORY_AND_DISK): Unit = writeLock {

    val planToCache = query.logicalPlan
    val sparkSession = query.sparkSession

    if (lookupCachedData(planToCache).nonEmpty) {
      logWarning("Asked to cache already cached data.")

    } else {

      // logInfo("Adding cached table info for table [" + tableName.get + "]")

      val logicalPlan = query.logicalPlan
      val optimizedPlan: LogicalPlan = sparkSession.sessionState.optimizer.execute(logicalPlan)
      val planner = sparkSession.sessionState.planner
      val sparkPlan: SparkPlan = {
        SparkSession.setActiveSession(sparkSession)
        planner.plan(ReturnAnswer(optimizedPlan)).next()
      }

      val ScanOP = findScanOP(sparkPlan)
      if(ScanOP.isInstanceOf[FileSourceScanExec] &&
        isGPTPartitiondTable(ScanOP.asInstanceOf[FileSourceScanExec])) {
        isTaskForCachingGPTable = true
        // logInfo("Caching GPT Partitioned table!")
      }

      if(isTaskForCachingGPTable) {
        // logInfo("Adding cached GPT table info for table [" + tableName.get + "]")

        val scanOP = ScanOP.asInstanceOf[FileSourceScanExec]
        val GPTPartitioningCols = scanOP.relation.bucketSpec.get.bucketColumnNames
        // logInfo("GPT cols: " + GPTPartitioningCols)
        var isAlreadyCached = false;
        val tblName = scanOP.tblName
        if(!cachedTableForGPT.contains(tblName)) {
          isTaskForCachingGPTable = true
          (0 until GPTPartitioningCols.size).foreach { bitIdx =>
            setTableNameForCachingGPT(tblName)
            val cachedName = Option(tblName + "_" + bitIdx)
            setBitIdxForRetrievingLogicalPlan(bitIdx)
            setTaskForCachingGPTable()
            val GPTTableScanLogicalPlan = sparkSession.table(tblName).logicalPlan
            val GPTTableScanPhysicalPlan = sparkSession.sessionState.executePlan(GPTTableScanLogicalPlan).executedPlan

            val tmpInMemRel = InMemoryRelation(
              sparkSession.sessionState.conf.useCompression,
              sparkSession.sessionState.conf.columnBatchSize,
              storageLevel,
              GPTTableScanPhysicalPlan,
              cachedName)

            val inMemLogicalPlan = GPTTableScanLogicalPlan transformDown {
              case currentFragment => {
                tmpInMemRel.withOutput(currentFragment.output)
              }
            }

            // logInfo("GPTTableScanLogicalPlan\n\n" + GPTTableScanLogicalPlan.numberedTreeString)

            lazy val inMemOptimizedPlan: LogicalPlan = sparkSession.sessionState.optimizer.execute(inMemLogicalPlan)

            lazy val inMemSparkPlan: SparkPlan = {
              SparkSession.setActiveSession(sparkSession)
              planner.plan(ReturnAnswer(inMemOptimizedPlan)).next()
            }

            // logInfo("inMemSparkPlan\n\n" + inMemSparkPlan.numberedTreeString)

            // logInfo("Temporary tmpInMemRel: \n\n" + tmpInMemRel.numberedTreeString)
            // logInfo("Temporary inMemLogicalPlan: \n\n" + inMemLogicalPlan.numberedTreeString)
            // logInfo("Temporary inMemOptimizedPlan: \n\n" + inMemOptimizedPlan.numberedTreeString)
            // logInfo("Temporary InMemSparkPlan: \n\n" + inMemSparkPlan.numberedTreeString)

            val inMemRel = InMemoryRelation(
              sparkSession.sessionState.conf.useCompression,
              sparkSession.sessionState.conf.columnBatchSize,
              storageLevel,
              GPTTableScanPhysicalPlan,
              cachedName)

            val cached = CachedData(GPTTableScanLogicalPlan, inMemRel)
            cachedDataForGPT.put((tblName, bitIdx.toString), cached)
            cachedPlanForGPT.put((tblName, bitIdx.toString), GPTTableScanLogicalPlan)
            cachedInMemRelationForGPT.put((tblName, bitIdx.toString), inMemRel)
            cachedTableForGPT.add(tblName);

            val ret = sparkSession.table(getTableIndentForCachingGPT).count()
            //val ret = inMemSparkPlan.execute().count()

            // logInfo("Scanning " + ret + " tuples!")
          }
          makeGPTCacheStatusDefault()
          sparkSession.conf.set("GPTTableCaching", "true") // prevent redundant early caching
          isTaskForCachingGPTable = false
        } else {
          logWarning("Asked to cache already cached data.")
        }
        /*
        val fakePlan = sparkSession.table(tblName).toDF().logicalPlan
        val fakeInMemRel = InMemoryRelation(
          sparkSession.sessionState.conf.useCompression,
          sparkSession.sessionState.conf.columnBatchSize,
          storageLevel,
          sparkSession.sessionState.executePlan(fakePlan).executedPlan,
          Option(tblName))
        val fakeCached = CachedData(fakePlan, fakeInMemRel)
        logInfo("Add fake Cached: " + tblName)
        cachedData.add(fakeCached)
        */
      } else {

        var inMemRel = InMemoryRelation(
          sparkSession.sessionState.conf.useCompression,
          sparkSession.sessionState.conf.columnBatchSize,
          storageLevel,
          sparkSession.sessionState.executePlan(planToCache).executedPlan,
          tableName)

        val cached = CachedData(planToCache,inMemRel)
        cachedData.add(cached)
        sparkSession.conf.set("GPTTableCaching", "false")
        // logInfo("[ADD NORMAL CACHE QUERY] PlanToCache:\n" + planToCache.numberedTreeString)
        // logInfo("[ADD NORMAL CACHE QUERY] ExecutedPlan for PlanToCache:\n" + sparkSession.sessionState.executePlan(planToCache).executedPlan)
        // logInfo("[ADD NORMAL CACHE QUERY] CACHED REPRESENTATION:\n" + cached.cachedRepresentation.numberedTreeString)
      }
    }
  }

  /*
  def cacheQueryForGPT(
                  newExecPlan: SparkPlan,
                  logicalPlan: LogicalPlan,
                  sparkSession: SparkSession,
                  tableName: Option[String] = None,
                  storageLevel: StorageLevel = MEMORY_AND_DISK): Unit = writeLock {
    val planToCache = logicalPlan

    if (cachedDataForGPT.contains(tableName.get)) {
      logWarning("Asked to cache already cached data: [" + tableName.get + "]")

    } else {

      val inMemRel = InMemoryRelation(
        sparkSession.sessionState.conf.useCompression,
        sparkSession.sessionState.conf.columnBatchSize,
        storageLevel,
        newExecPlan,
        tableName)

      val cached = CachedData(planToCache,inMemRel)
      cachedData.add(cached)
      logInfo("Caching QueryForGPT: table [" + tableName.get + "]")
      logInfo("Adding cacheQueryForGPT: table [" + tableName.get + "]")

      cachedDataForGPT.put(tableName.get, cached)
    }
  }
  */

  /**
   * Un-cache all the cache entries that refer to the given plan.
   */
  def uncacheQuery(query: Dataset[_], blocking: Boolean = true): Unit = writeLock {
    uncacheQuery(query.sparkSession, query.logicalPlan, blocking)
  }

  /**
   * Un-cache all the cache entries that refer to the given plan.
   */
  def uncacheQuery(spark: SparkSession, plan: LogicalPlan, blocking: Boolean): Unit = writeLock {
    val it = cachedData.iterator()
    while (it.hasNext) {
      val cd = it.next()
      if (cd.plan.find(_.sameResult(plan)).isDefined) {
        cd.cachedRepresentation.cachedColumnBuffers.unpersist(blocking)
        it.remove()
      }
    }
  }

  /**
   * Tries to re-cache all the cache entries that refer to the given plan.
   */
  def recacheByPlan(spark: SparkSession, plan: LogicalPlan): Unit = writeLock {
    recacheByCondition(spark, _.find(_.sameResult(plan)).isDefined)
  }

  private def recacheByCondition(spark: SparkSession, condition: LogicalPlan => Boolean): Unit = {
    val it = cachedData.iterator()
    val needToRecache = scala.collection.mutable.ArrayBuffer.empty[CachedData]
    while (it.hasNext) {
      val cd = it.next()
      if (condition(cd.plan)) {
        cd.cachedRepresentation.cachedColumnBuffers.unpersist()
        // Remove the cache entry before we create a new one, so that we can have a different
        // physical plan.
        it.remove()
        val newCache = InMemoryRelation(
          useCompression = cd.cachedRepresentation.useCompression,
          batchSize = cd.cachedRepresentation.batchSize,
          storageLevel = cd.cachedRepresentation.storageLevel,
          child = spark.sessionState.executePlan(cd.plan).executedPlan,
          tableName = cd.cachedRepresentation.tableName)
        needToRecache += cd.copy(cachedRepresentation = newCache)
      }
    }

    needToRecache.foreach(cachedData.add)
  }

  /*
  def lookupCachedData(query: Dataset[_], isGPTPlan : Boolean = false): Option[CachedData] = readLock {

    if (isGPTPlan) {
      lookupCachedData(query.logicalPlan, true)

    } else {
      lookupCachedData(query.logicalPlan, false)
    }

  }

  def lookupCachedData(plan: LogicalPlan, isGPTPlan : Boolean): Option[CachedData] = readLock {

    if (isGPTPlan) {
      val scanOP = plan.asInstanceOf[FileSourceScanExec]
      cachedData.asScala.find(cd => plan.sameResult(cd.plan))

    } else {
      cachedData.asScala.find(cd => plan.sameResult(cd.plan))
    }
  }
  */

  /** Optionally returns cached data for the given [[Dataset]] */
  def lookupCachedData(query: Dataset[_]): Option[CachedData] = readLock {
    lookupCachedData(query.logicalPlan)
  }

  /** Optionally returns cached data for the given [[LogicalPlan]]. */
  def lookupCachedData(plan: LogicalPlan): Option[CachedData] = readLock {
    cachedData.asScala.find(cd => plan.sameResult(cd.plan))
  }

  def lookupCachedDataForGPT(currentLogicalPlan: LogicalPlan, cachedTblName: String, bitIdx: String) : Option[LogicalPlan] = readLock {

    /*
    val logicalPlan = cachedPlanForGPT.get((cachedTblName,bitIdx))
    if(logicalPlan.isDefined) {
      cachedDataForGPT.foreach{ c =>
        val cachedPlan = c._2.plan
        if(cachedPlan.sameResult(logicalPlan.get)) {
          // logInfo("[GPT CACHING] Find same in memory scan plan\n\n" + cachedPlan.numberedTreeString)
          Option(cachedPlan)
        }
      }
      // logInfo("[GPT CACHING] CANNOT FIND SAME IN MEM SCAN PLAN\n\n" + logicalPlan.get.numberedTreeString)
      cachedData.asScala.find(cd => logicalPlan.get.sameResult(cd.plan))
    } else {
      // logInfo("[GPT CACHING] CANNOT FIND SAME IN MEM SCAN PLAN\n\n" + currentLogicalPlan.numberedTreeString)
      cachedData.asScala.find(cd => currentLogicalPlan.sameResult(cd.plan))
    }
    */

    if(cachedDataForGPT.get(cachedTblName,bitIdx).isDefined &&
      cachedPlanForGPT.get(cachedTblName,bitIdx).isDefined &&
      cachedPlanForGPT.get(cachedTblName,bitIdx).get.sameResult(currentLogicalPlan)) {
      cachedInMemRelationForGPT.get(cachedTblName,bitIdx).map(_.withOutput(currentLogicalPlan.output))
    } else {
      Option(currentLogicalPlan)
    }
  }


  /** Replaces segments of the given logical plan with cached versions where possible. */
  def useCachedData(plan: LogicalPlan): LogicalPlan = {

    /*
      var isGPTPlan = false
      var tbl = ""
      var bitIdx = ""
      var selected = false
      if(plan.getJoinCols.isEmpty && getrequestedCacheForGPTTable() != "None") {
        val tok = getrequestedCacheForGPTTable().split('_')
        tbl = getrequestedCacheForGPTTable().split("_[0-9]")(0)
        bitIdx = "0"
        isGPTPlan = true
      } else {
        plan.getJoinCols.foreach{ col =>
          if(joinColToIdxMap.contains(col) && !selected) {
            tbl = joinColToTblNameMap.get(col).get
            bitIdx = joinColToIdxMap.get(col).get.toString
            selected = true
            isGPTPlan = true
          }
        }

        if(isTaskForCachingGPTable) {
          isGPTPlan = true
        }
      }
    */

      if (isTaskForCachingGPTable) {

        // logInfo("Requesting access to cached GPT table: " + getTableNameForCachingGPT)

        val newPlan = plan transformDown {
          case currentFragment => {
            lookupCachedDataForGPT(currentFragment, getTableNameForCachingGPT, bitIdxForRetrievingLogicalPlan.toString).getOrElse(currentFragment)
          }
        }

        newPlan transformAllExpressions {
          case s: SubqueryExpression => {
            // logInfo("newPlan transformAllExpressions with case SubqueryExpression " + s.nodeName)
            s.withNewPlan(useCachedData(s.plan))
          }
        }

      } else {

        // logInfo("Status checking: tableName [" + this.getTableNameForCachingGPT() + "], bitIdx [" + this.getBitIdxForRetrievingLogicalPlan() + "]")
        var needOpportunityToCheckGPTCaching = true
        var newPlan = plan transformDown {
          case currentFragment => {
            if (lookupCachedData(currentFragment).isDefined) {
              needOpportunityToCheckGPTCaching = false
            }
            lookupCachedData(currentFragment)
              .map(_.cachedRepresentation.withOutput(currentFragment.output))
              .getOrElse(currentFragment)
          }
        }

        if (needOpportunityToCheckGPTCaching) {

          // logInfo("needOpportunityToCheckGPTCaching: True")
          needOpportunityToCheckGPTCaching = false
          var bitIdx = getBitIdxForRetrievingLogicalPlan().toString
          var retrievedPlan: LogicalPlan = null
          var table2BitIdxMap = new mutable.HashMap[String,Int]()
          var tblName = ""
          var ColBitIdx = ""

          val copiedPlan = plan transformDown {
            case currentFragment => {

              if(currentFragment.nodeName == "SubqueryAlias") {
                tblName = currentFragment.verboseStringWithSuffix.split(" ")(1)
                if(table2BitIdxMap.get(tblName).isDefined) {
                  ColBitIdx = table2BitIdxMap.get(tblName).get.toString
                } else {
                  ColBitIdx = "0"
                }
              }
              currentFragment
            }
          }

          val sparkSession = SparkSession.getActiveSession.getOrElse(null)
          val preOptPlan = sparkSession.sessionState.planner.plan(ReturnAnswer(sparkSession.sessionState.optimizer.execute(sparkSession.sessionState.analyzer.execute(copiedPlan)))).next()

          val analyzingGPTPlan = preOptPlan transformDown {
            case op => {
              if (op.isInstanceOf[FileSourceScanExec] && isGPTPartitiondTable(op.asInstanceOf[FileSourceScanExec])) {
                needOpportunityToCheckGPTCaching = true
                val scanOP = op.asInstanceOf[FileSourceScanExec]
                var commonJoinColumn = "None"
                scanOP.getJoinColInQuery.foreach{
                  c => if(scanOP.getPartitioningColumn.contains(c._1))
                    commonJoinColumn = c._1
                }
                if (commonJoinColumn != "None") {
                  var colIdx = scanOP.getJoinColToIdxMap(commonJoinColumn)
                  if (colIdx == -1) {
                    colIdx = 0
                  }
                  // logInfo("GPT TableName: " + scanOP.tblName + ", [JoinColumnInThisQuery: " + scanOP.getJoinColInQuery() + "], [PartitioningColumn: " + scanOP.getPartitioningColumn() + "], commonJoinColumn: " + commonJoinColumn + " -> " + colIdx)
                  table2BitIdxMap += scanOP.tblName -> colIdx
                }
              }
              op
            }
          }

          // table2BitIdxMap.foreach(c => logInfo("tbl2BitIdx| " + c._1 + " -> " + c._2))

          // logInfo("Before Transformation: \n" + plan.numberedTreeString)
          newPlan = plan transformDown {
            case currentFragment => {

              var findInMemScanAlternatives = false
              table2BitIdxMap.foreach{ c =>
                val tbl = c._1
                val colIdx = c._2.toString
                if(needOpportunityToCheckGPTCaching && this.cachedPlanForGPT.get(tbl,colIdx).isDefined &&
                  currentFragment.sameResult(this.cachedPlanForGPT.get(tbl,colIdx).get)) {
                  retrievedPlan = this.cachedInMemRelationForGPT.get(tbl, colIdx).get.withOutput(currentFragment.output)
                  // logInfo("needOpportunityToCheckGPTCaching: REPLACE TO IN-MEMORY SCAN PLAN [" + currentFragment.nodeName + "] --> \n")
                  // logInfo("\n\n" + retrievedPlan.numberedTreeString)
                  findInMemScanAlternatives = true
                }
              }
              if(!findInMemScanAlternatives) {
                retrievedPlan = currentFragment
              }
              retrievedPlan
            }
          }
        }
        // logInfo("After Transformation: \n" + newPlan.numberedTreeString)

        newPlan transformAllExpressions {
          case s: SubqueryExpression => {
            s.withNewPlan(useCachedData(s.plan))
          }
        }

        /*
        if(needOpportunityToCheckGPTCaching) {
          logInfo("Status checking: " + getTableNameForCachingGPT() + " -> " + getBitIdxForRetrievingLogicalPlan())
          val bitIdx = getBitIdxForRetrievingLogicalPlan().toString
          this.cachedDataForGPT.foreach{ cached => logInfo("CacheManager checking: " + cached._1)}
          logInfo("needOpportunityToCheckGPTCaching: True\n" + plan.numberedTreeString)

          val newPlan = plan transformDown {
            case currentFragment => {
              logInfo("needOpportunityToCheckGPTCaching: currentFragment [" + currentFragment.nodeName + "]")
              this.cachedPlanForGPT.get(getTableNameForCachingGPT,bitIdx).getOrElse(currentFragment)

              /*
              if(this.cachedDataForGPT.get(getTableNameForCachingGPT,bitIdx).isDefined &&
                this.cachedPlanForGPT.get(getTableNameForCachingGPT,bitIdx).isDefined &&
                this.cachedPlanForGPT.get(getTableNameForCachingGPT,bitIdx).get.sameResult(currentFragment)) {
                this.cachedPlanForGPT.get(getTableNameForCachingGPT,bitIdx).getOrElse(currentFragment)
              } else {
                currentFragment
              }
              */
            }
          }

          logInfo("needOpportunityToCheckGPTCaching: cachedPlan?\n" + newPlan.numberedTreeString)

          newPlan transformAllExpressions {
            case s: SubqueryExpression => {
              logInfo("newPlan transformAllExpressions with case SubqueryExpression " + s.nodeName)
              s.withNewPlan(useCachedData(s.plan))
            }
          }
          */
        /*
        else {
          val newPlan = plan transformDown {
            case currentFragment => {
              if(lookupCachedData(currentFragment).isDefined) {
                needOpportunityToCheckGPTCaching = false
              }
              lookupCachedData(currentFragment)
                .map(_.cachedRepresentation.withOutput(currentFragment.output))
                .getOrElse(currentFragment)
            }
          }

          newPlan transformAllExpressions {
            case s: SubqueryExpression => {
              // logInfo("newPlan transformAllExpressions with case SubqueryExpression " + s.nodeName)
              s.withNewPlan(useCachedData(s.plan))
            }
          }
        }
      }
      */
      }
  }

  /**
   * Tries to re-cache all the cache entries that contain `resourcePath` in one or more
   * `HadoopFsRelation` node(s) as part of its logical plan.
   */
  def recacheByPath(spark: SparkSession, resourcePath: String): Unit = writeLock {
    val (fs, qualifiedPath) = {
      val path = new Path(resourcePath)
      val fs = path.getFileSystem(spark.sessionState.newHadoopConf())
      (fs, fs.makeQualified(path))
    }

    recacheByCondition(spark, _.find(lookupAndRefresh(_, fs, qualifiedPath)).isDefined)
  }

  /**
   * Traverses a given `plan` and searches for the occurrences of `qualifiedPath` in the
   * [[org.apache.spark.sql.execution.datasources.FileIndex]] of any [[HadoopFsRelation]] nodes
   * in the plan. If found, we refresh the metadata and return true. Otherwise, this method returns
   * false.
   */
  private def lookupAndRefresh(plan: LogicalPlan, fs: FileSystem, qualifiedPath: Path): Boolean = {
    plan match {
      case lr: LogicalRelation => lr.relation match {
        case hr: HadoopFsRelation =>
          val prefixToInvalidate = qualifiedPath.toString
          val invalidate = hr.location.rootPaths
            .map(_.makeQualified(fs.getUri, fs.getWorkingDirectory).toString)
            .exists(_.startsWith(prefixToInvalidate))
          if (invalidate) hr.location.refresh()
          invalidate
        case _ => false
      }
      case _ => false
    }
  }
}
