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

import scala.collection.mutable.ArrayBuffer
import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.fs.{BlockLocation, FileStatus, LocatedFileStatus, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{InternalRow, TableIdentifier}
import org.apache.spark.sql.catalyst.catalog.BucketSpec
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenContext
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, UnknownPartitioning}
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.execution.datasources.parquet.{ParquetFileFormat => ParquetSource}
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources.BaseRelation
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.BitSet

import scala.collection.mutable

trait DataSourceScanExec extends LeafExecNode with CodegenSupport {
  val relation: BaseRelation
  val metastoreTableIdentifier: Option[TableIdentifier]

  protected val nodeNamePrefix: String = ""

  override val nodeName: String = {
    s"Scan $relation ${metastoreTableIdentifier.map(_.unquotedString).getOrElse("")}"
  }

  override def simpleString: String = {
    val metadataEntries = metadata.toSeq.sorted.map {
      case (key, value) =>
        key + ": " + StringUtils.abbreviate(redact(value), 100)
    }
    val metadataStr = Utils.truncatedString(metadataEntries, " ", ", ", "")
    s"$nodeNamePrefix$nodeName${Utils.truncatedString(output, "[", ",", "]")}$metadataStr"
  }

  override def verboseString: String = redact(super.verboseString)

  override def treeString(verbose: Boolean, addSuffix: Boolean): String = {
    redact(super.treeString(verbose, addSuffix))
  }

  /**
   * Shorthand for calling redactString() without specifying redacting rules
   */
  private def redact(text: String): String = {
    Utils.redact(SparkSession.getActiveSession.get.sparkContext.conf, text)
  }
}

/** Physical plan node for scanning data from a relation. */
case class RowDataSourceScanExec(
    output: Seq[Attribute],
    rdd: RDD[InternalRow],
    @transient relation: BaseRelation,
    override val outputPartitioning: Partitioning,
    override val metadata: Map[String, String],
    override val metastoreTableIdentifier: Option[TableIdentifier])
  extends DataSourceScanExec {

  override lazy val metrics =
    Map("numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"))

  val outputUnsafeRows = relation match {
    case r: HadoopFsRelation if r.fileFormat.isInstanceOf[ParquetSource] =>
      !SparkSession.getActiveSession.get.sessionState.conf.getConf(
        SQLConf.PARQUET_VECTORIZED_READER_ENABLED)
    case _: HadoopFsRelation => true
    case _ => false
  }

  protected override def doExecute(): RDD[InternalRow] = {
    val unsafeRow = if (outputUnsafeRows) {
      rdd
    } else {
      rdd.mapPartitionsWithIndexInternal { (index, iter) =>
        val proj = UnsafeProjection.create(schema)
        proj.initialize(index)
        iter.map(proj)
      }
    }

    val numOutputRows = longMetric("numOutputRows")
    unsafeRow.map { r =>
      numOutputRows += 1
      r
    }
  }

  override def inputRDDs(): Seq[RDD[InternalRow]] = {
    rdd :: Nil
  }

  override protected def doProduce(ctx: CodegenContext): String = {
    val numOutputRows = metricTerm(ctx, "numOutputRows")
    // PhysicalRDD always just has one input
    val input = ctx.freshName("input")
    ctx.addMutableState("scala.collection.Iterator", input, s"$input = inputs[0];")
    val exprRows = output.zipWithIndex.map{ case (a, i) =>
      BoundReference(i, a.dataType, a.nullable)
    }
    val row = ctx.freshName("row")
    ctx.INPUT_ROW = row
    ctx.currentVars = null
    val columnsRowInput = exprRows.map(_.genCode(ctx))
    val inputRow = if (outputUnsafeRows) row else null
    s"""
       |while ($input.hasNext()) {
       |  InternalRow $row = (InternalRow) $input.next();
       |  $numOutputRows.add(1);
       |  ${consume(ctx, columnsRowInput, inputRow).trim}
       |  if (shouldStop()) return;
       |}
     """.stripMargin
  }

  // Only care about `relation` and `metadata` when canonicalizing.
  override def preCanonicalized: SparkPlan =
    copy(rdd = null, outputPartitioning = null, metastoreTableIdentifier = None)
}

/**
 * Physical plan node for scanning data from HadoopFsRelations.
 *
 * @param relation The file-based relation to scan.
 * @param output Output attributes of the scan, including data attributes and partition attributes.
 * @param requiredSchema Required schema of the underlying relation, excluding partition columns.
 * @param partitionFilters Predicates to use for partition pruning.
 * @param dataFilters Filters on non-partition columns.
 * @param joinColumns used for GPT!
 * @param metastoreTableIdentifier identifier for the table in the metastore.
 */
case class FileSourceScanExec(
    @transient relation: HadoopFsRelation,
    output: Seq[Attribute],
    requiredSchema: StructType,
    partitionFilters: Seq[Expression],
    dataFilters: Seq[Expression],
    joinColumns: scala.collection.mutable.HashMap[String, scala.collection.mutable.HashSet[String]],
    override val metastoreTableIdentifier: Option[TableIdentifier])
  extends DataSourceScanExec with ColumnarBatchScan  {

  val supportsBatch: Boolean = relation.fileFormat.supportBatch(
    relation.sparkSession, StructType.fromAttributes(output))

  val needsUnsafeRowConversion: Boolean = if (relation.fileFormat.isInstanceOf[ParquetSource]) {
    SparkSession.getActiveSession.get.sessionState.conf.parquetVectorizedReaderEnabled
  } else {
    false
  }

  def getJoinColInQuery() : scala.collection.mutable.HashMap[String, scala.collection.mutable.HashSet[String]] = {
    joinColumns
  }

  def getPartitioningColumn() : Seq[String] = {
    if (relation.bucketSpec.isDefined) {
      relation.bucketSpec.get.bucketColumnNames
    } else {
      Seq.empty[String]
    }
  }

  def getJoinColToIdxMap(colName: String) : Int = {
    var colIdx = 0

    var retColIdx = -1

    relation.bucketSpec.get.bucketColumnNames.reverse.foreach{ c =>
      sqlContext.sharedState.cacheManager.joinColToIdxMap += (c -> colIdx)
      sqlContext.sharedState.cacheManager.joinColToTblNameMap += (c -> tblName)
      if(c == colName) {
        retColIdx = colIdx
      }
      colIdx += 1
    }
    retColIdx
  }


  @transient private lazy val selectedPartitions: Seq[PartitionDirectory] = {
    val optimizerMetadataTimeNs = relation.location.metadataOpsTimeNs.getOrElse(0L)
    val startTime = System.nanoTime()
    val ret = relation.location.listFiles(partitionFilters, dataFilters)
    val timeTakenMs = ((System.nanoTime() - startTime) + optimizerMetadataTimeNs) / 1000 / 1000

    metrics("numFiles").add(ret.map(_.files.size.toLong).sum)
    metrics("metadataTime").add(timeTakenMs)

    val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
    SQLMetrics.postDriverMetricUpdates(sparkContext, executionId,
      metrics("numFiles") :: metrics("metadataTime") :: Nil)

    ret
  }

  var isOuterJoinTable = false
  def setAsOuterJoinTable() : Unit = {
    isOuterJoinTable = true
  }
  def getIsOuterJoinTable() : Boolean = {
    this.isOuterJoinTable
  }

  var tblName = "None"
  var selectedJoinColForGPT = "None"
  var GPTPartitioning : org.apache.spark.sql.catalyst.plans.physical.Partitioning = null

  def isGPTCoverable() : Boolean = {

    val GPTCols = relation.bucketSpec.get.bucketColumnNames.toSeq
    GPTCols.foreach {
      col =>
        if (joinColumns.contains(col)) {
          true
        }
    }
    false
  }

  def getGPTJoinCols(bucketSpec: BucketSpec) : collection.mutable.ArrayBuffer[String] = {

    val GPTCols = bucketSpec.bucketColumnNames.toSeq
    var newJoinCols = new collection.mutable.ArrayBuffer[String]()

    GPTCols.foreach {
      col =>
       // if (joinColumns.contains(col)) newJoinCols += col
       if (relation.getJoinTypes.contains(col)) {

         // partitioning column selection

         // 1. If a query contains only broadcast hash joins?
         // --> Just select any partitioning column corresponding to broadcast hash join

         // 2. If a query contains only sort-merge joins?
         //  --> Just select any partitioning column corresponding to sort-merge join

         // 3. If a query contains both broadcast join and sort-merge join?
         //  --> We should prefer sort-merge join since it is much expensive than broadcasting small hashed relation due to expensive shuffling

         // cover case 2 and 3
         if(relation.getJoinTypes.get(col).get.contains("SortMergeJoinExec") && newJoinCols.isEmpty) {
           newJoinCols += col
         }
       }
    }

    // cover case 1
    if(newJoinCols.isEmpty) {
      GPTCols.foreach{ c =>
        if(relation.getJoinTypes.contains(c))
          newJoinCols += c
      }
    }

    newJoinCols
  }

  override val (outputPartitioning, outputOrdering): (Partitioning, Seq[SortOrder]) = {
    val bucketSpec = if (relation.sparkSession.sessionState.conf.bucketingEnabled) {
      relation.bucketSpec
    } else {
      None
    }

    if (bucketSpec.isDefined) {
      // logDebug(s"partitionSchema: " + relation.partitionSchema.toList)
      logDebug(s"bucketSpec: " + bucketSpec.get.toString)
      if (!selectedPartitions.isEmpty && !selectedPartitions(0).files.isEmpty) {
        var isGPTPartitioned = selectedPartitions(0).files(0).getPath.getName.split("/")(0).contains("GPT")
        var tokens = selectedPartitions(0).files(0).getPath.toString().split("-")
        tblName = tokens(1).replace("_text","")

        if (isGPTPartitioned) {
          relation.sparkSession.sessionState.conf.setConfString("GPTBucket", relation.bucketSpec.get.numBuckets.toString)
          bucketSpec.get.setGPTPartitioned()
        }
      }
    }

    var isGPT = false;
    if (bucketSpec.isDefined) {
      isGPT = bucketSpec.get.isGPTPartitioned
    }

    if (isGPT) {
      def toAttribute(colName: String): Option[Attribute] =
        output.find(_.name == colName)

      def isLocalTask(joinColumns: scala.collection.mutable.HashMap[String, scala.collection.mutable.HashSet[String]], spec: Set[String]) : Boolean = {
        var ret = false
        joinColumns.foreach {
          p => if (spec.contains(p._1)) {
            ret = true
          }
        }
        ret
      }

      val spec = bucketSpec.get
      val bucketColumns = spec.bucketColumnNames.flatMap(n => toAttribute(n))
      var sortOrder : Seq[org.apache.spark.sql.catalyst.expressions.SortOrder] = null

      var GPTJoinCols = getGPTJoinCols(spec)

      var GPTJoinColsAsAttributes = GPTJoinCols.flatMap(n => toAttribute(n)).toSeq

      GPTPartitioning = HashPartitioning(GPTJoinColsAsAttributes, spec.numBuckets)

      logDebug(s"\t\tGPT Partitioning Columns: " + GPTJoinCols)
      logDebug(s"\t\tJoin Columns in the query")
      joinColumns.foreach{ c =>
        logDebug(s"\t\t\tjoinKey: " + c._1 + ", joinTypes: " + c._2.mkString(", "))
      }
      logDebug(s"\t\tJoin Columns in the HadoopFSRelation")
      relation.getJoinTypes.foreach{ c =>
        logDebug(s"\t\t\tjoinKey: " + c._1 + ", joinTypes: " + c._2.mkString(", "))
      }
      logDebug(s"\t\tSort Columns: " + spec.sortColumnNames.map(x => toAttribute(x)).takeWhile(x => x.isDefined).map(_.get))
      // Filters: List((isnull(cr_item_sk#63) || isnotnull(cr_item_sk#63)))
      logDebug(s"\t\tFilters: " + dataFilters)

      if (isLocalTask(joinColumns, spec.bucketColumnNames.toSet)) {
        val sortColumns =
          spec.sortColumnNames.map(x => toAttribute(x)).takeWhile(x => x.isDefined).map(_.get)

        sortOrder = if (sortColumns.nonEmpty) {
          // In case of bucketing, its possible to have multiple files belonging to the
          // same bucket in a given relation. Each of these files are locally sorted
          // but those files combined together are not globally sorted. Given that,
          // the RDD partition will not be sorted even if the relation has sort columns set
          // Current solution is to check if all the buckets have a single file in it

          val files = selectedPartitions.flatMap(partition => partition.files)
          val bucketToFilesGrouping =
            files.map(_.getPath.getName).groupBy(file => BucketingUtils.getBucketId(file))
          val singleFilePartitions = bucketToFilesGrouping.forall(p => p._2.length <= 1)

          if (singleFilePartitions) {
            // TODO Currently Spark does not support writing columns sorting in descending order
            // so using Ascending order. This can be fixed in future
            sortColumns.map(attribute => SortOrder(attribute, Ascending))
          } else {
            Nil
          }
        } else {
          Nil
        }
      }
      logDebug("This Scan OP is for GPT | Join Column ^ spec.bucketColumnName != Nil | " +
        "[PARTITIONING: HashPartitioning(GPTCols: " + (if(GPTJoinColsAsAttributes.isEmpty) {"Empty"} else {GPTJoinColsAsAttributes.mkString})+ ", numBuckets: " + spec.numBuckets)
      (GPTPartitioning, sortOrder)
    } else {
        // NOT FOR GPT
        bucketSpec match {
          case Some(spec) =>
            // For bucketed columns:
            // -----------------------
            // `HashPartitioning` would be used only when:
            // 1. ALL the bucketing columns are being read from the table
            //
            // For sorted columns:
            // ---------------------
            // Sort ordering should be used when ALL these criteria's match:
            // 1. `HashPartitioning` is being used
            // 2. A prefix (or all) of the sort columns are being read from the table.
            //
            // Sort ordering would be over the prefix subset of `sort columns` being read
            // from the table.
            // eg.
            // Assume (col0, col2, col3) are the columns read from the table
            // If sort columns are (col0, col1), then sort ordering would be considered as (col0)
            // If sort columns are (col1, col0), then sort ordering would be empty as per rule #2
            // above

            def toAttribute(colName: String): Option[Attribute] =
              output.find(_.name == colName)

            val bucketColumns = spec.bucketColumnNames.flatMap(n => toAttribute(n))

            if (bucketColumns.size == spec.bucketColumnNames.size) {
              val partitioning = HashPartitioning(bucketColumns, spec.numBuckets)
              val sortColumns =
                spec.sortColumnNames.map(x => toAttribute(x)).takeWhile(x => x.isDefined).map(_.get)

              val sortOrder = if (sortColumns.nonEmpty) {
                // In case of bucketing, its possible to have multiple files belonging to the
                // same bucket in a given relation. Each of these files are locally sorted
                // but those files combined together are not globally sorted. Given that,
                // the RDD partition will not be sorted even if the relation has sort columns set
                // Current solution is to check if all the buckets have a single file in it

                val files = selectedPartitions.flatMap(partition => partition.files)
                val bucketToFilesGrouping =
                  files.map(_.getPath.getName).groupBy(file => BucketingUtils.getBucketId(file))
                val singleFilePartitions = bucketToFilesGrouping.forall(p => p._2.length <= 1)

                if (singleFilePartitions) {
                  // TODO Currently Spark does not support writing columns sorting in descending order
                  // so using Ascending order. This can be fixed in future
                  sortColumns.map(attribute => SortOrder(attribute, Ascending))
                } else {
                  Nil
                }
              } else {
                Nil
              }
              logDebug("[This Scan OP != GPT] and for the case Some(Spec) AND bucketColumns.size == spec.bucketColumnNames.size | " +
                "[PARTITIONING: HashPartitioning(BucketCol: " + bucketColumns.mkString +
                ", bucketNums: " + spec.numBuckets + ", [SORTING: " + sortOrder.mkString + "]")
              (partitioning, sortOrder)
            } else {
              logDebug("[This Scan OP != GPT] and for the case Some(Spec) AND bucketColumns.size != spec.bucketColumnNames.size | [SORTING: Nil]")
              logDebug("\tbucketColumns: " + bucketColumns.mkString + ", spec.bucketColumnNames: " + spec.bucketColumnNames.mkString)
              (UnknownPartitioning(0), Nil)
            }
          case _ => {
            logDebug("[This Scan OP !=  GPT] and for the case _")
            (UnknownPartitioning(0), Nil)
          }
        }
      }
    /*
    if(isGPT) {
      logDebug("[WIERD] This Scan OP is for GPT | [SORTING: Nil]")
      (GPTPartitioning, Nil)
    } else {
      logDebug("[WIERD] This Scan OP is not for GPT | [SORTING: Nil]")
      (UnknownPartitioning(0), Nil)
    }
    */
  }
  /*
  var isGPTPartitionCaching = false
  dataFilters.foreach { c =>
    // (isnull(cr_item_sk#3) || isnotnull(cr_item_sk#3))
    val str = c.simpleString
    val newStr = str.substring(1, str.length-1)
    val tmpFilterCols = new collection.mutable.ArrayBuffer[String]()
    val tmpFilterColNames = new collection.mutable.HashSet[String]()
    val tmpFilterExprs = new collection.mutable.ArrayBuffer[String]()

    str.substring(1, str.length-1).split('|').foreach(c => if(c.contains('(')) {tmpFilterCols += c.trim})
    tmpFilterCols.foreach{c =>
      val expr = c.split('(')(0)
      val colNameWithNum = c.split('(')(1).split(')')(0)
      val colTok = colNameWithNum.split('#')
      val colName = colTok(0)
      val colIdx = colTok(1)
      tmpFilterExprs += expr
      tmpFilterColNames += colName
    }

    if (tmpFilterColNames.size == 1 && tmpFilterExprs.size == 2 &&
      tmpFilterExprs.contains("isnull") && tmpFilterExprs.contains("isnotnull")) {

      // FOR GPT TABLE CACHING
      isGPTPartitionCaching = true
    }
  }
  */

  @transient
  private val pushedDownFilters = dataFilters.flatMap(DataSourceStrategy.translateFilter)


  // These metadata values make scan plans uniquely identifiable for equality checking.
  override val metadata: Map[String, String] = {
    def seqToString(seq: Seq[Any]) = seq.mkString("[", ", ", "]")
    val location = relation.location
    val locationDesc =
      location.getClass.getSimpleName + seqToString(location.rootPaths)
    val isGPTPartitioned = if (relation.bucketSpec.isDefined) {
      relation.bucketSpec.get.isGPTPartitioned.toString
    } else {
      "false"
    }
    val metadata =
      Map(
        "Format" -> relation.fileFormat.toString,
        "ReadSchema" -> requiredSchema.catalogString,
        "Batched" -> supportsBatch.toString,
        "PartitionFilters" -> seqToString(partitionFilters),
        "PushedFilters" -> seqToString(pushedDownFilters),
        "Location" -> locationDesc,
         "GPT" -> isGPTPartitioned)
    val withOptPartitionCount =
      relation.partitionSchemaOption.map { _ =>
        metadata + ("PartitionCount" -> selectedPartitions.size.toString)
      } getOrElse {
        metadata
      }
    withOptPartitionCount
  }

  private lazy val inputRDD: RDD[InternalRow] = {
    val readFile: (PartitionedFile) => Iterator[InternalRow] =
      relation.fileFormat.buildReaderWithPartitionValues(
        sparkSession = relation.sparkSession,
        dataSchema = relation.dataSchema,
        partitionSchema = relation.partitionSchema,
        requiredSchema = requiredSchema,
        filters = pushedDownFilters,
        options = relation.options,
        hadoopConf = relation.sparkSession.sessionState.newHadoopConfWithOptions(relation.options))

    // logDebug(s"FileSourceScanExec | filters: " + pushedDownFilters.toList)
    // logDebug(s"FileSourceScanExec | options: " + relation.options.toList)

    relation.bucketSpec match {
      case Some(bucketing) if relation.sparkSession.sessionState.conf.bucketingEnabled =>
        createBucketedReadRDD(bucketing, readFile, selectedPartitions, getIsOuterJoinTable(), relation.getJoinTypes, relation)
      case _ =>
        createNonBucketedReadRDD(readFile, selectedPartitions, relation)
    }
  }

  /*
  def renewInputRDD(bitIdx: Int): Unit = {
    val readFile: (PartitionedFile) => Iterator[InternalRow] =
      relation.fileFormat.buildReaderWithPartitionValues(
        sparkSession = relation.sparkSession,
        dataSchema = relation.dataSchema,
        partitionSchema = relation.partitionSchema,
        requiredSchema = requiredSchema,
        filters = pushedDownFilters,
        options = relation.options,
        hadoopConf = relation.sparkSession.sessionState.newHadoopConfWithOptions(relation.options))

    val bucketSpec = relation.bucketSpec.get
    val handleOuterJoin = true

    var joinCols : collection.mutable.ArrayBuffer[String] = getGPTJoinCols(bucketSpec)
    var joinColToIdxMap : collection.mutable.HashMap[String, Int] = new collection.mutable.HashMap[String, Int]()

    val numGPTCols = bucketSpec.bucketColumnNames.size

    // test for implementing GPT -- start --
    // logDebug(s"cretedBucketedReadRDD | bucketSpec.bucketColumnNames: " + bucketSpec.bucketColumnNames)
    // logDebug(s"cretedBucketedReadRDD | bucketSpec.numBuckets: " + bucketSpec.numBuckets)
    // joinColumns.foreach(c => logDebug(s"cretedBucketedReadRDD |" + s" join column intersection: " + c))

    val validBitIdx = new collection.mutable.ListBuffer[Int]()
    var colIdx = 0

    validBitIdx += bitIdx

    joinCols.foreach(c => logDebug(s"Join column in the query: " + c))
    validBitIdx.foreach(c => logDebug(s"validBitIdx: " + c))

    def genBitSets(numBits: Int): collection.mutable.ListBuffer[String] = {

      val bitSets = new collection.mutable.ListBuffer[String]()
      var minSize = Integer.MAX_VALUE;
      val maxNum = math.pow(numBits.toLong, 2).toInt
      for (num <- 0 to maxNum) {
        val bitStr = Integer.toString(num, 2)
        val s = bitStr.reverse.padTo(numBits, "0").reverse.toString()
        if (minSize > s.size) {
          minSize = s.size
        }
        if (s.size == minSize) {
          bitSets += s
        }
      }
      bitSets
    }

    def pow2(bitset: BitSet,
             numBits: Int) : Int = {

      var partitionIdentifier = 0
      for(i <- 0 to numBits-1) {
        val isBitSet = bitset.get(i)
        var bitVal = -1
        if(isBitSet) {
          bitVal = 1
        } else {
          bitVal = 0
        }
        partitionIdentifier += (bitVal * math.pow(2, i).toInt)
      }
      partitionIdentifier
    }

    def repeatChar(c: Char, n: Int): String = c.toString * n

    val bitsets = genBitSets(numGPTCols)
    val candidateBitSets = new collection.mutable.ListBuffer[String]()
    val validBitSets = new collection.mutable.ListBuffer[String]()

    val pattern = "[0-9]".r
    for (bitset <- bitsets) {
      // logDebug(s"cretedBucketedReadRDD | candidate bitset: " + bitset)
      val tempArr = new Array[Int](numGPTCols)
      val str = (pattern findAllIn bitset).toList
      val idx = 0
      for (i <- 0 to str.size-1) {
        tempArr(i) = str(i).toInt
      }
      candidateBitSets += tempArr.mkString("")
    }

    for (elem <- candidateBitSets) {
      var ret = true
      validBitIdx.foreach(c => if (elem.reverse.charAt(c) == '0') ret = false)
      if (ret) {
        validBitSets += elem
      }
    }

    validBitSets.foreach(c => logDebug(s"Selected Sub-Partitions: " + c))

    var nullSubPartitionBitVals = new scala.collection.mutable.ListBuffer[Int]()
    var nullSubPartitions = new scala.collection.mutable.ListBuffer[String]()
    val nullPartitionFile = new scala.collection.mutable.ListBuffer[PartitionedFile]

    var bitV = validBitSets.map { c =>
      val bits = new BitSet(numGPTCols)
      var idx = 0
      c.reverse.foreach{b => if (b == '1') bits.set(idx)
        idx += 1}
      nullSubPartitionBitVals += bits.pow2().toInt
      bits.pow2()
    }

    nullSubPartitionBitVals.foreach{ b =>
      val bitVal = b
      val numPad = 5 - bitVal.toString.length
      val bitString = repeatChar('0', numPad) + bitVal
      nullSubPartitions += bitString
    }

    if (handleOuterJoin) {
      nullSubPartitionBitVals.foreach( c => logDebug(s"nullSubPartitionBitVals: " + c))
      nullSubPartitions.foreach( c => logDebug(s"nullSubPartitionBitVals: " + c))
    }

    // hdfs://10.150.20.72:8020/tpcds/SS_2_C/GPT-store_sales_text-0-00000.csv
    var partitionMap = new collection.mutable.HashMap[Int, collection.mutable.ListBuffer[PartitionedFile]]()
    selectedPartitions.foreach { p =>
      p.files.foreach{ f =>
        val tokens = f.getPath.toString().split("-")

        val prefix = tokens(0)
        val tblName = tokens(1)
        val bitVal = tokens(2).toInt
        val naivePartitionID = tokens(3).substring(0, 5)
        val partitionID = GPTUtils.getPartitionID(tokens(3).substring(0, 5))

        if (handleOuterJoin) {

          if (bitVal == 0) {
            logDebug(s"candidate null subpartition: " + f.getPath.getName)
            var findNullSubPartition = false
            nullSubPartitions.foreach(c => if (c == naivePartitionID) findNullSubPartition = true)

            if (findNullSubPartition) {
              logDebug("Handling outer join! adding partitions: 0-" + naivePartitionID)
              val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
              val partitionedFile = PartitionedFile(p.values, f.getPath.toUri.toString, 0, f.getLen, hosts)
              nullPartitionFile += partitionedFile
            }
          }
        }

        if (bitV.contains(bitVal)) {

          val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
          val partitionedFile = PartitionedFile(p.values, f.getPath.toUri.toString,
            0, f.getLen, hosts)

          if (!partitionMap.contains(partitionID)) {
            var partitionedFiles = new collection.mutable.ListBuffer[PartitionedFile]()
            partitionedFiles += partitionedFile
            partitionMap += (partitionID -> partitionedFiles)
          } else {
            partitionMap(partitionID) += partitionedFile
          }
        }
      }


      if (handleOuterJoin) {
        val nullBitVal = 0
        partitionMap.foreach{ p =>
          p._2 ++ nullPartitionFile
        }
      }
    }

    val filePartitions = Seq.tabulate(bucketSpec.numBuckets) { bucketId =>
      FilePartition(bucketId, partitionMap.getOrElse(bucketId, Nil))
    }

    sparkContext.conf.set("GPTTask", "true")

    inputRDD = new FileScanRDD(relation.sparkSession, readFile, filePartitions)
  }
  */

  override def inputRDDs(): Seq[RDD[InternalRow]] = {
    inputRDD :: Nil
  }

  override lazy val metrics =
    Map("numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
      "numFiles" -> SQLMetrics.createMetric(sparkContext, "number of files"),
      "metadataTime" -> SQLMetrics.createMetric(sparkContext, "metadata time (ms)"),
      "scanTime" -> SQLMetrics.createTimingMetric(sparkContext, "scan time"))

  protected override def doExecute(): RDD[InternalRow] = {
    if (supportsBatch) {
      // in the case of fallback, this batched scan should never fail because of:
      // 1) only primitive types are supported
      // 2) the number of columns should be smaller than spark.sql.codegen.maxFields
      WholeStageCodegenExec(this).execute()
    } else {
      val unsafeRows = {
        val scan = inputRDD
        if (needsUnsafeRowConversion) {
          scan.mapPartitionsWithIndexInternal { (index, iter) =>
            val proj = UnsafeProjection.create(schema)
            proj.initialize(index)
            iter.map(proj)
          }
        } else {
          scan
        }
      }
      val numOutputRows = longMetric("numOutputRows")
      unsafeRows.map { r =>
        numOutputRows += 1
        r
      }
    }
  }

  override val nodeNamePrefix: String = "File"

  override protected def doProduce(ctx: CodegenContext): String = {
    if (supportsBatch) {
      return super.doProduce(ctx)
    }
    val numOutputRows = metricTerm(ctx, "numOutputRows")
    // PhysicalRDD always just has one input
    val input = ctx.freshName("input")
    ctx.addMutableState("scala.collection.Iterator", input, s"$input = inputs[0];")
    val exprRows = output.zipWithIndex.map{ case (a, i) =>
      BoundReference(i, a.dataType, a.nullable)
    }
    val row = ctx.freshName("row")
    ctx.INPUT_ROW = row
    ctx.currentVars = null
    val columnsRowInput = exprRows.map(_.genCode(ctx))
    val inputRow = if (needsUnsafeRowConversion) null else row
    s"""
       |while ($input.hasNext()) {
       |  InternalRow $row = (InternalRow) $input.next();
       |  $numOutputRows.add(1);
       |  ${consume(ctx, columnsRowInput, inputRow).trim}
       |  if (shouldStop()) return;
       |}
     """.stripMargin
  }

  /**
   * Create an RDD for bucketed reads.
   * The non-bucketed variant of this function is [[createNonBucketedReadRDD]].
   *
   * The algorithm is pretty simple: each RDD partition being returned should include all the files
   * with the same bucket id from all the given Hive partitions.
   *
   * @param bucketSpec the bucketing spec.
   * @param readFile a function to read each (part of a) file.
   * @param selectedPartitions Hive-style partition that are part of the read.
   * @param fsRelation [[HadoopFsRelation]] associated with the read.
   */
  private def createBucketedReadRDD(
      bucketSpec: BucketSpec,
      readFile: (PartitionedFile) => Iterator[InternalRow],
      selectedPartitions: Seq[PartitionDirectory],
      handleOuterJoin: Boolean,
      joinColumns: scala.collection.mutable.HashMap[String, scala.collection.mutable.HashSet[String]],
      fsRelation: HadoopFsRelation): RDD[InternalRow] = {

    logDebug(s"Planning with ${bucketSpec.numBuckets} buckets")
    var containingSubPartitionForOuterJoin = true
    /*
    logDebug(s"debugString!: " + selectedPartitions(0).files(0).getPath.getName)
    var idx = 0
    selectedPartitions.foreach{ p =>
      logDebug(s"partition[" + idx + "]")
      p.files.foreach{ s =>
        logDebug(s"\t" + s)
      }
      idx += 1
    }
    */
    sparkContext.conf.set("GPTTask", "false")
    var isGPT = bucketSpec.isGPTPartitioned

    if (isGPT) {

      sparkContext.conf.set("GPTTask", "true")

      var joinColToIdxMap : collection.mutable.HashMap[String, Int] = new collection.mutable.HashMap[String, Int]()
      val numGPTCols = bucketSpec.bucketColumnNames.size
      val validBitIdx = new collection.mutable.ListBuffer[Int]()
      var colIdx = 0

      def getGPTJoinCols(bucketSpec: BucketSpec) : collection.mutable.ArrayBuffer[String] = {

        val GPTCols = bucketSpec.bucketColumnNames.toSeq
        var newJoinCols = new collection.mutable.ArrayBuffer[String]()

        GPTCols.foreach {
          col =>
            // if (joinColumns.contains(col)) newJoinCols += col
            if (relation.getJoinTypes.contains(col)) {

              // partitioning column selection

              // 1. If a query contains only broadcast hash joins?
              // --> Just select any partitioning column corresponding to broadcast hash join

              // 2. If a query contains only sort-merge joins?
              //  --> Just select any partitioning column corresponding to sort-merge join

              // 3. If a query contains both broadcast join and sort-merge join?
              //  --> We should prefer sort-merge join since it is much expensive than broadcasting small hashed relation due to expensive shuffling

              // cover case 2 and 3
              if(relation.getJoinTypes.get(col).get.contains("SortMergeJoinExec") && newJoinCols.isEmpty) {
                newJoinCols += col
              }
            }
        }

        // cover case 1
        if(newJoinCols.isEmpty) {
          GPTCols.foreach{ c =>
            if(relation.getJoinTypes.contains(c))
              newJoinCols += c
          }
        }

        newJoinCols
      }

      bucketSpec.bucketColumnNames.reverse.foreach{ c =>
        relation.dataSchema.foreach{ k =>
          if(k.name == c)
            joinColToIdxMap += (c -> colIdx)
        }
        sqlContext.sharedState.cacheManager.joinColToIdxMap += (c -> colIdx)
        sqlContext.sharedState.cacheManager.joinColToTblNameMap += (c -> tblName)
        logDebug("Building joinColToIdxMap (Partitioning Column Name -> Idx): " + c + " -> " + colIdx)
        colIdx += 1
      }

      val GPTJoinCols = getGPTJoinCols(bucketSpec)

      colIdx = 0
      if (!joinColumns.isEmpty) {
        // |GPT Partitioning Columns| > |join Columns|
        logDebug(s"[SCAN with GPT Column]")
        var setValidBitIdx = false
        GPTJoinCols.foreach{ c =>
          if (bucketSpec.bucketColumnNames.contains(c) && !setValidBitIdx) {
            val bitIdx = joinColToIdxMap.get(c).get
            validBitIdx += bitIdx
            setValidBitIdx = true
            val cachingRelName = tblName + "_" + joinColToIdxMap.get(c).get
            selectedJoinColForGPT = c
            if (sqlContext.sharedState.cacheManager.cachedDataForGPT.contains(tblName, bitIdx.toString)) {
              sqlContext.sharedState.cacheManager.setTaskForCachingGPTable()
              sqlContext.sharedState.cacheManager.setTableNameForCachingGPT(tblName)
              sqlContext.sharedState.cacheManager.setBitIdxForRetrievingLogicalPlan(bitIdx)
            }
            /*
            if(sqlContext.sharedState.cacheManager.cachedDataForGPT.contains(tblName, joinColToIdxMap.get(elem).get.toString)) {
              logDebug("Set Requested Cache for GPT Table: " + tblName + " -> " + cachingRelName)
              sqlContext.sharedState.cacheManager.setrequestedCacheForGPTTable(cachingRelName)
            } else {
              sqlContext.sharedState.cacheManager.setrequestedCacheForGPTTable("None")
            }
            */
          }
        }
        if(validBitIdx.isEmpty){
          // run default mode
          logDebug(s"[SCAN without GPT Column] Run in default validBitIdx Selection!")
          validBitIdx += 0
        }

      } else if (joinColumns.isEmpty && validBitIdx.isEmpty) {

        /*
        if(dataFilters.size > 0) {

          dataFilters.foreach { c =>
            // (isnull(cr_item_sk#3) || isnotnull(cr_item_sk#3))
            val str = c.simpleString
            val newStr = str.substring(1, str.length-1)
            val tmpFilterCols = new collection.mutable.ArrayBuffer[String]()
            val tmpFilterColNames = new collection.mutable.HashSet[String]()
            val tmpFilterExprs = new collection.mutable.ArrayBuffer[String]()

            str.substring(1, str.length-1).split('|').foreach(c => if(c.contains('(')) {tmpFilterCols += c.trim})
            tmpFilterCols.foreach{c =>
              val expr = c.split('(')(0)
              val colNameWithNum = c.split('(')(1).split(')')(0)
              val colTok = colNameWithNum.split('#')
              val colName = colTok(0)
              val colIdx = colTok(1)
              tmpFilterExprs += expr
              tmpFilterColNames += colName
            }

            if (tmpFilterColNames.size == 1 && tmpFilterExprs.size == 2 &&
              tmpFilterExprs.contains("isnull") && tmpFilterExprs.contains("isnotnull")) {

              // FOR GPT CACHING TABLE

              var setValidBitIdx = false
              tmpFilterColNames.foreach{c =>
                if (bucketSpec.bucketColumnNames.contains(c) && !setValidBitIdx) {
                  logDebug("set validBitIdx due to filter column matches GPT partitioning column: (" + c + " -> " + joinColToIdxMap.get(c).get + ")")
                  validBitIdx += joinColToIdxMap.get(c).get
                  setValidBitIdx = true
                  containingSubPartitionForOuterJoin = true
                }
              }
            }
          }
          */
        var cachingTableName = "None"
        var cachingBitIdx = -1
        val isGPTCachingTask = sqlContext.sharedState.cacheManager.getTaskForCachingGPTable()
        if(isGPTCachingTask) {
          logDebug(s"[CACHING] Run for caching GPT Partitioned table's validBitIdx Selection! [tblName: " + tblName + ", cachedName: " + tblName + "_0" + "]")
          cachingBitIdx = sqlContext.sharedState.cacheManager.getBitIdxForRetrievingLogicalPlan()
          cachingTableName = tblName + "_" + cachingBitIdx.toString
          validBitIdx += cachingBitIdx
          containingSubPartitionForOuterJoin = true
        } else {
          // run default mode
          logDebug(s"[SCAN without GPT Column] Run in default validBitIdx Selection for GPT Partitioned table! [tblName: " + tblName + "]")
          validBitIdx += 0
          if (sqlContext.sharedState.cacheManager.cachedDataForGPT.contains(tblName, "0")) {
            sqlContext.sharedState.cacheManager.setTaskForCachingGPTable()
            sqlContext.sharedState.cacheManager.setTableNameForCachingGPT(tblName)
            sqlContext.sharedState.cacheManager.setBitIdxForRetrievingLogicalPlan(0)
          }
          /*
          if(sqlContext.sharedState.cacheManager.cachedDataForGPT.contains(tblName, "0")) {
            logDebug("Set Requested Cache for GPT Table: " + tblName + " -> " + cachingRelName)
            sqlContext.sharedState.cacheManager.setrequestedCacheForGPTTable(cachingRelName)
          } else {
            sqlContext.sharedState.cacheManager.setrequestedCacheForGPTTable("None")
          }
          */
        }
      } else {
        logDebug(s"[SCAN without GPT Column] Run in default validBitIdx Selection!")
        validBitIdx += 0
      }

      joinColumns.foreach(c => logDebug(s"Join column in the query: " + c))
      validBitIdx.foreach(c => logDebug(s"validBitIdx: " + c))

      def genBitSets(numBits: Int): collection.mutable.ListBuffer[String] = {

        val bitSets = new collection.mutable.ListBuffer[String]()
        var minSize = Integer.MAX_VALUE;
        val maxNum = math.pow(numBits.toLong, 2).toInt
        for (num <- 0 to maxNum) {
          val bitStr = Integer.toString(num, 2)
          val s = bitStr.reverse.padTo(numBits, "0").reverse.toString()
          if (minSize > s.size) {
            minSize = s.size
          }
          if (s.size == minSize) {
            bitSets += s
          }
        }
        bitSets
      }

      def pow2(bitset: BitSet,
               numBits: Int) : Int = {

        var partitionIdentifier = 0
        for(i <- 0 to numBits-1) {
          val isBitSet = bitset.get(i)
          var bitVal = -1
          if(isBitSet) {
            bitVal = 1
          } else {
            bitVal = 0
          }
          partitionIdentifier += (bitVal * math.pow(2, i).toInt)
        }
        partitionIdentifier
      }

      def repeatChar(c: Char, n: Int): String = c.toString * n

      val bitsets = genBitSets(numGPTCols)
      val candidateBitSets = new collection.mutable.ListBuffer[String]()
      var validBitSets = new collection.mutable.ListBuffer[String]()

      val pattern = "[0-9]".r
      for (bitset <- bitsets) {
        // logDebug(s"cretedBucketedReadRDD | candidate bitset: " + bitset)
        val tempArr = new Array[Int](numGPTCols)
        val str = (pattern findAllIn bitset).toList
        val idx = 0
        for (i <- 0 to str.size-1) {
          tempArr(i) = str(i).toInt
        }
        candidateBitSets += tempArr.mkString("")
      }

      for (elem <- candidateBitSets) {
        var ret = true
        validBitIdx.foreach(c => if (elem.reverse.charAt(c) == '0') ret = false)
        if (ret) {
          validBitSets += elem
        }
      }

      validBitSets.foreach(c => logDebug(s"Selected Sub-Partitions: " + c))

      val dupElimBySubPartition = sparkContext.getConf.get("spark.GPT.enableSubPartition", "true")
      if(dupElimBySubPartition == "false") {
        validBitSets.clear()
        candidateBitSets.foreach(c => validBitSets += c)
        containingSubPartitionForOuterJoin = false
      }

      var nullSubPartitionBitVals = new scala.collection.mutable.ListBuffer[Int]()
      var nullSubPartitions = new scala.collection.mutable.ListBuffer[String]()
      val nullPartitionFile = new scala.collection.mutable.ListBuffer[PartitionedFile]

      var bitV = validBitSets.map { c =>
        val bits = new BitSet(numGPTCols)
        var idx = 0
        c.reverse.foreach{b => if (b == '1') bits.set(idx)
          idx += 1}
        nullSubPartitionBitVals += bits.pow2().toInt
        bits.pow2()
      }

      nullSubPartitionBitVals.foreach{ b =>
        val bitVal = b
        val numPad = 5 - bitVal.toString.length
        val bitString = repeatChar('0', numPad) + bitVal
        nullSubPartitions += bitString
      }

      if (containingSubPartitionForOuterJoin) {
        nullSubPartitionBitVals.foreach( c => logDebug(s"nullSubPartitionBitVals: " + c))
        nullSubPartitions.foreach( c => logDebug(s"nullSubPartitionBitVals: " + c))
      }

      // hdfs://10.150.20.72:8020/tpcds/SS_2_C/GPT-store_sales_text-0-00000.csv
      var partitionMap = new collection.mutable.HashMap[Int, collection.mutable.ListBuffer[PartitionedFile]]()
      selectedPartitions.foreach { p =>
        p.files.foreach{ f =>
          val tokens = f.getPath.toString().split("-")

          val prefix = tokens(0)
          val tblName = tokens(1)
          val bitVal = tokens(2).toInt
          val naivePartitionID = tokens(3).substring(0, 5)
          val partitionID = GPTUtils.getPartitionID(tokens(3).substring(0, 5))

          if (containingSubPartitionForOuterJoin) {

            if (bitVal == 0) {
              logDebug(s"candidate null subpartition: " + f.getPath.getName)
              var findNullSubPartition = false
              nullSubPartitions.foreach(c => if (c == naivePartitionID) findNullSubPartition = true)

              if (findNullSubPartition) {
                logDebug("Handling outer join! adding partitions: 0-" + naivePartitionID)
                val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
                val partitionedFile = PartitionedFile(p.values, f.getPath.toUri.toString, 0, f.getLen, hosts)

                if (!partitionMap.contains(partitionID)) {
                  var partitionedFiles = new collection.mutable.ListBuffer[PartitionedFile]()
                  partitionedFiles += partitionedFile
                  partitionMap += (partitionID -> partitionedFiles)
                } else {
                  partitionMap(partitionID) += partitionedFile
                }
              }
            }
          }

          if (bitV.contains(bitVal)) {

            val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
            val partitionedFile = PartitionedFile(p.values, f.getPath.toUri.toString,
              0, f.getLen, hosts)

            if (!partitionMap.contains(partitionID)) {
              var partitionedFiles = new collection.mutable.ListBuffer[PartitionedFile]()
              partitionedFiles += partitionedFile
              partitionMap += (partitionID -> partitionedFiles)
            } else {
              partitionMap(partitionID) += partitionedFile
            }
          }
        }
      }

      /*
      if (containingSubPartitionForOuterJoin) {

        if (bitVal == 0) {
          logDebug(s"candidate null subpartition: " + f.getPath.getName)
          var findNullSubPartition = false
          nullSubPartitions.foreach(c => if (c == naivePartitionID) findNullSubPartition = true)

          if (findNullSubPartition) {
            logDebug("Handling outer join! adding partitions: 0-" + naivePartitionID)
            val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
            val partitionedFile = PartitionedFile(p.values, f.getPath.toUri.toString, 0, f.getLen, hosts)
            nullPartitionFile += partitionedFile
          }
        }
      }

      if (containingSubPartitionForOuterJoin) {
        val nullBitVal = 0
        partitionMap.foreach{ p =>
          p._2 ++ nullPartitionFile
        }
      }
      */

      val filePartitions = Seq.tabulate(bucketSpec.numBuckets) { bucketId =>
        FilePartition(bucketId, partitionMap.getOrElse(bucketId, Nil))
      }

      sparkContext.conf.set("GPTTask", "true")
      // filePartitions.foreach{ c => c.files.foreach(p => logDebug(p.filePath))}

      new FileScanRDD(fsRelation.sparkSession, readFile, filePartitions)

    } else {

      val bucketed =
      selectedPartitions.flatMap { p =>
        p.files.map { f =>
          val hosts = getBlockHosts(getBlockLocations(f), 0, f.getLen)
          //logDebug(s"file: " + f.getPath.toUri)
          PartitionedFile(p.values, f.getPath.toUri.toString, 0, f.getLen, hosts)
        }
      }.groupBy { f =>
        BucketingUtils
          .getBucketId(new Path(f.filePath).getName)
          .getOrElse(sys.error(s"Invalid bucket file ${f.filePath}"))
      }

    val filePartitions = Seq.tabulate(bucketSpec.numBuckets) { bucketId =>
      FilePartition(bucketId, bucketed.getOrElse(bucketId, Nil))
    }

      sqlContext.sharedState.cacheManager.setrequestedCacheForGPTTable("None")
      sqlContext.sharedState.cacheManager.makeGPTCacheStatusDefault()

    new FileScanRDD(fsRelation.sparkSession, readFile, filePartitions)
    }
  }

  /**
   * Create an RDD for non-bucketed reads.
   * The bucketed variant of this function is [[createBucketedReadRDD]].
   *
   * @param readFile a function to read each (part of a) file.
   * @param selectedPartitions Hive-style partition that are part of the read.
   * @param fsRelation [[HadoopFsRelation]] associated with the read.
   */
  private def createNonBucketedReadRDD(
      readFile: (PartitionedFile) => Iterator[InternalRow],
      selectedPartitions: Seq[PartitionDirectory],
      fsRelation: HadoopFsRelation): RDD[InternalRow] = {
    logDebug(s"I thought this will be used for scanning CSV!")

    val defaultMaxSplitBytes =
      fsRelation.sparkSession.sessionState.conf.filesMaxPartitionBytes
    val openCostInBytes = fsRelation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val defaultParallelism = fsRelation.sparkSession.sparkContext.defaultParallelism

    val totalBytes = selectedPartitions.flatMap(_.files.map(_.getLen + openCostInBytes)).sum
    val bytesPerCore = totalBytes / defaultParallelism

    val maxSplitBytes = Math.min(defaultMaxSplitBytes, Math.max(openCostInBytes, bytesPerCore))
    val splitFiles = selectedPartitions.flatMap { partition =>
      partition.files.flatMap { file =>
        //logDebug(s"Scanning CSV File: " + file.getPath.toUri)
        val blockLocations = getBlockLocations(file)
        if (fsRelation.fileFormat.isSplitable(
            fsRelation.sparkSession, fsRelation.options, file.getPath)) {
          (0L until file.getLen by maxSplitBytes).map { offset =>
            val remaining = file.getLen - offset
            val size = if (remaining > maxSplitBytes) maxSplitBytes else remaining
            val hosts = getBlockHosts(blockLocations, offset, size)
            PartitionedFile(
              partition.values, file.getPath.toUri.toString, offset, size, hosts)
          }
        } else {
          val hosts = getBlockHosts(blockLocations, 0, file.getLen)
          Seq(PartitionedFile(
            partition.values, file.getPath.toUri.toString, 0, file.getLen, hosts))
        }
      }
    }.toArray.sortBy(_.length)(implicitly[Ordering[Long]].reverse)

    val partitions = new ArrayBuffer[FilePartition]
    val currentFiles = new ArrayBuffer[PartitionedFile]
    var currentSize = 0L

    /** Close the current partition and move to the next. */
    def closePartition(): Unit = {
      if (currentFiles.nonEmpty) {
        val newPartition =
          FilePartition(
            partitions.size,
            currentFiles.toArray.toSeq) // Copy to a new Array.
        partitions += newPartition
      }
      currentFiles.clear()
      currentSize = 0
    }

    // Assign files to partitions using "First Fit Decreasing" (FFD)
    splitFiles.foreach { file =>
      if (currentSize + file.length > maxSplitBytes) {
        closePartition()
      }
      // Add the given file to the current partition.
      currentSize += file.length + openCostInBytes
      currentFiles += file
    }
    closePartition()

    sqlContext.sharedState.cacheManager.makeGPTCacheStatusDefault()
    new FileScanRDD(fsRelation.sparkSession, readFile, partitions)
  }

  private def getBlockLocations(file: FileStatus): Array[BlockLocation] = file match {
    case f: LocatedFileStatus => f.getBlockLocations
    case f => Array.empty[BlockLocation]
  }

  // Given locations of all blocks of a single file, `blockLocations`, and an `(offset, length)`
  // pair that represents a segment of the same file, find out the block that contains the largest
  // fraction the segment, and returns location hosts of that block. If no such block can be found,
  // returns an empty array.
  private def getBlockHosts(
      blockLocations: Array[BlockLocation], offset: Long, length: Long): Array[String] = {
    val candidates = blockLocations.map {
      // The fragment starts from a position within this block
      case b if b.getOffset <= offset && offset < b.getOffset + b.getLength =>
        b.getHosts -> (b.getOffset + b.getLength - offset).min(length)

      // The fragment ends at a position within this block
      case b if offset <= b.getOffset && offset + length < b.getLength =>
        b.getHosts -> (offset + length - b.getOffset).min(length)

      // The fragment fully contains this block
      case b if offset <= b.getOffset && b.getOffset + b.getLength <= offset + length =>
        b.getHosts -> b.getLength

      // The fragment doesn't intersect with this block
      case b =>
        b.getHosts -> 0L
    }.filter { case (hosts, size) =>
      size > 0L
    }

    if (candidates.isEmpty) {
      Array.empty[String]
    } else {
      val (hosts, _) = candidates.maxBy { case (_, size) => size }
      hosts
    }
  }

  override lazy val canonicalized: FileSourceScanExec = {
    FileSourceScanExec(
      relation,
      output.map(QueryPlan.normalizeExprId(_, output)),
      requiredSchema,
      QueryPlan.normalizePredicates(partitionFilters, output),
      QueryPlan.normalizePredicates(dataFilters, output),
      joinColumns,
      None)
  }
}
