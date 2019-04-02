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

import java.util.{Date, UUID}

import com.sun.prism.PixelFormat.DataType

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.output.{FileOutputCommitter, FileOutputFormat}
import org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.internal.io.{FileCommitProtocol, SparkHadoopWriterUtils}
import org.apache.spark.internal.io.FileCommitProtocol.TaskCommitMessage
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.catalog.{BucketSpec, ExternalCatalogUtils}
import org.apache.spark.sql.catalyst.catalog.CatalogTypes.TablePartitionSpec
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.{CaseInsensitiveMap, DateTimeUtils}
import org.apache.spark.sql.execution.{QueryExecution, SQLExecution, SortExec}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.util.{SerializableConfiguration, Utils}
import org.apache.spark.sql.types._


/** A helper object for writing FileFormat data out to a location. */
object FileFormatWriter extends Logging {

  /**
   * Max number of files a single task writes out due to file size. In most cases the number of
   * files written should be very small. This is just a safe guard to protect some really bad
   * settings, e.g. maxRecordsPerFile = 1.
   */
  private val MAX_FILE_COUNTER = 1000 * 1000

  /** Describes how output files should be placed in the filesystem. */
  case class OutputSpec(
    outputPath: String, customPartitionLocations: Map[TablePartitionSpec, String])

  /** A shared job description for all the write tasks. */
  private class WriteJobDescription(
      val uuid: String,  // prevent collision between different (appending) write jobs
      val serializableHadoopConf: SerializableConfiguration,
      val outputWriterFactory: OutputWriterFactory,
      val allColumns: Seq[Attribute],
      val dataColumns: Seq[Attribute],
      val partitionColumns: Seq[Attribute],
      val bucketIdExpression: Option[Expression],
      val path: String,
      val customPartitionLocations: Map[TablePartitionSpec, String],
      val maxRecordsPerFile: Long,
      val timeZoneId: String)
    extends Serializable {

    assert(AttributeSet(allColumns) == AttributeSet(partitionColumns ++ dataColumns),
      s"""
         |All columns: ${allColumns.mkString(", ")}
         |Partition columns: ${partitionColumns.mkString(", ")}
         |Data columns: ${dataColumns.mkString(", ")}
       """.stripMargin)
  }

  private class GPTPartitioningDescription(
     val isGPTPartitioning: Boolean,
     val isNeedSort: Boolean,
     val tblName: String,
     val bucketCols: ListBuffer[String],
     val bucketColIdxs: ListBuffer[Integer],
     val numBuckets: Integer)
    extends Serializable {

  }

  /** The result of a successful write task. */
  private case class WriteTaskResult(commitMsg: TaskCommitMessage, updatedPartitions: Set[String])

  /**
   * Basic work flow of this command is:
   * 1. Driver side setup, including output committer initialization and data source specific
   *    preparation work for the write job to be issued.
   * 2. Issues a write job consists of one or more executor side tasks, each of which writes all
   *    rows within an RDD partition.
   * 3. If no exception is thrown in a task, commits that task, otherwise aborts that task;  If any
   *    exception is thrown during task commitment, also aborts that task.
   * 4. If all tasks are committed, commit the job, otherwise aborts the job;  If any exception is
   *    thrown during job commitment, also aborts the job.
   */
  def write(
      sparkSession: SparkSession,
      queryExecution: QueryExecution,
      fileFormat: FileFormat,
      committer: FileCommitProtocol,
      outputSpec: OutputSpec,
      hadoopConf: Configuration,
      partitionColumnNames: Seq[String],
      bucketSpec: Option[BucketSpec],
      refreshFunction: (Seq[TablePartitionSpec]) => Unit,
      options: Map[String, String]): Unit = {

    //val coresite_path = sparkSession.sparkContext.getConf.get("spark.GPT.haddopConfPath") + "core-site.xml";
    //hadoopConf.addResource(coresite_path);
    //hadoopConf.set("fs.defaultFS", "hdfs://10.150.20.24:8021");

    def getColIdx(DBName: String, tblName: String, colName: String): Int = {
      var idx = 0
      var retVal = 0

      val tblSchema = sparkSession.sqlContext.sessionState.
        catalog.externalCatalog.getTable(DBName, tblName).schema
      for (elem <- tblSchema) {
        if (elem.name == colName) {
          retVal = idx
        } else {
          idx += 1
        }
      }
      retVal
    }

    val job = Job.getInstance(hadoopConf)
    job.setOutputKeyClass(classOf[Void])
    job.setOutputValueClass(classOf[InternalRow])
    FileOutputFormat.setOutputPath(job, new Path(outputSpec.outputPath))

    val allColumns = queryExecution.executedPlan.output
    // Get the actual partition columns as attributes after matching them by name with
    // the given columns names.
    val partitionColumns = partitionColumnNames.map { col =>
      val nameEquality = sparkSession.sessionState.conf.resolver
      allColumns.find(f => nameEquality(f.name, col)).getOrElse {
        throw new RuntimeException(
          s"Partition column $col not found in schema ${queryExecution.executedPlan.schema}")
      }
    }
    val partitionSet = AttributeSet(partitionColumns)
    val dataColumns = allColumns.filterNot(partitionSet.contains)

    val bucketIdExpression = bucketSpec.map { spec =>
      val bucketColumns = spec.bucketColumnNames.map(c => dataColumns.find(_.name == c).get)
      // Use `HashPartitioning.partitionIdExpression` as our bucket id expression, so that we can
      // guarantee the data distribution is same between shuffle and bucketed data source, which
      // enables us to only shuffle one side when join a bucketed table and a normal one.
      HashPartitioning(bucketColumns, spec.numBuckets).partitionIdExpression
    }
    val sortColumns = bucketSpec.toSeq.flatMap {
      spec => spec.sortColumnNames.map(c => dataColumns.find(_.name == c).get)
    }

    val caseInsensitiveOptions = CaseInsensitiveMap(options)

    // Note: prepareWrite has side effect. It sets "job".
    val outputWriterFactory =
      fileFormat.prepareWrite(sparkSession, job, caseInsensitiveOptions, dataColumns.toStructType)

    val uuid = UUID.randomUUID().toString
    val maxRecordsPerFile = caseInsensitiveOptions.get("maxRecordsPerFile").map(_.toLong)
      .getOrElse(sparkSession.sessionState.conf.maxRecordsPerFile)
    val timeZoneId = caseInsensitiveOptions.get(DateTimeUtils.TIMEZONE_OPTION)
      .getOrElse(sparkSession.sessionState.conf.sessionLocalTimeZone)

    var bucketCols = new ListBuffer[String]();
    bucketSpec.map { spec =>
      spec.bucketColumnNames.foreach{c => bucketCols += dataColumns.find(_.name == c).get.name}
    }
    var numBuckets = 0
    bucketSpec.map { spec =>
      numBuckets = spec.numBuckets
    }

    logInfo(s"""[WriteJobDescription]
               |uuid: ${uuid}
               |Partition columns: ${partitionColumns.mkString(" ")}
               |bucketingCols : ${bucketCols.mkString( " and ")}
               |numBuckets: ${numBuckets}
               |bucketIdExpression: ${bucketIdExpression.mkString(" ")}
               |SortColumns: ${sortColumns.mkString(" ")}
               |outPath: ${outputSpec.outputPath.toString}
               |maxRecordsPerFile: ${maxRecordsPerFile}
               |timeZoneId: ${timeZoneId}
               |customPLoc: ${outputSpec.customPartitionLocations.toString()}""".stripMargin)
    val description = new WriteJobDescription(
      uuid,
      serializableHadoopConf = new SerializableConfiguration(job.getConfiguration),
      outputWriterFactory = outputWriterFactory,
      allColumns = allColumns,
      dataColumns = dataColumns,
      partitionColumns = partitionColumns,
      bucketIdExpression = bucketIdExpression,
      path = outputSpec.outputPath,
      customPartitionLocations = outputSpec.customPartitionLocations,
      maxRecordsPerFile,
      timeZoneId
    )

    // We should first sort by partition columns, then bucket id, and finally sorting columns.
    val requiredOrdering = partitionColumns ++ bucketIdExpression ++ sortColumns
    // the sort order doesn't matter
    val actualOrdering = queryExecution.executedPlan.outputOrdering.map(_.child)
    val orderingMatched = if (requiredOrdering.length > actualOrdering.length) {
      false
    } else {
      requiredOrdering.zip(actualOrdering).forall {
        case (requiredOrder, childOutputOrder) =>
          requiredOrder.semanticEquals(childOutputOrder)
      }
    }

    logInfo(s"actualOrdering: " + actualOrdering)
    logInfo(s"requiredOrdering: " + requiredOrdering)
    sparkSession.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

    SQLExecution.withNewExecutionId(sparkSession, queryExecution) {
      // This call shouldn't be put into the `try` block below because it only initializes and
      // prepares the job, any exception thrown from here shouldn't cause abortJob() to be called.
      committer.setupJob(job)


      var isGPTPartitioning = false
      var tblName = "none"
      var isNeedSort = false
      var DBName = "none"
      var bucketColIdxs = new ListBuffer[Integer]();

      options.foreach(x =>
        if (x._1=="GPT" && x._2=="true") { isGPTPartitioning = true }
        else if (x._1=="TableName") { tblName = x._2 }
        else if (x._1=="Sort" && x._2=="true") { isNeedSort = true }
        else if (x._1=="DBName") { DBName = x._2 }
      )

      if(isGPTPartitioning) {

        bucketCols.foreach(bucketCol => bucketColIdxs += getColIdx(DBName, tblName, bucketCol))

        var bucketColInfoString = ""
        logInfo(s"[GPT Partitioning] tableName: " + tblName + ", needSort: " + isNeedSort)
        bucketCols.foreach(bucketCol => bucketColInfoString
          += new String(bucketCol + ": " + getColIdx(DBName, tblName, bucketCol) + " | "))
        logInfo(s"[GPT Partitioning] partitioning column info: {" + bucketColInfoString + "}")

        bucketSpec.get.setGPTPartitioned()
        logInfo("[GPT Task] Table Partitioning!")
      }

      try {

          val rdd = if (orderingMatched || isGPTPartitioning) {
            logInfo("orderingMatched! NO Sort!")
            queryExecution.toRdd
          } else {
            logInfo("ordering UnMatched! DO Sort!")
            SortExec(
              requiredOrdering.map(SortOrder(_, Ascending)),
              global = false,
              child = queryExecution.executedPlan).execute()
          }

        val GPTDescription = new GPTPartitioningDescription(
          isGPTPartitioning,
          isNeedSort,
          tblName,
          bucketCols,
          bucketColIdxs,
          numBuckets
        )

          val ret = new Array[WriteTaskResult](rdd.partitions.length)
        sparkSession.sparkContext.runJob(
          rdd,
          (taskContext: TaskContext, iter: Iterator[InternalRow]) => {
            executeTask(
              GPTDescription,
              description = description,
              sparkStageId = taskContext.stageId(),
              sparkPartitionId = taskContext.partitionId(),
              sparkAttemptNumber = taskContext.attemptNumber(),
              committer,
              iterator = iter)
          },
          0 until rdd.partitions.length,
          (index, res: WriteTaskResult) => {
            committer.onTaskCommit(res.commitMsg)
            ret(index) = res
          })

        val commitMsgs = ret.map(_.commitMsg)
        val updatedPartitions = ret.flatMap(_.updatedPartitions)
          .distinct.map(PartitioningUtils.parsePathFragment)
        logInfo(s"Job committed.")
        committer.commitJob(job, commitMsgs)
        logInfo(s"Job has finisheds.")
        refreshFunction(updatedPartitions)
      } catch { case cause: Throwable =>
        logError(s"Aborting job ${job.getJobID}.", cause)
        committer.abortJob(job)
        throw new SparkException("Job aborted.", cause)
      }
    }
  }

  /** Writes data out in a single Spark task. */
  private def executeTask(
      GPTDescription: GPTPartitioningDescription,
      description: WriteJobDescription,
      sparkStageId: Int,
      sparkPartitionId: Int,
      sparkAttemptNumber: Int,
      committer: FileCommitProtocol,
      iterator: Iterator[InternalRow]): WriteTaskResult = {

    val jobId = SparkHadoopWriterUtils.createJobID(new Date, sparkStageId)
    val taskId = new TaskID(jobId, TaskType.MAP, sparkPartitionId)
    val taskAttemptId = new TaskAttemptID(taskId, sparkAttemptNumber)
    logInfo(s"jobID: " + jobId +", taskID: " + taskId + ", taskAttemptID: " + taskAttemptId)

    // Set up the attempt context required to use in the output committer.
    val taskAttemptContext: TaskAttemptContext = {
      // Set up the configuration object
      val hadoopConf = description.serializableHadoopConf.value
      hadoopConf.set("mapreduce.job.id", jobId.toString)
      hadoopConf.set("mapreduce.task.id", taskAttemptId.getTaskID.toString)
      hadoopConf.set("mapreduce.task.attempt.id", taskAttemptId.toString)
      hadoopConf.setBoolean("mapreduce.task.ismap", true)
      hadoopConf.setInt("mapreduce.task.partition", 0)

      if (GPTDescription.isGPTPartitioning) {
        hadoopConf.setBoolean("GPT", true)
      } else {
      hadoopConf.setBoolean("GPT", false)
      }

      new TaskAttemptContextImpl(hadoopConf, taskAttemptId)
    }

    committer.setupTask(taskAttemptContext)

    val writeTask =
      if (GPTDescription.isGPTPartitioning) {
        logInfo(s"GPT Partitioning will be executed.")
        new GPTPartitionWriteTask(GPTDescription, description, taskAttemptContext, committer)
      } else if (description.partitionColumns.isEmpty && description.bucketIdExpression.isEmpty) {
        logInfo(s"SingleDirectoryWriteTask will be executed.")
        new SingleDirectoryWriteTask(description, taskAttemptContext, committer)
      } else {
        logInfo(s"DynamicPartitionWriteTask will be executed.")
        new DynamicPartitionWriteTask(description, taskAttemptContext, committer)
      }

    try {
      Utils.tryWithSafeFinallyAndFailureCallbacks(block = {
        // Execute the task to write rows out and commit the task.
        val outputPartitions = writeTask.execute(iterator)
        writeTask.releaseResources()
        WriteTaskResult(committer.commitTask(taskAttemptContext), outputPartitions)
      })(catchBlock = {
        // If there is an error, release resource and then abort the task
        try {
          writeTask.releaseResources()
        } finally {
          committer.abortTask(taskAttemptContext)
          logError(s"Job $jobId aborted.")
        }
      })
    } catch {
      case t: Throwable =>
        throw new SparkException("Task failed while writing rows", t)
    }
  }

  /**
   * A simple trait for writing out data in a single Spark task, without any concerns about how
   * to commit or abort tasks. Exceptions thrown by the implementation of this trait will
   * automatically trigger task aborts.
   */
  private trait ExecuteWriteTask {
    /**
     * Writes data out to files, and then returns the list of partition strings written out.
     * The list of partitions is sent back to the driver and used to update the catalog.
     */
    def execute(iterator: Iterator[InternalRow]): Set[String]
    def releaseResources(): Unit
  }

  /** Writes data to a single directory (used for non-dynamic-partition writes). */
  private class SingleDirectoryWriteTask(
      description: WriteJobDescription,
      taskAttemptContext: TaskAttemptContext,
      committer: FileCommitProtocol) extends ExecuteWriteTask {

    logInfo(s"SingleDirectoryWriteTask!")
    private[this] var currentWriter: OutputWriter = _

    private def newOutputWriter(fileCounter: Int): Unit = {
      val ext = description.outputWriterFactory.getFileExtension(taskAttemptContext)
      val tmpFilePath = committer.newTaskTempFile(
        taskAttemptContext,
        None,
        f"-c$fileCounter%03d" + ext)

      currentWriter = description.outputWriterFactory.newInstance(
        path = tmpFilePath,
        dataSchema = description.dataColumns.toStructType,
        context = taskAttemptContext)
    }

    override def execute(iter: Iterator[InternalRow]): Set[String] = {
      logInfo(s"SingleDirectoryWriteTask --> execute")

      var fileCounter = 0
      var recordsInFile: Long = 0L
      newOutputWriter(fileCounter)
      while (iter.hasNext) {
        if (description.maxRecordsPerFile > 0 && recordsInFile >= description.maxRecordsPerFile) {
          fileCounter += 1
          assert(fileCounter < MAX_FILE_COUNTER,
            s"File counter $fileCounter is beyond max value $MAX_FILE_COUNTER")

          recordsInFile = 0
          releaseResources()
          newOutputWriter(fileCounter)
        }

        val internalRow = iter.next()
        currentWriter.write(internalRow)
        recordsInFile += 1
      }
      releaseResources()
      Set.empty
    }

    override def releaseResources(): Unit = {
      if (currentWriter != null) {
        try {
          currentWriter.close()
        } finally {
          currentWriter = null
        }
      }
    }
  }

  /**
   * Writes data to using dynamic partition writes, meaning this single function can write to
   * multiple directories (partitions) or files (bucketing).
   */
  private class DynamicPartitionWriteTask(
      desc: WriteJobDescription,
      taskAttemptContext: TaskAttemptContext,
      committer: FileCommitProtocol) extends ExecuteWriteTask {

    // currentWriter is initialized whenever we see a new key
    private var currentWriter: OutputWriter = _
    // logInfo(s"DynamicPartitionWriteTask!")

    /** Expressions that given partition columns build a path string like: col1=val/col2=val/... */
    private def partitionPathExpression: Seq[Expression] = {

      // logInfo(s"DynamicPartitionWriteTask | partitionPathExpression ==>")
      desc.partitionColumns.zipWithIndex.flatMap { case (c, i) =>
      // logInfo(s"PPE | partitionPathExpression: (c,i) --> [" + c + ", " + i + "]")

        val partitionName = ScalaUDF(
          ExternalCatalogUtils.getPartitionPathString _,
          StringType,
          Seq(Literal(c.name), Cast(c, StringType, Option(desc.timeZoneId))))

        // logInfo(s"PPE | c.name = " + c.name + ", " + ExternalCatalogUtils.getPartitionPathString _)
        // logInfo(s"PPE | partitionName = " + partitionName)

        if (i == 0) Seq(partitionName) else Seq(Literal(Path.SEPARATOR), partitionName)
      }
    }
    // logInfo(s"PPE | partitionPathExpression: " + partitionPathExpression.length)

    def getStagingPath(
      partColsAndBucketId: InternalRow,
      getPartitionPath: UnsafeProjection,
      fileCounter: Int,
      updatedPartitions: mutable.Set[String]): String = {

      val partDir = if (desc.partitionColumns.isEmpty) {
        None
      } else {
        Option(getPartitionPath(partColsAndBucketId).getString(0))
      }
      partDir.foreach(updatedPartitions.add)

      // If the bucketId expression is defined, the bucketId column is right after the partition
      // columns.
      val bucketId = if (desc.bucketIdExpression.isDefined) {
        BucketingUtils.bucketIdToString(partColsAndBucketId.getInt(desc.partitionColumns.length))
      } else {
        ""
      }

      // This must be in a form that matches our bucketing format. See BucketingUtils.
      val ext = f"$bucketId.c$fileCounter%03d" +
        desc.outputWriterFactory.getFileExtension(taskAttemptContext)

      val customPath = partDir match {
        case Some(dir) =>
          desc.customPartitionLocations.get(PartitioningUtils.parsePathFragment(dir))
        case _ =>
          None
      }
      val path = if (customPath.isDefined) {
        committer.newTaskTempFileAbsPath(taskAttemptContext, customPath.get, ext)
      } else {
        committer.newTaskTempFile(taskAttemptContext, partDir, ext)
      }
      val folder = path.split("part")(0)
      folder
    }

    /**
     * Opens a new OutputWriter given a partition key and optional bucket id.
     * If bucket id is specified, we will append it to the end of the file name, but before the
     * file extension, e.g. part-r-00009-ea518ad4-455a-4431-b471-d24e03814677-00002.gz.parquet
     *
     * @param partColsAndBucketId a row consisting of partition columns and a bucket id for the
     *                            current row.
     * @param getPartitionPath a function that projects the partition values into a path string.
     * @param fileCounter the number of files that have been written in the past for this specific
     *                    partition. This is used to limit the max number of records written for a
     *                    single file. The value should start from 0.
     * @param updatedPartitions the set of updated partition paths, we should add the new partition
     *                          path of this writer to it.
     */
    private def newOutputWriter(
        partColsAndBucketId: InternalRow,
        getPartitionPath: UnsafeProjection,
        fileCounter: Int,
        updatedPartitions: mutable.Set[String]): Unit = {
      // logInfo(s"DynamicPartitionWriteTask | newOutputWriter ==>")

      val partDir = if (desc.partitionColumns.isEmpty) {
        None
      } else {
        Option(getPartitionPath(partColsAndBucketId).getString(0))
      }
      // logInfo(s"NOW | partColsAndBucketId: " + partColsAndBucketId)
      // logInfo(s"NOW | partDir: " + getPartitionPath(partColsAndBucketId).getString(0).length)
      // logInfo(s"NOW | updatedPartitions: " + updatedPartitions.mkString(" "))

      partDir.foreach(updatedPartitions.add)

      // If the bucketId expression is defined, the bucketId column is right after the partition
      // columns.
      val bucketId = if (desc.bucketIdExpression.isDefined) {
        BucketingUtils.bucketIdToString(partColsAndBucketId.getInt(desc.partitionColumns.length))
      } else {
        ""
      }

      // logInfo(s"NOW | desc.partitionColumns: " + desc.partitionColumns.mkString(" "))
      // logInfo(s"NOW | bucket.getInt: " + partColsAndBucketId.getInt(desc.partitionColumns.length))
      // logInfo(s"NOW | bucketId: " + bucketId)
      // logInfo(s"NOW | fileCounter: " + fileCounter)
      val split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
      val ext1 = f"$bucketId.c$fileCounter%03d"
      // This must be in a form that matches our bucketing format. See BucketingUtils.
      val ext = ext1 +
        desc.outputWriterFactory.getFileExtension(taskAttemptContext)

      // logInfo(s"NOW | split: " + split)
      // logInfo(s"NOW | frontExt " + ext1 )
      // logInfo(s"NOW | FileExt: " + desc.outputWriterFactory.getFileExtension(taskAttemptContext))

      val customPath = partDir match {
        case Some(dir) =>
          desc.customPartitionLocations.get(PartitioningUtils.parsePathFragment(dir))
        case _ =>
          None
      }
      // logInfo(s"NOW | customPath: " + customPath)

      val path = if (customPath.isDefined) {
        committer.newTaskTempFileAbsPath(taskAttemptContext, customPath.get, ext)
      } else {
        committer.newTaskTempFile(taskAttemptContext, partDir, ext)
      }

      logInfo(s"NOW | path: " + path)

      currentWriter = desc.outputWriterFactory.newInstance(
        path = path,
        dataSchema = desc.dataColumns.toStructType,
        context = taskAttemptContext)
    }

    override def execute(iter: Iterator[InternalRow]): Set[String] = {

      logInfo(s"DynamicPartitionWriteTask!")

      val getPartitionColsAndBucketId = UnsafeProjection.create(
        desc.partitionColumns ++ desc.bucketIdExpression, desc.allColumns)

      // Generates the partition path given the row generated by `getPartitionColsAndBucketId`.
      val getPartPath = UnsafeProjection.create(
        Seq(Concat(partitionPathExpression)), desc.partitionColumns)

      // logInfo(s"exec | partitionPathExpression: " + partitionPathExpression.length)
      // logInfo(s"exec | desc.partitionColumns: " + desc.partitionColumns.length)

      // Returns the data columns to be written given an input row
      val getOutputRow = UnsafeProjection.create(desc.dataColumns, desc.allColumns)

      // logInfo(s"exec | desc.dataColumns: " + desc.dataColumns)
      // logInfo(s"exec | desc.allColumns: " + desc.allColumns)
      // logInfo(s"exec | getOutputRow: " + getOutputRow.toString())

      // If anything below fails, we should abort the task.
      var recordsInFile: Long = 0L
      var fileCounter = 0
      var currentPartColsAndBucketId: UnsafeRow = null
      val updatedPartitions = mutable.Set[String]()

      // logInfo(s"exec | recordsInFile: " + recordsInFile)
      // logInfo(s"exec | fileCounter: " + fileCounter)
      // logInfo(s"exec | currentPartColsAndBucketId: " + currentPartColsAndBucketId)
      // logInfo(s"exec | updatedPartitions: " + updatedPartitions.mkString(" "))

      for (row <- iter) {

        val nextPartColsAndBucketId = getPartitionColsAndBucketId(row)

        if (currentPartColsAndBucketId != nextPartColsAndBucketId) {
          // See a new partition or bucket - write to a new partition dir (or a new bucket file).
          currentPartColsAndBucketId = nextPartColsAndBucketId.copy()
          // logInfo(s"exec | Writing partition: $currentPartColsAndBucketId")
          val folder = getStagingPath(currentPartColsAndBucketId, getPartPath,
            fileCounter, updatedPartitions)

          recordsInFile = 0
          fileCounter = 0

          releaseResources()
          newOutputWriter(currentPartColsAndBucketId, getPartPath, fileCounter, updatedPartitions)

        } else if (desc.maxRecordsPerFile > 0 &&
            recordsInFile >= desc.maxRecordsPerFile) {

          val folder = getStagingPath(currentPartColsAndBucketId, getPartPath,
            fileCounter, updatedPartitions)
          
          // Exceeded the threshold in terms of the number of records per file.
          // Create a new file by increasing the file counter.
          recordsInFile = 0
          fileCounter += 1
          assert(fileCounter < MAX_FILE_COUNTER,
            s"File counter $fileCounter is beyond max value $MAX_FILE_COUNTER")

          releaseResources()
          newOutputWriter(currentPartColsAndBucketId, getPartPath, fileCounter, updatedPartitions)
        }

        currentWriter.write(getOutputRow(row))
        recordsInFile += 1
      }
      releaseResources()
      updatedPartitions.toSet
    }

    override def releaseResources(): Unit = {
      if (currentWriter != null) {
        try {
          currentWriter.close()
        } finally {
          currentWriter = null
        }
      }
    }
  }

  private class GPTPartitionWriteTask(
    GPTDescription: GPTPartitioningDescription,
    desc: WriteJobDescription,
    taskAttemptContext: TaskAttemptContext,
    committer: FileCommitProtocol) extends ExecuteWriteTask {

    logInfo(s"GPT Partitioning task!")

    // currentWriter is initialized whenever we see a new key
    private var currentWriter: OutputWriter = _
    //var rowBuf = scala.collection.mutable.Map.empty[Tuple2[Int, Int], scala.collection.mutable.ArrayBuffer[InternalRow]]
    var rowWriters = scala.collection.mutable.Map.empty[Tuple2[Int, Int], OutputWriter]

    /** Expressions that given partition columns build a path string like: col1=val/col2=val/... */
    private def partitionPathExpression: Seq[Expression] = {
      desc.partitionColumns.zipWithIndex.flatMap { case (c, i) =>
        val partitionName = ScalaUDF(
          ExternalCatalogUtils.getPartitionPathString _,
          StringType,
          Seq(Literal(c.name), Cast(c, StringType, Option(desc.timeZoneId))))
        if (i == 0) Seq(partitionName) else Seq(Literal(Path.SEPARATOR), partitionName)
      }
    }

    def getStagingPath(
     partColsAndBucketId: InternalRow,
     getPartitionPath: UnsafeProjection,
     fileCounter: Int,
     updatedPartitions: mutable.Set[String]): String = {

      val partDir = if (desc.partitionColumns.isEmpty) {
        None
      } else {
        Option(getPartitionPath(partColsAndBucketId).getString(0))
      }
      partDir.foreach(updatedPartitions.add)

      val ext =
        desc.outputWriterFactory.getFileExtension(taskAttemptContext)

      val customPath = partDir match {
        case Some(dir) =>
          desc.customPartitionLocations.get(PartitioningUtils.parsePathFragment(dir))
        case _ =>
          None
      }
      val path = if (customPath.isDefined) {
        committer.newTaskTempFileAbsPath(taskAttemptContext, customPath.get, ext)
      } else {
        committer.newTaskTempFile(taskAttemptContext, partDir, ext)
      }
      val folder = path.split("GPT")(0)
      folder
    }

    // GPT partitioning fime name: GPT-{split}-{TableName}-{subPartitionID}-{PartitionID}
    // e.g., kappa = 2 --> PartitionIndex: {0 (00), 1 (01), 2 (10), 3(11)}
    //                     PartitionID:    0 ~ numBuckets-1
    //
    // for table store_sales with 4 buckets using 2 partitioning columns?
    // --------------------------------------------------------------------------------
    // Partition 0: GPT-{split}-store_sales-0-00000 | GPT-{split}-store_sales-1-00000
    //              GPT-{split}-store_sales-2-00000 | GPT-{split}-store_sales-3-00000
    // --------------------------------------------------------------------------------
    // Partition 1: GPT-{split}-store_sales-0-00001 | GPT-{split}-store_sales-1-00001
    //              GPT-{split}-store_sales-2-00001 | GPT-{split}-store_sales-3-00001
    // --------------------------------------------------------------------------------
    // Partition 2: GPT-{split}-store_sales-0-00002 | GPT-{split}-store_sales-1-00002
    //              GPT-{split}-store_sales-2-00002 | GPT-{split}-store_sales-3-00002
    // --------------------------------------------------------------------------------
    // Partition 3: GPT-{split}-store_sales-0-00003 | GPT-{split}-store_sales-1-00003
    //              GPT-{split}-store_sales-2-00003 | GPT-{split}-store_sales-3-00003
    // --------------------------------------------------------------------------------

    def fileName(tableName: String, subPartition: Int, PartitionID: Int) : String = {
      val prefix = "GPT-" + tableName + "-" + subPartition + "-"
      f"$prefix$subPartition%05d"
    }

    def init() : Unit = {

      val numSubPartitions = scala.math.pow(2.toDouble,
        GPTDescription.bucketCols.size.toDouble).toInt

      val getPartPath = UnsafeProjection.create(
        Seq(Concat(partitionPathExpression)), desc.partitionColumns)
      val updatedPartitions = mutable.Set[String]()

      for(i <- 1 to numSubPartitions-1 ; j <- 0 to GPTDescription.numBuckets-1) {
        //rowBuf.put((i,j), new scala.collection.mutable.ArrayBuffer[InternalRow]())

        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-" + i + "-"
        var prefix2 = f"$prefix$j%05d"
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName
        rowWriters.put((i,j), desc.outputWriterFactory.newInstance(path = path, dataSchema = desc.dataColumns.toStructType, context = taskAttemptContext));

      }

      // for handling outer join
      for(i <- 1 until numSubPartitions) {
        //rowBuf.put((0,i), new scala.collection.mutable.ArrayBuffer[InternalRow]())

        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-0-"
        var prefix2 = f"$prefix$i%05d"
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName
        rowWriters.put((0,i), desc.outputWriterFactory.newInstance(path = path, dataSchema = desc.dataColumns.toStructType, context = taskAttemptContext));
      }
    }

    /*
    def initWithWriterSet() : Unit = {

      val numSubPartitions = scala.math.pow(2.toDouble,
        GPTDescription.bucketCols.size.toDouble).toInt

      val getPartPath = UnsafeProjection.create(
        Seq(Concat(partitionPathExpression)), desc.partitionColumns)

      val updatedPartitions = mutable.Set[String]()

      var split0 = taskAttemptContext.getTaskAttemptID.getTaskID.getId
      var prefix0 = "GPT-" + split0 + "-" + GPTDescription.tblName + "-0-00000"
      var fileName0 = prefix0 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
      val path0 = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName0

      // writerSet += ((0,0) -> desc.outputWriterFactory.newInstance(
      //  path = path0,
      //  dataSchema = desc.dataColumns.toStructType,
      //  context = taskAttemptContext))

      desc.outputWriterFactory.newInstance(
        path = path0,
        dataSchema = desc.dataColumns.toStructType,
        context = taskAttemptContext).close()

      for(i <- 1 to numSubPartitions-1 ; j <- 0 to GPTDescription.numBuckets-1) {

        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-" + i + "-"
        var prefix2 = f"$prefix$j%05d"
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        val path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName

        writerSet += ((i,j) -> desc.outputWriterFactory.newInstance(
          path = path,
          dataSchema = desc.dataColumns.toStructType,
          context = taskAttemptContext))
      }

      // for handling outer join
      for(i <- 1 to numSubPartitions) {
        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-" + i + "-"
        var prefix2 = f"$prefix$i%05d"
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        val path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName

        writerSet += ((0,i) -> desc.outputWriterFactory.newInstance(
          path = path,
          dataSchema = desc.dataColumns.toStructType,
          context = taskAttemptContext))
      }
    }
    */
    /*
    def finalyzing_GPT_Partitions() : Unit = {
      // for handling all Null case

      // Generates the partition path given the row generated by `getPartitionColsAndBucketId`.
      val getPartPath = UnsafeProjection.create(
        Seq(Concat(partitionPathExpression)), desc.partitionColumns)

      // Returns the data columns to be written given an input row
      val getOutputRow = UnsafeProjection.create(desc.dataColumns, desc.allColumns)

      val numSubPartitions = scala.math.pow(2.toDouble,
        GPTDescription.bucketCols.size.toDouble).toInt

      val updatedPartitions = mutable.Set[String]()

      var writers = scala.collection.mutable.ArrayBuffer[OutputWriter]()
      //var overFlowIndex = 0;

      for(i <- 1 until numSubPartitions) {
        // hdfs://10.150.20.24:8021/tpcds/test_CB/GPT-0-test_text-0-00000-{OverFlow Index}.csv
        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-0-"
        var prefix2 = f"$prefix$i%05d"

        //var overFlowTicket = f"$overFlowIndex%05d"
        //var fileName = prefix2 + "-" + overFlowTicket + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName

        val currentRowBuf = rowBuf.getOrElse((0,i), null)

        var writer = desc.outputWriterFactory.newInstance(
          path = path,
          dataSchema = desc.dataColumns.toStructType,
          context = taskAttemptContext)
        logDebug(s"Writing [" + i + "] partitions with " + currentRowBuf.size + " rows to [" + path + "]")

        /*
        var count = 0;
        currentRowBuf.foreach{ r =>
          if(count > 3000000) {
            //writers += writer
            writer.close()
            overFlowIndex += 1
            overFlowTicket = f"$overFlowIndex%05d"
            fileName = prefix2 + "-" + overFlowTicket + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
            path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName
            writer = desc.outputWriterFactory.newInstance(
              path = path,
              dataSchema = desc.dataColumns.toStructType,
              context = taskAttemptContext)
            count = 0;
            logInfo(s"Writing [" + i + "] partitions: overFlow Bucket " + overFlowIndex + " Enabled.")
          }
          writer.write(r)
          count += 1
          */
        currentRowBuf.foreach{ r => writer.write(r) }
        currentRowBuf.clear()
        writer.close()
      }
        //writers += writer

      // overFlowIndex = 0

      for(i <- 1 to numSubPartitions-1 ; j <- 0 to GPTDescription.numBuckets-1) {

        // hdfs://10.150.20.24:8021/tpcds/test_CB/GPT-0-test_text-0-00000-{OverFlow Index}.csv
        var split = taskAttemptContext.getTaskAttemptID.getTaskID.getId
        var prefix = "GPT-" + split + "-" + GPTDescription.tblName + "-" + i + "-"
        var prefix2 = f"$prefix$j%05d"
        //var overFlowTicket = f"$overFlowIndex%05d"
        //var fileName = prefix2 + "-" + overFlowTicket + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var fileName = prefix2 + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
        var path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName

        // logInfo(s"GPT exec | createFileName: " + fileName)
        // logInfo(s"GPT exec | stagingPath: " + path)

        var writer = desc.outputWriterFactory.newInstance(
          path = path,
          dataSchema = desc.dataColumns.toStructType,
          context = taskAttemptContext)

        val currentRowBuf = rowBuf.getOrElse((i,j), null)
        logDebug(s"Writing [" + i + "," + j +"] partitions with " + currentRowBuf.size + " rows to [" + path + "]")
        /*
        var count = 0;
        currentRowBuf.foreach{ r =>
          if(count > 3000000) {
            writer.close()
            overFlowIndex += 1
            overFlowTicket = f"$overFlowIndex%05d"
            fileName = prefix2 + "-" + overFlowTicket + desc.outputWriterFactory.getFileExtension(taskAttemptContext)
            path = getStagingPath(null, getPartPath, 0, updatedPartitions) + fileName
            writer = desc.outputWriterFactory.newInstance(
              path = path,
              dataSchema = desc.dataColumns.toStructType,
              context = taskAttemptContext)
            count = 0;
            logInfo(s"Writing [" + i + "," + j +"] partitions: overFlow Bucket " + overFlowIndex + " Enabled.")
          }
          writer.write(r)
          count += 1
        }
        */
        var cnt = 0;
        currentRowBuf.foreach{ r =>
          /*
          var rowProjected = ""
          (0 to 10).foreach( k => rowProjected += r.getInt(k) + ", ")
          logDebug(s"Written row [count: " + cnt + "] : " + rowProjected)
          cnt += 1
          */
          writer.write(r) }
        currentRowBuf.clear()
        writer.close()
        //writers += writer
        // writers.put((i, j), newWriter)
      }
      //writers.foreach( c => c.close())
    }

    def flushBuffer() : Unit = {}

    def getWriter(partitionID: Int, subPartitionID: Int, writers: scala.collection.mutable.Map[Tuple2[Int, Int], OutputWriter])
    : OutputWriter = {writers.getOrElse((partitionID, subPartitionID), null)}
    */
    def pow2(bitset: org.apache.spark.util.collection.BitSet, numBits: Int) : Int = {

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

    def bitSetString(bitset: org.apache.spark.util.collection.BitSet,
                     numBucketCols: Int) : String = {
      var ret = ""
      for (i <- 0 to numBucketCols-1) {
        if (bitset.get(i)) {
          ret += "1"
        } else {
          ret += "0"
        }
      }
      ret
    }

    // vec: (2), p: 3 => (4,1), (2,0) ==> (subPartitionID, partitionID)
    // logInfo(s"GPT Partitioning | partVec --> " + vec + ", p: " + p)
    var subPartitions = new scala.collection.mutable.ListBuffer[Tuple2[Int, Int]]()
    var subPartitionBitVecs = new scala.collection.mutable.ListBuffer[org.apache.spark.util.collection.BitSet]()
    var nullBitValIdx = new scala.collection.mutable.ListBuffer[Int]()
    val partBitVec = new org.apache.spark.util.collection.BitSet(GPTDescription.bucketCols.size)
    val partBitVecStorage = new scala.collection.mutable.ListBuffer[org.apache.spark.util.collection.BitSet]()

    def pmodInt(a: Int, n: Int): Int = {
      val r = a % n
      if (r < 0) {(r + n) % n} else r
    }

    //new Murmur3Hash(expressions), Literal(numPartitions)
    val buf =  new scala.collection.mutable.ListBuffer[Expression]()
    // (2451406,78349) / 120 / 2
    def partVec(vec: Seq[Int], p: Int, numBucketCols: Int) = {

      //val Trans = vec.map(x => if (x == 0) 0 else ((x%p) + 1))

      val Trans = vec.map{x =>
        if (x == 0)
          0
        else {
          pmodInt(Murmur3HashFunction.hash(x, org.apache.spark.sql.types.IntegerType, 42).toInt, p) + 1
        }
      }

      // (((2451406 % 120) + 1) ==> 47, ((78349 % 120) + 1) ==> 110) ==> (47, 110)

      val partSet = Trans.toSet.filter(x => x != 0)
      // Set(47, 110)

      val partVecs = partSet.map(y => (y, Trans.map(x => if (x == y) 1 else 0)))
      // Set((47,ListBuffer(1, 0)), (110,ListBuffer(0, 1)))

      subPartitions.clear()
      subPartitionBitVecs.clear()
      partBitVec.clear();
      partBitVecStorage.clear();
      for(x <- partVecs) {
        //printf("x._1: " + x._1 + ", x._2: " + x._2.mkString + "\n")
        var idx = 0
        for(y <- x._2.reverse) {
          //printf("idx: " + idx + ", y: " + y + "\n")
          if (y == 1) {
            partBitVec.set(idx)
          }
          idx += 1
        }
        subPartitionBitVecs += partBitVec
        subPartitions += Tuple2( pow2(partBitVec, numBucketCols), x._1 - 1)
        //printf("partBitVec: " + bitSetString(partBitVec, numBucketCols) + "\n")
        //printf("subPartitions: " + subPartitions.mkString + "\n")
        var tempBitVec = new org.apache.spark.util.collection.BitSet(numBucketCols)
        tempBitVec = tempBitVec | partBitVec;
        //printf("tempBitVec: " + bitSetString(tempBitVec, numBucketCols) + "\n")

        partBitVecStorage += tempBitVec
        partBitVec.clear()
      }

      //printf("partBitVecStorage: " + partBitVecStorage.foreach(c => printf(bitSetString(c, numBucketCols) + " | ")) + "\n")


      if (subPartitions.isEmpty) {
        subPartitions += Tuple2( 0, (math.pow(2, numBucketCols).toInt - 1) )

      } else {

        nullBitValIdx.clear()

        (0 until numBucketCols).foreach{ i =>
          var isAllBitValZero = partBitVecStorage.forall{ b =>
            //printf(i + "-th subpartitionBitVec: " + bitSetString(b, numBucketCols) + "\n")
            !(b.get(i)) }
          if (isAllBitValZero) {
            nullBitValIdx += i
          }
        }
        //printf("nullBitValIdx: " + nullBitValIdx.mkString + "\n")

        if (!nullBitValIdx.isEmpty) {
          val nullBitVec = new org.apache.spark.util.collection.BitSet(numBucketCols)
          (0 until numBucketCols).foreach { i =>
            if (nullBitValIdx.contains(i)) {
              nullBitVec.set(i)
            }
          }
          //printf("nullBitVec: " + bitSetString(nullBitVec, numBucketCols) + "\n")
          subPartitions += Tuple2( 0, pow2(nullBitVec, numBucketCols) )
        }
      }
      subPartitions
    }


    override def execute(iter: Iterator[InternalRow]): Set[String] = {

      var currentPartColsAndBucketId: UnsafeRow = null
      val updatedPartitions = mutable.Set[String]()
      val numBuckets = GPTDescription.numBuckets

      val numBucketCols = GPTDescription.bucketCols.size
      val numSubPartitions = scala.math.pow(2.toDouble,
        GPTDescription.bucketCols.size.toDouble).toInt

      val getOutputRow = UnsafeProjection.create(desc.dataColumns, desc.allColumns)

      /*
      partVecs.map {
          x =>
            var acc = 0
            x._2.zipWithIndex.foreach(y => acc += y._1 * math.pow(2, x._2.length - y._2 - 1).toInt)
            (acc, x._1 - 1)
        }
      */

      var numRow = 0
      init()
      var vec = new scala.collection.mutable.ListBuffer[Int]()

      while (iter.hasNext) {

        val row = iter.next()
        vec.clear()

        for (elem <- GPTDescription.bucketColIdxs) {

          val partitioningColVal = row.getInt(elem)

          if (partitioningColVal > 0) {
            vec += partitioningColVal
          } else {
            vec += 0
          }
        }

        //logDebug(s"debugString vec: " + vec + ", numBuckets: " + numBuckets)
        val partitionInfo = partVec(vec, numBuckets, numBucketCols)
        val cnt = 0;
        for(elem2 <- partitionInfo) {
          //val buf = rowBuf.getOrElse(elem2, null)
          //logDebug(s"colVec: [" + vec.toList + "] --> bitVec: [ " + elem2 + "]")
          //buf += row <-- failed
          //buf += getOutputRow(row).clone().asInstanceOf[InternalRow]
          //val copied = row.copy()
          //buf += copied
          val writer = rowWriters.getOrElse(elem2, null)
          if(writer != null) writer.write(getOutputRow(row))
          /*
          if(numRow > 500 & numRow < 1000) {
            var rowProjected = ""
            (0 to 10).foreach( k => rowProjected += copied.getInt(k) + ", ")
            logDebug(s"Copied row [numRow=" + numRow + "]: " + rowProjected)
            var rowStored = ""
            (0 to 10).foreach( k => rowStored += buf(buf.size-1).getInt(k) + ", ")
            logDebug(s"Stored row [numRow=" + numRow + "]: " + rowStored)
          }
          */
          numRow += 1
        }
        partitionInfo.clear()
      }

      //logInfo(s"total " + numRow + " rows are buffered! do finalyzing!")
      //finalyzing_GPT_Partitions()
      //logInfo(s"num outputs: " + numRow)
      //rowBuf.foreach{ c =>c._2.clear()}
      rowWriters.foreach{ c =>
        c._2.close()
      }
      updatedPartitions.toSet
    }

    /*
    override def execute(iter: Iterator[InternalRow]): Set[String] = {

      var currentPartColsAndBucketId: UnsafeRow = null
      val updatedPartitions = mutable.Set[String]()
      val numBuckets = GPTDescription.numBuckets

      val numBucketCols = GPTDescription.bucketCols.size
      val numSubPartitions = scala.math.pow(2.toDouble,
        GPTDescription.bucketCols.size.toDouble).toInt

      var numRow = 0
      initWithWriterSet()
      while (iter.hasNext) {

        val row = iter.next()
        var vec = new scala.collection.mutable.ListBuffer[Int]()

        val getOutputRow = UnsafeProjection.create(desc.dataColumns, desc.allColumns)

        for (elem <- GPTDescription.bucketColIdxs) {

          val partitioningColVal = row.getInt(elem)

          if (partitioningColVal > 0) {
            vec += partitioningColVal
          } else {
            vec += 0
          }
        }

        //logInfo(s"debugString vec: " + vec + ", numBuckets: " + numBuckets)
        val partitionInfo = partVec(vec, numBuckets, numBucketCols)

        for(elem2 <- partitionInfo) {
          val writer = writerSet.getOrElse(elem2, null)
          writer.write(getOutputRow(row))
          numRow += 1
        }
      }

      //logInfo(s"total " + numRow + " rows are buffered! do finalyzing!")
      //finalyzing_GPT_Partitions()
      logInfo(s"num outputs: " + numRow)
      releaseResources()
      updatedPartitions.toSet
    }
    */
    override def releaseResources(): Unit = {
        // do nothing!
    }
  }
}
