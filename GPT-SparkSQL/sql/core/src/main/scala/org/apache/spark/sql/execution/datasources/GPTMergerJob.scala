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
  * Created by Ronymin on 09/12/2017.
  */
case class GPTMergerJob(
       outputPath: Path,
       options: Map[String, String]) extends RunnableCommand {

  override def run(sparkSession: SparkSession): Seq[Row] = {

    val committer = FileCommitProtocol.instantiate(
      sparkSession.sessionState.conf.fileCommitProtocolClass,
      jobId = java.util.UUID.randomUUID().toString,
      outputPath = outputPath.toString,
      isAppend = true)

    merge(outputPath, sparkSession, sparkSession.sparkContext.hadoopConfiguration)

    // ADD MERGE TASK! (PARALLEL VERSION!)
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------

    case class MergeTaskResult(commitMsg: TaskCommitMessage, updatedPartitions: Set[String])

    class MergeJobDescription(
                               val uuid: String,  // prevent collision between different (appending) write jobs
                               val serializableHadoopConf: SerializableConfiguration,
                               val timeZoneId: String)
      extends Serializable {}

    def merge(
         outputPath: Path,
         sparkSession: SparkSession,
         hadoopConf: Configuration): Unit = {

      val coresite_path = sparkSession.sparkContext.getConf.get("spark.GPT.haddopConfPath") + "core-site.xml";
      hadoopConf.addResource(coresite_path);
      hadoopConf.set("fs.defaultFS", "hdfs://10.150.20.24:8021");
      val job = Job.getInstance(hadoopConf)
      job.setOutputKeyClass(classOf[Void])
      job.setOutputValueClass(classOf[InternalRow])
      val prefix_path_hdfs = sparkSession.conf.get("spark.GPT.HDFSPathPrefix")
      val jobOutputPath = new Path("/MergerTaskOutput")
      FileOutputFormat.setOutputPath(job, jobOutputPath)

      val caseInsensitiveOptions = CaseInsensitiveMap(options)

      val uuid = UUID.randomUUID().toString
      val timeZoneId = caseInsensitiveOptions.get(DateTimeUtils.TIMEZONE_OPTION)
        .getOrElse(sparkSession.sessionState.conf.sessionLocalTimeZone)

      val description = new MergeJobDescription(
        uuid,
        serializableHadoopConf = new SerializableConfiguration(job.getConfiguration),
        timeZoneId
      )

      val dfs = outputPath.getFileSystem(hadoopConf)

      var tableName = ""

      val tmpPathStr = "/GPT_tmp/"
      val tmpPath = new Path(tmpPathStr)

      val mergerInfoSrc = "/Merger_Tmp"
      val mergerInfoPath: Path = new Path(mergerInfoSrc)

      if (dfs.exists(tmpPath)) {
        dfs.delete(tmpPath, true)
      }
      dfs.mkdirs(tmpPath)

      if (dfs.exists(mergerInfoPath)) {
        dfs.delete(mergerInfoPath, true)
      }
      dfs.mkdirs(mergerInfoPath)

      val qualifiedOutputPath = outputPath.makeQualified(dfs.getUri, dfs.getWorkingDirectory)
      val status = dfs.listStatus {new org.apache.hadoop.fs.Path(qualifiedOutputPath.toString)}

      logDebug(s"qualifiedOutputPath: " + qualifiedOutputPath)

      var partitionMap = scala.collection.mutable.Map[String, scala.collection.mutable.ListBuffer[String]]()

      var isCSV = false
      var isParquet = false

      // hdfs://10.150.20.24:8021/tpcds/test_CB/GPT-0-test_text-0-00000-{OverFlow Index}.csv
      val totLength = status.length
      var curProcessedTask = 0
      logDebug(s"Get Ready for Merging: organizing partitionMap: (" + curProcessedTask + " / " + totLength + ")")
      for (partition <- status) {
        var partitionFile = partition.getPath.toString()
        logDebug(s"Check previous results: " + partitionFile)
        if (partitionFile.contains("GPT")) {

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
          logDebug(s"Will-Be-Merged File: " + partitionFile + ", partitionKey: " + partitionKey)

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

      // saving PartitionMap for merging GPT partitions

      val mergerTmpSrc = "/Merger_tmp/"
      val mergerPath: Path = new Path(mergerTmpSrc) with Serializable
      if (dfs.exists(mergerPath)) {
        dfs.delete(mergerPath, true)
      }
      dfs.mkdirs(mergerPath)
      logInfo(s"Processing PartitionMap with java.io.OutputStreamWriter for merging")
      var totPartitionMap = 0
      partitionMap.foreach { p =>
        totPartitionMap = totPartitionMap + p._2.length
      }

      var curPartitionMapPrepared = 0
      partitionMap.foreach { p =>
        val mergePath = partitionOutputMap(p._1)
        var inputFilePaths = ""
        p._2.foreach( c => inputFilePaths += (c + "|"))
        var mergerTaskInfo = mergePath + "|" + inputFilePaths
        logDebug(s"PartitionMap[" + curPartitionMapPrepared + "]: " + mergerTaskInfo)
        var infoFilePath = mergerTmpSrc + "/" + mergePath
        infoFilePath = infoFilePath.replace("snappy.parquet", "csv")
        val bw = new java.io.BufferedWriter(new java.io.OutputStreamWriter(dfs.create(new Path(infoFilePath)), "UTF-8"))
        bw.write(mergerTaskInfo)
        bw.close()
        curPartitionMapPrepared += 1
      }

      val mergerTaskInfoPath = "/Merger_tmp/GPT_tmp"
      val logicalPlan = sparkSession.sqlContext.read.textFile(mergerTaskInfoPath).logicalPlan
      val qe = new QueryExecution(sparkSession, logicalPlan)
      logInfo(s"Preparing Query Execution Plan for Merging Task")

      SQLExecution.withNewExecutionId(sparkSession, qe) {
        // This call shouldn't be put into the `try` block below because it only initializes and
        // prepares the job, any exception thrown from here shouldn't cause abortJob() to be called.
        committer.setupJob(job)

        val rdd = qe.toRdd

        logInfo(s"Launching Parallel Merging with # partitons: " + rdd.partitions.length)
        try {
          val ret = new Array[MergeTaskResult](rdd.partitions.length)

          sparkSession.sparkContext.runJob(
            rdd,
            (taskContext: TaskContext, iter: Iterator[InternalRow]) => {
              executeGPTMergeTask(
                description = description,
                isCSV,
                sparkStageId = taskContext.stageId(),
                sparkPartitionId = taskContext.partitionId(),
                sparkAttemptNumber = taskContext.attemptNumber(),
                committer,
                iterator = iter)
            },
            0 until rdd.partitions.length,
            (index, res: MergeTaskResult) => {
              committer.onTaskCommit(res.commitMsg)
              ret(index) = res
            })

          val commitMsgs = ret.map(_.commitMsg)
          committer.commitJob(job, commitMsgs)

          logInfo(s"Parallel Merging Job committed.")
        } catch { case cause: Throwable =>
          logError(s"Aborting job ${job.getJobID}.", cause)
          committer.abortJob(job)
          throw new SparkException("Job aborted.", cause)
        }
      }
    }

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------

    def executeGPTMergeTask(
           description: MergeJobDescription,
           isCSV: Boolean,
           sparkStageId: Int,
           sparkPartitionId: Int,
           sparkAttemptNumber: Int,
           committer: FileCommitProtocol,
           iterator: Iterator[InternalRow]): MergeTaskResult = {

      val jobId = SparkHadoopWriterUtils.createJobID(new Date, sparkStageId)
      val taskId = new TaskID(jobId, TaskType.MAP, sparkPartitionId)
      val taskAttemptId = new TaskAttemptID(taskId, sparkAttemptNumber)
      logDebug(s"jobID: " + jobId +", taskID: " + taskId + ", taskAttemptID: " + taskAttemptId)

      // Set up the attempt context required to use in the output committer.
      val taskAttemptContext: TaskAttemptContext = {
        // Set up the configuration object
        val hadoopConf = description.serializableHadoopConf.value
        hadoopConf.set("mapreduce.job.id", jobId.toString)
        hadoopConf.set("mapreduce.task.id", taskAttemptId.getTaskID.toString)
        hadoopConf.set("mapreduce.task.attempt.id", taskAttemptId.toString)
        hadoopConf.setBoolean("mapreduce.task.ismap", true)
        hadoopConf.setInt("mapreduce.task.partition", 0)

        new TaskAttemptContextImpl(hadoopConf, taskAttemptId)
      }

      committer.setupTask(taskAttemptContext)

      val mergeTask = new GPTMergeTask(description, isCSV, taskAttemptContext, committer)


      try {
        Utils.tryWithSafeFinallyAndFailureCallbacks(block = {
          // Execute the task to write rows out and commit the task.
          val outputPartitions = mergeTask.execute(iterator)
          mergeTask.releaseResources()
          MergeTaskResult(committer.commitTask(taskAttemptContext), outputPartitions)
        })(catchBlock = {
          // If there is an error, release resource and then abort the task
          try {
            mergeTask.releaseResources()
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

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------


    trait ExecuteMergeTask {

      def execute(iterator: Iterator[InternalRow]): Set[String]
      def releaseResources(): Unit
    }

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------

    class GPTMergeTask(
            desc: MergeJobDescription,
            isCSV: Boolean,
            taskAttemptContext: TaskAttemptContext,
            committer: FileCommitProtocol) extends ExecuteMergeTask {

      logInfo(s"Task for merging GPT Partitions!")
      var finishedPartitionMap = new mutable.HashSet[String]()

      override def execute(iter: Iterator[InternalRow]): Set[String] = {

        val updatedPartitions = mutable.Set[String]()
        val hadoopConf = desc.serializableHadoopConf.value
        val fs = FileSystem.get(hadoopConf)

        var mergeInfo = ""
        while(iter.hasNext) {

          val internalRow = iter.next()
          var rddStringIdx = 0
          if (!internalRow.isNullAt(rddStringIdx)) {
            rddStringIdx += 1
          }
          (0 to rddStringIdx-1).foreach(c => mergeInfo += internalRow.getString(c))

          logDebug("Processing partitions: " + mergeInfo)

          val tokens = mergeInfo.split('|')
          var partitionMap = new mutable.HashMap[String, ListBuffer[String]]()
          var outputPartitionIndex = new mutable.ListBuffer[Int]()

          var splitIndex = 0
          var curFileList: ListBuffer[String] = null
          var curMergedPartitionPath : String = "init"
          tokens.foreach{ c =>

            if (!c.contains("hdfs://")) {

              if (curMergedPartitionPath != "init") {
                partitionMap += (curMergedPartitionPath -> curFileList)
              }

              curMergedPartitionPath = c
              var inputFiles = new scala.collection.mutable.ListBuffer[String]()
              curFileList = inputFiles

            } else {
              curFileList += c
            }
          }

          var dupCheck = false
          partitionMap.foreach{ c =>
            if (c == curMergedPartitionPath) {
              dupCheck = true
            }
          }
          if (!dupCheck) {
            partitionMap += (curMergedPartitionPath -> curFileList)
          }

          logDebug("( " + taskAttemptContext.getTaskAttemptID + ") Print ParttionMap: " + partitionMap.toList)

          var isDoneFile = false;

          partitionMap.foreach{ i =>
              val mergedPartitionPath = new Path(i._1)
              var inputFiles = i._2
              isDoneFile = false;

              if (finishedPartitionMap.contains(mergedPartitionPath.getName)) {
                isDoneFile = true
              }

            logDebug("Writing Output Partition: " + mergedPartitionPath)
            logDebug("List of Input Partitions: " + inputFiles.zipWithIndex.toList)

            var dupTask = false
            if (fs.exists(mergedPartitionPath)) {
              dupTask = true
            }

              if (!isDoneFile && !dupTask) {

                if (isCSV) {

                  var numRowScanned = 0

                  val partitionPath: Array[Path] = inputFiles.map(f => new Path(f)).toArray[Path]

                  val dataOutputStream = fs.create(mergedPartitionPath)
                  val bw: java.io.BufferedWriter = new java.io.BufferedWriter(
                    new java.io.OutputStreamWriter(dataOutputStream, "UTF-8") with Serializable) with Serializable

                  partitionPath.foreach {
                    r =>
                      val filePath = r
                      val dataInputStream = fs.open(filePath)
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
                          //logDebug(s"readLine: " + readLine.get)
                          bw.write(readLine.get)
                          bw.newLine()
                          numRowScanned += 1
                        }
                      }
                      br.close()
                  }

                  bw.close
                  logInfo(s"Merging GPT partitions FINISHED [num rows: " + numRowScanned + "]")

                } else {

                  // PARQUET file format
                  val hadoopConf = desc.serializableHadoopConf.value
                  logInfo(s"( " + taskAttemptContext.getTaskAttemptID + ") Merging GPT partitions [Parquet]: " + mergedPartitionPath.toString)

                  var parquetFile = ParquetFileReader.open(hadoopConf, new Path(inputFiles(0)))
                  val fileMetaData = parquetFile.getFooter.getFileMetaData.getSchema()

                  val writer = new ParquetFileWriter(hadoopConf, fileMetaData, mergedPartitionPath)

                  writer.start()
                  inputFiles.foreach { m =>
                    parquetFile = ParquetFileReader.open(hadoopConf, new Path(m))
                    writer.appendFile(hadoopConf, new Path(m))
                    parquetFile.close()
                  }
                  writer.end(new util.HashMap[String, String]())
                }
              }
          }
        }
        updatedPartitions.toSet
      }

      override def releaseResources(): Unit = {
        // nothing to do
      }

    }

    val tmpPathStr = "/GPT_tmp/"
    val tmpPath = new Path(tmpPathStr)
    val coresite_path = sparkSession.sparkContext.getConf.get("spark.GPT.haddopConfPath") + "core-site.xml";
    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    hadoopConf.addResource(coresite_path);
    hadoopConf.set("fs.defaultFS", "hdfs://10.150.20.24:8021");
    val dfs = FileSystem.get(hadoopConf)
    dfs.delete(outputPath, true)
    dfs.rename(tmpPath, outputPath)

    dfs.delete(new Path("/Merger_Tmp"), true)
    dfs.delete(new Path("/Merger_tmp"), true)
    dfs.delete(new Path("/MergerTaskOutput"), true)

    Seq.empty[Row]
  }

}
