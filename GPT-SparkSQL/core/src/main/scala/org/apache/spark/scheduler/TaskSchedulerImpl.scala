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

package org.apache.spark.scheduler

import java.nio.ByteBuffer
import java.util.{Locale, Timer, TimerTask}
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

import scala.collection.{Set, mutable}
import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet}
import scala.util.Random
import org.apache.spark._
import org.apache.spark.TaskState.TaskState
import org.apache.spark.internal.Logging
import org.apache.spark.internal.config
import org.apache.spark.scheduler.SchedulingMode.SchedulingMode
import org.apache.spark.scheduler.TaskLocality.TaskLocality
import org.apache.spark.scheduler.local.LocalSchedulerBackend
import org.apache.spark.storage.BlockManagerId
import org.apache.spark.util.{AccumulatorV2, ThreadUtils, Utils}

/**
 * Schedules tasks for multiple types of clusters by acting through a SchedulerBackend.
 * It can also work with a local setup by using a `LocalSchedulerBackend` and setting
 * isLocal to true. It handles common logic, like determining a scheduling order across jobs, waking
 * up to launch speculative tasks, etc.
 *
 * Clients should first call initialize() and start(), then submit task sets through the
 * runTasks method.
 *
 * THREADING: [[SchedulerBackend]]s and task-submitting clients can call this class from multiple
 * threads, so it needs locks in public API methods to maintain its state. In addition, some
 * [[SchedulerBackend]]s synchronize on themselves when they want to send events here, and then
 * acquire a lock on us, so we need to make sure that we don't try to lock the backend while
 * we are holding a lock on ourselves.
 */
private[spark] class TaskSchedulerImpl private[scheduler](
    val sc: SparkContext,
    val maxTaskFailures: Int,
    private[scheduler] val blacklistTrackerOpt: Option[BlacklistTracker],
    isLocal: Boolean = false)
  extends TaskScheduler with Logging {

  import TaskSchedulerImpl._

  def this(sc: SparkContext) = {
    this(
      sc,
      sc.conf.get(config.MAX_TASK_FAILURES),
      TaskSchedulerImpl.maybeCreateBlacklistTracker(sc))
  }

  def this(sc: SparkContext, maxTaskFailures: Int, isLocal: Boolean) = {
    this(
      sc,
      maxTaskFailures,
      TaskSchedulerImpl.maybeCreateBlacklistTracker(sc),
      isLocal = isLocal)
  }

  val conf = sc.conf

  // How often to check for speculative tasks
  val SPECULATION_INTERVAL_MS = conf.getTimeAsMs("spark.speculation.interval", "100ms")

  // Duplicate copies of a task will only be launched if the original copy has been running for
  // at least this amount of time. This is to avoid the overhead of launching speculative copies
  // of tasks that are very short.
  val MIN_TIME_TO_SPECULATION = 100

  private val speculationScheduler =
    ThreadUtils.newDaemonSingleThreadScheduledExecutor("task-scheduler-speculation")

  // Threshold above which we warn user initial TaskSet may be starved
  val STARVATION_TIMEOUT_MS = conf.getTimeAsMs("spark.starvation.timeout", "15s")

  // CPUs to request per task
  val CPUS_PER_TASK = conf.getInt("spark.task.cpus", 1)

  // TaskSetManagers are not thread safe, so any access to one should be synchronized
  // on this class.
  private val taskSetsByStageIdAndAttempt = new HashMap[Int, HashMap[Int, TaskSetManager]]

  // Protected by `this`
  private[scheduler] val taskIdToTaskSetManager = new HashMap[Long, TaskSetManager]
  val taskIdToExecutorId = new HashMap[Long, String]

  @volatile private var hasReceivedTask = false
  @volatile private var hasLaunchedTask = false
  private val starvationTimer = new Timer(true)

  // Incrementing task IDs
  val nextTaskId = new AtomicLong(0)

  // IDs of the tasks running on each executor
  private val executorIdToRunningTaskIds = new HashMap[String, HashSet[Long]]

  def runningTasksByExecutors: Map[String, Int] = synchronized {
    executorIdToRunningTaskIds.toMap.mapValues(_.size)
  }

  // The set of executors we have on each host; this is used to compute hostsAlive, which
  // in turn is used to decide when we can attain data locality on a given host
  protected val hostToExecutors = new HashMap[String, HashSet[String]]

  protected val hostsByRack = new HashMap[String, HashSet[String]]

  protected val executorIdToHost = new HashMap[String, String]

  // Listener object to pass upcalls into
  var dagScheduler: DAGScheduler = null

  var backend: SchedulerBackend = null

  val mapOutputTracker = SparkEnv.get.mapOutputTracker


  val isGPTTask =  false
  /*
    // for GPT for cluster environment:
  // building partition ID map
  // GPT follows round-robin block location placement algorithm with consideration of
  // partition ID as P  and  replication factor R
  // from P = 0, GPT selects a sequence of R datanodes from datanode 1 to datanode R
  // in a round-robin manner
  var datanodeList : Array[String] = new Array[String](0)
  var isGPTTask =  false

  var hadoopConf = sc.hadoopConfiguration
  var numNodes = -1
  var numMaxPartition = -1
  var replicationFactor = -1

  var GPTTaskSets = new collection.mutable.HashMap[String, Boolean]()

  var GPTHost2PartitionMap = new mutable.HashMap[String, mutable.ListBuffer[Int]]()
  var GPTPartition2HostMap = new HashMap[Int, collection.mutable.ListBuffer[String]]

  def getGPTPartition2HostMap : collection.mutable.HashMap
    [Int, collection.mutable.ListBuffer[String]] = GPTPartition2HostMap

  def getGPTHost2PartitionMap : collection.mutable.HashMap
    [String, collection.mutable.ListBuffer[Int]] = GPTHost2PartitionMap

  def getDatanodeList: Array[String] = datanodeList
  def getDatanodeIdx(dnAddr: String) : Int = {
    var idx = 0
    for(i <- 0 to datanodeList.length-1) {
      if (datanodeList(i) == dnAddr) idx = i
    }
    idx
  }

  if (isGPTTask) {
    datanodeList = sc.hadoopConfiguration.get("GPT.Datanodes").split(",")
    isGPTTask =  false

    hadoopConf = sc.hadoopConfiguration
    numNodes = hadoopConf.get("GPT.NumNodes").toInt
    numMaxPartition = hadoopConf.get("GPT.MaxPartitions").toInt
    replicationFactor = hadoopConf.get("dfs.replication").toInt

    GPTTaskSets = new collection.mutable.HashMap[String, Boolean]()

    GPTHost2PartitionMap = new mutable.HashMap[String, mutable.ListBuffer[Int]]()
    datanodeList.foreach(d => GPTHost2PartitionMap(d) = new collection.mutable.ListBuffer[Int])

    var dnIdx = 0
    for (p <- 0 to numMaxPartition-1) {

      for (i <- 0 to replicationFactor - 1) {

        if (dnIdx == numNodes) dnIdx = 0
        val list = GPTHost2PartitionMap(datanodeList(dnIdx))
        list += p

        dnIdx += 1
      }
    }

    logDebug(s"Creating GPTHost2PartitionMap: " + GPTHost2PartitionMap.size + " nodes!")

    var nodeIdx = 0
    val GPTPartition2HostMap = new HashMap[Int, collection.mutable.ListBuffer[String]]
    for (p <- 0 to numMaxPartition-1) {

      for (r <- 0 to replicationFactor-1) {

        if (nodeIdx >= numNodes) nodeIdx = 0

        val dn = datanodeList(nodeIdx)

        if (GPTPartition2HostMap.contains(p)) {
          GPTPartition2HostMap(p) += dn
        } else {
          var nodeList = new collection.mutable.ListBuffer[String]
          nodeList += dn
          GPTPartition2HostMap(p) = nodeList
        }
        nodeIdx += 1
      }
    }
    logDebug(s"Creating GPTPartition2HostMap: " + GPTPartition2HostMap.size + " partitions!")
  }
  */

  private var schedulableBuilder: SchedulableBuilder = null
  // default scheduler is FIFO
  private val schedulingModeConf = conf.get(SCHEDULER_MODE_PROPERTY, SchedulingMode.FIFO.toString)
  val schedulingMode: SchedulingMode =
    try {
      SchedulingMode.withName(schedulingModeConf.toUpperCase(Locale.ROOT))
    } catch {
      case e: java.util.NoSuchElementException =>
        throw new SparkException(s"Unrecognized $SCHEDULER_MODE_PROPERTY: $schedulingModeConf")
    }

  val rootPool: Pool = new Pool("", schedulingMode, 0, 0)

  // This is a var so that we can reset it for testing purposes.
  private[spark] var taskResultGetter = new TaskResultGetter(sc.env, this)

  override def setDAGScheduler(dagScheduler: DAGScheduler) {
    this.dagScheduler = dagScheduler
  }

  def initialize(backend: SchedulerBackend) {
    this.backend = backend
    schedulableBuilder = {
      schedulingMode match {
        case SchedulingMode.FIFO =>
          new FIFOSchedulableBuilder(rootPool)
        case SchedulingMode.FAIR =>
          new FairSchedulableBuilder(rootPool, conf)
        case _ =>
          throw new IllegalArgumentException(s"Unsupported $SCHEDULER_MODE_PROPERTY: " +
          s"$schedulingMode")
      }
    }
    schedulableBuilder.buildPools()
  }

  def newTaskId(): Long = nextTaskId.getAndIncrement()
  def getCurTaskID(): Long = nextTaskId.get()

  override def start() {
    backend.start()

    if (!isLocal && conf.getBoolean("spark.speculation", false)) {
      logDebug("Starting speculative execution thread")
      speculationScheduler.scheduleWithFixedDelay(new Runnable {
        override def run(): Unit = Utils.tryOrStopSparkContext(sc) {
          checkSpeculatableTasks()
        }
      }, SPECULATION_INTERVAL_MS, SPECULATION_INTERVAL_MS, TimeUnit.MILLISECONDS)
    }
  }

  override def postStartHook() {
    waitBackendReady()
  }

  override def submitTasks(taskSet: TaskSet) {
    val tasks = taskSet.tasks
    val sanityVal = math.random

    /*
    logDebug("Adding task set " + taskSet.id + " with " + tasks.length + " tasks! tsm sanity val: [" + sanityVal + "]")

    val GPTTaskConfName = "GPTTask_" + tasks.length
    val newGPTConfName = GPTTaskConfName + "_" + taskSet.id

    val initialScanTask = sc.conf.getOption(GPTTaskConfName).isDefined
    if (initialScanTask) {
      var GPTConf = sc.conf.get(GPTTaskConfName)
      if (GPTConf != None) {
        isGPTTask = if (GPTConf == "true") true else false
      }

      if (isGPTTask) {
        GPTTaskSets += (GPTTaskConfName -> true)
        sc.conf.set(newGPTConfName, "true")
        logDebug(s"Need to schedule for GPT Taskset! SET " + GPTTaskConfName + " as TRUE!")
      } else {
        GPTTaskSets += (GPTTaskConfName -> false)
        logDebug(s"Need to schedule for NON-GPT Taskset! (taskName: " + GPTTaskConfName + ")")
      }
    } else {
      GPTTaskSets += (GPTTaskConfName -> false)
      logDebug(s"Need to schedule for NON-GPT Taskset!")
    }
    */

    this.synchronized {
      val manager = createTaskSetManager(taskSet, maxTaskFailures)
      /*
      manager.setSanityCheck(sanityVal)
      logDebug(s"CREATE TaskSetManager: " + manager.getSanityCheckValue() + "[# taskSet: " + taskSet.tasks.length + "]")
      */
      val stage = taskSet.stageId
      val stageTaskSets =
        taskSetsByStageIdAndAttempt.getOrElseUpdate(stage, new HashMap[Int, TaskSetManager])
      stageTaskSets(taskSet.stageAttemptId) = manager
      val conflictingTaskSet = stageTaskSets.exists { case (_, ts) =>
        ts.taskSet != taskSet && !ts.isZombie
      }
      if (conflictingTaskSet) {
        throw new IllegalStateException(s"more than one active taskSet for stage $stage:" +
          s" ${stageTaskSets.toSeq.map{_._2.taskSet.id}.mkString(",")}")
      }
      schedulableBuilder.addTaskSetManager(manager, manager.taskSet.properties)

      if (!isLocal && !hasReceivedTask) {
        starvationTimer.scheduleAtFixedRate(new TimerTask() {
          override def run() {
            if (!hasLaunchedTask) {
              logWarning("Initial job has not accepted any resources; " +
                "check your cluster UI to ensure that workers are registered " +
                "and have sufficient resources")
            } else {
              this.cancel()
            }
          }
        }, STARVATION_TIMEOUT_MS, STARVATION_TIMEOUT_MS)
      }
      hasReceivedTask = true
    }
    backend.reviveOffers()
  }

  // Label as private[scheduler] to allow tests to swap in different task set managers if necessary
  private[scheduler] def createTaskSetManager(
      taskSet: TaskSet,
      maxTaskFailures: Int): TaskSetManager = {
    val tsm = new TaskSetManager(this, taskSet, maxTaskFailures, blacklistTrackerOpt)
    /*
      val GPTTaskConfName = "GPTTask_" + taskSet.tasks.length
      isGPTTask = GPTTaskSets(GPTTaskConfName)
      isGPTTask = false;
      if (isGPTTask) {
      =
      logDebug(s"createTaskSetManager] GPTTaskConfName: " + GPTTaskConfName + ", isGPTTask: " + true)
      val numNodes = GPTHost2PartitionMap.size
      val replicationFactor = sc.hadoopConfiguration.get("dfs.replication").toInt
      val datanodeList = sc.hadoopConfiguration.get("GPT.Datanodes").split(",").sorted

      var GPTHost2PartitionStack = new scala.collection.mutable.HashMap[String, scala.collection.mutable.Stack[Int]]()
      var pivotVec = Seq(0,3,7,1,4,8,2,5,9,6)

      // init GPTHost2PartitionStack
      for (i <- 0 to numNodes - 1) {
        val d = datanodeList(i)
        var partitionStack = new scala.collection.mutable.Stack[Int]()
        partitionStack.push(pivotVec(i))
        logDebug(s"Init GPT Partition Stack | dn: " + d + ", pivotVal: " + pivotVec(i))
        GPTHost2PartitionStack(d) = partitionStack
      }

      var partitionString = ""
      var idx = 0
      GPTHost2PartitionStack.foreach { p =>
        partitionString += (p._1 + " -> " + p._2)
        partitionString += " | "
        idx += 1
      }
      logDebug(s"Initialized GPT Partition Stack")
      logDebug(s"\t" + partitionString)

      val round = (taskSet.tasks.length / numNodes) - 1

      for (c <- 1 to round) {
        var dnIdx = 0

        for (k <- 0 to numNodes -1) {
          val dn = datanodeList(k)
          GPTHost2PartitionStack(dn).push(pivotVec(k) + (c * numNodes))
          dnIdx += 1
        }
      }

      if (round > 1) {
        logDebug(s"Expanding GPT Partition Stack")
        partitionString = ""
        idx = 0
        GPTHost2PartitionStack.foreach { p =>
          partitionString += (p._1 + " -> " + p._2)
          partitionString += " | "
          idx += 1
        }
        logDebug(s"\t" + partitionString)
      }
      tsm.setGPTHost2PartitionStack(GPTHost2PartitionStack)

      tsm.setGPTTaskScheduling()

      } else {
      //logDebug(s"createTaskSetManager] GPTTaskConfName: " + GPTTaskConfName + ", isGPTTask: " + false)
      tsm.deSetGPTTaskScheduling()
      }
      */
    tsm
  }

  override def cancelTasks(stageId: Int, interruptThread: Boolean): Unit = synchronized {
    logInfo("Cancelling stage " + stageId)
    taskSetsByStageIdAndAttempt.get(stageId).foreach { attempts =>
      attempts.foreach { case (_, tsm) =>
        // There are two possible cases here:
        // 1. The task set manager has been created and some tasks have been scheduled.
        //    In this case, send a kill signal to the executors to kill the task and then abort
        //    the stage.
        // 2. The task set manager has been created but no tasks has been scheduled. In this case,
        //    simply abort the stage.
        tsm.runningTasksSet.foreach { tid =>
          val execId = taskIdToExecutorId(tid)
          backend.killTask(tid, execId, interruptThread, reason = "stage cancelled")
        }
        tsm.abort("Stage %s cancelled".format(stageId))
        logInfo("Stage %d was cancelled".format(stageId))
      }
    }
  }

  override def killTaskAttempt(taskId: Long, interruptThread: Boolean, reason: String): Boolean = {
    logInfo(s"Killing task $taskId: $reason")
    val execId = taskIdToExecutorId.get(taskId)
    if (execId.isDefined) {
      backend.killTask(taskId, execId.get, interruptThread, reason)
      true
    } else {
      logWarning(s"Could not kill task $taskId because no task with that ID was found.")
      false
    }
  }

  /**
   * Called to indicate that all task attempts (including speculated tasks) associated with the
   * given TaskSetManager have completed, so state associated with the TaskSetManager should be
   * cleaned up.
   */
  def taskSetFinished(manager: TaskSetManager): Unit = synchronized {
    taskSetsByStageIdAndAttempt.get(manager.taskSet.stageId).foreach { taskSetsForStage =>
      taskSetsForStage -= manager.taskSet.stageAttemptId
      if (taskSetsForStage.isEmpty) {
        taskSetsByStageIdAndAttempt -= manager.taskSet.stageId
      }
    }
    manager.parent.removeSchedulable(manager)
    logInfo(s"Removed TaskSet ${manager.taskSet.id}, whose tasks have all completed, from pool" +
      s" ${manager.parent.name}")
  }

  private def resourceOfferSingleTaskSet(
      taskSet: TaskSetManager,
      maxLocality: TaskLocality,
      shuffledOffers: Seq[WorkerOffer],
      availableCpus: Array[Int],
      tasks: IndexedSeq[ArrayBuffer[TaskDescription]]) : Boolean = {
    var launchedTask = false
    // nodes and executors that are blacklisted for the entire application have already been
    // filtered out by this point

    for (i <- 0 until shuffledOffers.size) {

      val execId = shuffledOffers(i).executorId
      val host = shuffledOffers(i).host

      if (availableCpus(i) >= CPUS_PER_TASK) {
        try {

          for (task <- taskSet.resourceOffer(execId, host, maxLocality)) {
            tasks(i) += task
            val tid = task.taskId
            taskIdToTaskSetManager(tid) = taskSet
            taskIdToExecutorId(tid) = execId
            executorIdToRunningTaskIds(execId).add(tid)
            availableCpus(i) -= CPUS_PER_TASK
            assert(availableCpus(i) >= 0)
            launchedTask = true
          }

          /*
          val numTasks = taskSet.numTasks
          val confName = "GPTTask_" + numTasks

          if (GPTTaskSets.get(confName).isDefined && GPTTaskSets(confName)) {
            logDebug(s"Calling taskSet.resourceOfferForGPT: confName: " + confName)
            for (task <- taskSet.resourceOfferForGPT(execId, host, maxLocality)) {
                tasks(i) += task
                val tid = task.taskId
                taskIdToTaskSetManager(tid) = taskSet
                taskIdToExecutorId(tid) = execId
                executorIdToRunningTaskIds(execId).add(tid)
                availableCpus(i) -= CPUS_PER_TASK
                assert(availableCpus(i) >= 0)
                launchedTask = true
              }
          } else {
            logDebug(s"Calling taskSet.resourceOffer: taskSet.name: " + taskSet.name)
          }
          */
        } catch {
          case e: TaskNotSerializableException =>
            logError(s"Resource offer failed, task set ${taskSet.name} was not serializable")
            // Do not offer resources for this task, but don't throw an error to allow other
            // task sets to be submitted.
            return launchedTask
        }
      }
    }
    return launchedTask
  }

  /**
   * Called by cluster manager to offer resources on slaves. We respond by asking our active task
   * sets for tasks in order of priority. We fill each node with tasks in a round-robin manner so
   * that tasks are balanced across the cluster.
   */
  def resourceOffers(offers: IndexedSeq[WorkerOffer]): Seq[Seq[TaskDescription]] = synchronized {
    // Mark each slave as alive and remember its hostname
    // Also track if new executor is added
    var newExecAvail = false
    for (o <- offers) {
      if (!hostToExecutors.contains(o.host)) {
        hostToExecutors(o.host) = new HashSet[String]()
      }
      if (!executorIdToRunningTaskIds.contains(o.executorId)) {
        hostToExecutors(o.host) += o.executorId
        executorAdded(o.executorId, o.host)
        executorIdToHost(o.executorId) = o.host
        executorIdToRunningTaskIds(o.executorId) = HashSet[Long]()
        newExecAvail = true
      }
      for (rack <- getRackForHost(o.host)) {
        hostsByRack.getOrElseUpdate(rack, new HashSet[String]()) += o.host
      }
    }

    // Before making any offers, remove any nodes from the blacklist whose blacklist has expired. Do
    // this here to avoid a separate thread and added synchronization overhead, and also because
    // updating the blacklist is only relevant when task offers are being made.
    blacklistTrackerOpt.foreach(_.applyBlacklistTimeout())

    val filteredOffers = blacklistTrackerOpt.map { blacklistTracker =>
      offers.filter { offer =>
        !blacklistTracker.isNodeBlacklisted(offer.host) &&
          !blacklistTracker.isExecutorBlacklisted(offer.executorId)
      }
    }.getOrElse(offers)

    val shuffledOffers = shuffleOffers(filteredOffers)
    // Build a list of tasks to assign to each worker.
    val tasks = shuffledOffers.map(o => new ArrayBuffer[TaskDescription](o.cores))
    val availableCpus = shuffledOffers.map(o => o.cores).toArray
    val sortedTaskSets = rootPool.getSortedTaskSetQueue
    for (taskSet <- sortedTaskSets) {
      logDebug("parentName: %s, name: %s, runningTasks: %s".format(
        taskSet.parent.name, taskSet.name, taskSet.runningTasks))
      if (newExecAvail) {
        taskSet.executorAdded()
      }
    }

    // Take each TaskSet in our scheduling order, and then offer it each node in increasing order
    // of locality levels so that it gets a chance to launch local tasks on all of them.
    // NOTE: the preferredLocality order: PROCESS_LOCAL, NODE_LOCAL, NO_PREF, RACK_LOCAL, ANY
    for (taskSet <- sortedTaskSets) {

      var launchedAnyTask = false
      var launchedTaskAtCurrentMaxLocality = false
      for (currentMaxLocality <- taskSet.myLocalityLevels) {
        do {
          launchedTaskAtCurrentMaxLocality = resourceOfferSingleTaskSet(
            taskSet, currentMaxLocality, shuffledOffers, availableCpus, tasks)
          launchedAnyTask |= launchedTaskAtCurrentMaxLocality
        } while (launchedTaskAtCurrentMaxLocality)
      }
      if (!launchedAnyTask) {
        taskSet.abortIfCompletelyBlacklisted(hostToExecutors)
      }
    }

    if (tasks.size > 0) {
      hasLaunchedTask = true
    }
    return tasks
  }

  /**
   * Shuffle offers around to avoid always placing tasks on the same workers.  Exposed to allow
   * overriding in tests, so it can be deterministic.
   */
  protected def shuffleOffers(offers: IndexedSeq[WorkerOffer]): IndexedSeq[WorkerOffer] = {
    Random.shuffle(offers)
  }

  def statusUpdate(tid: Long, state: TaskState, serializedData: ByteBuffer) {
    var failedExecutor: Option[String] = None
    var reason: Option[ExecutorLossReason] = None
    synchronized {
      try {
        taskIdToTaskSetManager.get(tid) match {
          case Some(taskSet) =>
            if (state == TaskState.LOST) {
              // TaskState.LOST is only used by the deprecated Mesos fine-grained scheduling mode,
              // where each executor corresponds to a single task, so mark the executor as failed.
              val execId = taskIdToExecutorId.getOrElse(tid, throw new IllegalStateException(
                "taskIdToTaskSetManager.contains(tid) <=> taskIdToExecutorId.contains(tid)"))
              if (executorIdToRunningTaskIds.contains(execId)) {
                reason = Some(
                  SlaveLost(s"Task $tid was lost, so marking the executor as lost as well."))
                removeExecutor(execId, reason.get)
                failedExecutor = Some(execId)
              }
            }
            if (TaskState.isFinished(state)) {
              cleanupTaskState(tid)
              taskSet.removeRunningTask(tid)
              if (state == TaskState.FINISHED) {
                taskResultGetter.enqueueSuccessfulTask(taskSet, tid, serializedData)
              } else if (Set(TaskState.FAILED, TaskState.KILLED, TaskState.LOST).contains(state)) {
                taskResultGetter.enqueueFailedTask(taskSet, tid, state, serializedData)
              }
            }
          case None =>
            logError(
              ("Ignoring update with state %s for TID %s because its task set is gone (this is " +
                "likely the result of receiving duplicate task finished status updates) or its " +
                "executor has been marked as failed.")
                .format(state, tid))
        }
      } catch {
        case e: Exception => logError("Exception in statusUpdate", e)
      }
    }
    // Update the DAGScheduler without holding a lock on this, since that can deadlock
    if (failedExecutor.isDefined) {
      assert(reason.isDefined)
      dagScheduler.executorLost(failedExecutor.get, reason.get)
      backend.reviveOffers()
    }
  }

  /**
   * Update metrics for in-progress tasks and let the master know that the BlockManager is still
   * alive. Return true if the driver knows about the given block manager. Otherwise, return false,
   * indicating that the block manager should re-register.
   */
  override def executorHeartbeatReceived(
      execId: String,
      accumUpdates: Array[(Long, Seq[AccumulatorV2[_, _]])],
      blockManagerId: BlockManagerId): Boolean = {
    // (taskId, stageId, stageAttemptId, accumUpdates)
    val accumUpdatesWithTaskIds: Array[(Long, Int, Int, Seq[AccumulableInfo])] = synchronized {
      accumUpdates.flatMap { case (id, updates) =>
        val accInfos = updates.map(acc => acc.toInfo(Some(acc.value), None))
        taskIdToTaskSetManager.get(id).map { taskSetMgr =>
          (id, taskSetMgr.stageId, taskSetMgr.taskSet.stageAttemptId, accInfos)
        }
      }
    }
    dagScheduler.executorHeartbeatReceived(execId, accumUpdatesWithTaskIds, blockManagerId)
  }

  def handleTaskGettingResult(taskSetManager: TaskSetManager, tid: Long): Unit = synchronized {
    taskSetManager.handleTaskGettingResult(tid)
  }

  def handleSuccessfulTask(
      taskSetManager: TaskSetManager,
      tid: Long,
      taskResult: DirectTaskResult[_]): Unit = synchronized {
    taskSetManager.handleSuccessfulTask(tid, taskResult)
  }

  def handleFailedTask(
      taskSetManager: TaskSetManager,
      tid: Long,
      taskState: TaskState,
      reason: TaskFailedReason): Unit = synchronized {
    taskSetManager.handleFailedTask(tid, taskState, reason)
    if (!taskSetManager.isZombie && !taskSetManager.someAttemptSucceeded(tid)) {
      // Need to revive offers again now that the task set manager state has been updated to
      // reflect failed tasks that need to be re-run.
      backend.reviveOffers()
    }
  }

  def error(message: String) {
    synchronized {
      if (taskSetsByStageIdAndAttempt.nonEmpty) {
        // Have each task set throw a SparkException with the error
        for {
          attempts <- taskSetsByStageIdAndAttempt.values
          manager <- attempts.values
        } {
          try {
            manager.abort(message)
          } catch {
            case e: Exception => logError("Exception in error callback", e)
          }
        }
      } else {
        // No task sets are active but we still got an error. Just exit since this
        // must mean the error is during registration.
        // It might be good to do something smarter here in the future.
        throw new SparkException(s"Exiting due to error from cluster scheduler: $message")
      }
    }
  }

  override def stop() {
    speculationScheduler.shutdown()
    if (backend != null) {
      backend.stop()
    }
    if (taskResultGetter != null) {
      taskResultGetter.stop()
    }
    starvationTimer.cancel()
  }

  override def defaultParallelism(): Int = backend.defaultParallelism()

  // Check for speculatable tasks in all our active jobs.
  def checkSpeculatableTasks() {
    var shouldRevive = false
    synchronized {
      shouldRevive = rootPool.checkSpeculatableTasks(MIN_TIME_TO_SPECULATION)
    }
    if (shouldRevive) {
      backend.reviveOffers()
    }
  }

  override def executorLost(executorId: String, reason: ExecutorLossReason): Unit = {
    var failedExecutor: Option[String] = None

    synchronized {
      if (executorIdToRunningTaskIds.contains(executorId)) {
        val hostPort = executorIdToHost(executorId)
        logExecutorLoss(executorId, hostPort, reason)
        removeExecutor(executorId, reason)
        failedExecutor = Some(executorId)
      } else {
        executorIdToHost.get(executorId) match {
          case Some(hostPort) =>
            // If the host mapping still exists, it means we don't know the loss reason for the
            // executor. So call removeExecutor() to update tasks running on that executor when
            // the real loss reason is finally known.
            logExecutorLoss(executorId, hostPort, reason)
            removeExecutor(executorId, reason)

          case None =>
            // We may get multiple executorLost() calls with different loss reasons. For example,
            // one may be triggered by a dropped connection from the slave while another may be a
            // report of executor termination from Mesos. We produce log messages for both so we
            // eventually report the termination reason.
            logError(s"Lost an executor $executorId (already removed): $reason")
        }
      }
    }
    // Call dagScheduler.executorLost without holding the lock on this to prevent deadlock
    if (failedExecutor.isDefined) {
      dagScheduler.executorLost(failedExecutor.get, reason)
      backend.reviveOffers()
    }
  }

  private def logExecutorLoss(
      executorId: String,
      hostPort: String,
      reason: ExecutorLossReason): Unit = reason match {
    case LossReasonPending =>
      logDebug(s"Executor $executorId on $hostPort lost, but reason not yet known.")
    case ExecutorKilled =>
      logInfo(s"Executor $executorId on $hostPort killed by driver.")
    case _ =>
      logError(s"Lost executor $executorId on $hostPort: $reason")
  }

  /**
   * Cleans up the TaskScheduler's state for tracking the given task.
   */
  private def cleanupTaskState(tid: Long): Unit = {
    taskIdToTaskSetManager.remove(tid)
    taskIdToExecutorId.remove(tid).foreach { executorId =>
      executorIdToRunningTaskIds.get(executorId).foreach { _.remove(tid) }
    }
  }

  /**
   * Remove an executor from all our data structures and mark it as lost. If the executor's loss
   * reason is not yet known, do not yet remove its association with its host nor update the status
   * of any running tasks, since the loss reason defines whether we'll fail those tasks.
   */
  private def removeExecutor(executorId: String, reason: ExecutorLossReason) {
    // The tasks on the lost executor may not send any more status updates (because the executor
    // has been lost), so they should be cleaned up here.
    executorIdToRunningTaskIds.remove(executorId).foreach { taskIds =>
      logDebug("Cleaning up TaskScheduler state for tasks " +
        s"${taskIds.mkString("[", ",", "]")} on failed executor $executorId")
      // We do not notify the TaskSetManager of the task failures because that will
      // happen below in the rootPool.executorLost() call.
      taskIds.foreach(cleanupTaskState)
    }

    val host = executorIdToHost(executorId)
    val execs = hostToExecutors.getOrElse(host, new HashSet)
    execs -= executorId
    if (execs.isEmpty) {
      hostToExecutors -= host
      for (rack <- getRackForHost(host); hosts <- hostsByRack.get(rack)) {
        hosts -= host
        if (hosts.isEmpty) {
          hostsByRack -= rack
        }
      }
    }

    if (reason != LossReasonPending) {
      executorIdToHost -= executorId
      rootPool.executorLost(executorId, host, reason)
    }
    blacklistTrackerOpt.foreach(_.handleRemovedExecutor(executorId))
  }

  def executorAdded(execId: String, host: String) {
    dagScheduler.executorAdded(execId, host)
  }

  def getExecutorsAliveOnHost(host: String): Option[Set[String]] = synchronized {
    hostToExecutors.get(host).map(_.toSet)
  }

  def hasExecutorsAliveOnHost(host: String): Boolean = synchronized {
    hostToExecutors.contains(host)
  }

  def hasHostAliveOnRack(rack: String): Boolean = synchronized {
    hostsByRack.contains(rack)
  }

  def isExecutorAlive(execId: String): Boolean = synchronized {
    executorIdToRunningTaskIds.contains(execId)
  }

  def isExecutorBusy(execId: String): Boolean = synchronized {
    executorIdToRunningTaskIds.get(execId).exists(_.nonEmpty)
  }

  /**
   * Get a snapshot of the currently blacklisted nodes for the entire application.  This is
   * thread-safe -- it can be called without a lock on the TaskScheduler.
   */
  def nodeBlacklist(): scala.collection.immutable.Set[String] = {
    blacklistTrackerOpt.map(_.nodeBlacklist()).getOrElse(scala.collection.immutable.Set())
  }

  // By default, rack is unknown
  def getRackForHost(value: String): Option[String] = None

  private def waitBackendReady(): Unit = {
    if (backend.isReady) {
      return
    }
    while (!backend.isReady) {
      // Might take a while for backend to be ready if it is waiting on resources.
      if (sc.stopped.get) {
        // For example: the master removes the application for some reason
        throw new IllegalStateException("Spark context stopped while waiting for backend")
      }
      synchronized {
        this.wait(100)
      }
    }
  }

  override def applicationId(): String = backend.applicationId()

  override def applicationAttemptId(): Option[String] = backend.applicationAttemptId()

  private[scheduler] def taskSetManagerForAttempt(
      stageId: Int,
      stageAttemptId: Int): Option[TaskSetManager] = {
    for {
      attempts <- taskSetsByStageIdAndAttempt.get(stageId)
      manager <- attempts.get(stageAttemptId)
    } yield {
      manager
    }
  }

}


private[spark] object TaskSchedulerImpl {

  val SCHEDULER_MODE_PROPERTY = "spark.scheduler.mode"

  /**
   * Used to balance containers across hosts.
   *
   * Accepts a map of hosts to resource offers for that host, and returns a prioritized list of
   * resource offers representing the order in which the offers should be used. The resource
   * offers are ordered such that we'll allocate one container on each host before allocating a
   * second container on any host, and so on, in order to reduce the damage if a host fails.
   *
   * For example, given {@literal <h1, [o1, o2, o3]>}, {@literal <h2, [o4]>} and
   * {@literal <h3, [o5, o6]>}, returns {@literal [o1, o5, o4, o2, o6, o3]}.
   */
  def prioritizeContainers[K, T] (map: HashMap[K, ArrayBuffer[T]]): List[T] = {
    val _keyList = new ArrayBuffer[K](map.size)
    _keyList ++= map.keys

    // order keyList based on population of value in map
    val keyList = _keyList.sortWith(
      (left, right) => map(left).size > map(right).size
    )

    val retval = new ArrayBuffer[T](keyList.size * 2)
    var index = 0
    var found = true

    while (found) {
      found = false
      for (key <- keyList) {
        val containerList: ArrayBuffer[T] = map.getOrElse(key, null)
        assert(containerList != null)
        // Get the index'th entry for this host - if present
        if (index < containerList.size) {
          retval += containerList.apply(index)
          found = true
        }
      }
      index += 1
    }

    retval.toList
  }

  private def maybeCreateBlacklistTracker(sc: SparkContext): Option[BlacklistTracker] = {
    if (BlacklistTracker.isBlacklistEnabled(sc.conf)) {
      val executorAllocClient: Option[ExecutorAllocationClient] = sc.schedulerBackend match {
        case b: ExecutorAllocationClient => Some(b)
        case _ => None
      }
      Some(new BlacklistTracker(sc, executorAllocClient))
    } else {
      None
    }
  }

}
