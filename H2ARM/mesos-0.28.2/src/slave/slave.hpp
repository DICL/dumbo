// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __SLAVE_HPP__
#define __SLAVE_HPP__

#include <stdint.h>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <boost/circular_buffer.hpp>

#include <mesos/attributes.hpp>
#include <mesos/resources.hpp>
#include <mesos/type_utils.hpp>

#include <mesos/executor/executor.hpp>

#include <mesos/module/authenticatee.hpp>

#include <mesos/slave/qos_controller.hpp>
#include <mesos/slave/resource_estimator.hpp>

#include <mesos/v1/executor/executor.hpp>

#include <process/http.hpp>
#include <process/future.hpp>
#include <process/owned.hpp>
#include <process/process.hpp>
#include <process/protobuf.hpp>

#include <stout/bytes.hpp>
#include <stout/linkedhashmap.hpp>
#include <stout/hashmap.hpp>
#include <stout/hashset.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/path.hpp>
#include <stout/recordio.hpp>
#include <stout/uuid.hpp>

#include "common/http.hpp"
#include "common/protobuf_utils.hpp"

#include "files/files.hpp"

#include "internal/evolve.hpp"

#include "master/detector.hpp"

#include "messages/messages.hpp"

#include "slave/constants.hpp"
#include "slave/containerizer/containerizer.hpp"
#include "slave/flags.hpp"
#include "slave/gc.hpp"
#include "slave/metrics.hpp"
#include "slave/monitor.hpp"
#include "slave/paths.hpp"
#include "slave/state.hpp"

// `REGISTERING` is used as an enum value, but it's actually defined as a
// constant in the Windows SDK.
#ifdef __WINDOWS__
#undef REGISTERING
#endif // __WINDOWS__

namespace mesos {
namespace internal {

class MasterDetector; // Forward declaration.

namespace slave {

using namespace process;

// Some forward declarations.
class StatusUpdateManager;
struct Executor;
struct Framework;
struct HttpConnection;

class Slave : public ProtobufProcess<Slave>
{
public:
  Slave(const std::string& id,
        const Flags& flags,
        MasterDetector* detector,
        Containerizer* containerizer,
        Files* files,
        GarbageCollector* gc,
        StatusUpdateManager* statusUpdateManager,
        mesos::slave::ResourceEstimator* resourceEstimator,
        mesos::slave::QoSController* qosController);

  virtual ~Slave();

  void shutdown(const process::UPID& from, const std::string& message);

  void registered(
      const process::UPID& from,
      const SlaveID& slaveId,
      const MasterSlaveConnection& connection);

  void reregistered(
      const process::UPID& from,
      const SlaveID& slaveId,
      const std::vector<ReconcileTasksMessage>& reconciliations,
      const MasterSlaveConnection& connection);

  void doReliableRegistration(Duration maxBackoff);

  // Made 'virtual' for Slave mocking.
  virtual void runTask(
      const process::UPID& from,
      const FrameworkInfo& frameworkInfo,
      const FrameworkID& frameworkId,
      const process::UPID& pid,
      TaskInfo task);

  // Made 'virtual' for Slave mocking.
  virtual void _runTask(
      const process::Future<bool>& future,
      const FrameworkInfo& frameworkInfo,
      const TaskInfo& task);

  process::Future<bool> unschedule(const std::string& path);

  // Made 'virtual' for Slave mocking.
  virtual void killTask(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const TaskID& taskId);

  void shutdownExecutor(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId);

  void shutdownFramework(
      const process::UPID& from,
      const FrameworkID& frameworkId);

  void schedulerMessage(
      const SlaveID& slaveId,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const std::string& data);

  void updateFramework(
      const FrameworkID& frameworkId,
      const process::UPID& pid);

  void checkpointResources(const std::vector<Resource>& checkpointedResources);

  void subscribe(
    HttpConnection http,
    const executor::Call::Subscribe& subscribe,
    Framework* framework,
    Executor* executor);

  void registerExecutor(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId);

  // Called when an executor re-registers with a recovering slave.
  // 'tasks' : Unacknowledged tasks (i.e., tasks that the executor
  //           driver never received an ACK for.)
  // 'updates' : Unacknowledged updates.
  void reregisterExecutor(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const std::vector<TaskInfo>& tasks,
      const std::vector<StatusUpdate>& updates);

  void _reregisterExecutor(
      const process::Future<Nothing>& future,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const ContainerID& containerId);

  void executorMessage(
      const SlaveID& slaveId,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const std::string& data);

  void ping(const process::UPID& from, bool connected);

  // Handles the status update.
  // NOTE: If 'pid' is a valid UPID an ACK is sent to this pid
  // after the update is successfully handled. If pid == UPID()
  // no ACK is sent. The latter is used by the slave to send
  // status updates it generated (e.g., TASK_LOST). If pid == None()
  // an ACK is sent to the corresponding HTTP based executor.
  // NOTE: StatusUpdate is passed by value because it is modified
  // to ensure source field is set.
  void statusUpdate(StatusUpdate update, const Option<process::UPID>& pid);

  // Called when the slave receives a `StatusUpdate` from an executor
  // and the slave needs to retrieve the container status for the
  // container associated with the executor.
  void _statusUpdate(
      StatusUpdate update,
      const Option<process::UPID>& pid,
      const ExecutorID& executorId,
      const Future<ContainerStatus>& containerStatus);

  // Continue handling the status update after optionally updating the
  // container's resources.
  void __statusUpdate(
      const Option<Future<Nothing>>& future,
      const StatusUpdate& update,
      const Option<process::UPID>& pid,
      const ExecutorID& executorId,
      const ContainerID& containerId,
      bool checkpoint);

  // This is called when the status update manager finishes
  // handling the update. If the handling is successful, an
  // acknowledgment is sent to the executor.
  void ___statusUpdate(
      const process::Future<Nothing>& future,
      const StatusUpdate& update,
      const Option<process::UPID>& pid);

  // This is called by status update manager to forward a status
  // update to the master. Note that the latest state of the task is
  // added to the update before forwarding.
  void forward(StatusUpdate update);

  void statusUpdateAcknowledgement(
      const process::UPID& from,
      const SlaveID& slaveId,
      const FrameworkID& frameworkId,
      const TaskID& taskId,
      const std::string& uuid);

  void _statusUpdateAcknowledgement(
      const process::Future<bool>& future,
      const TaskID& taskId,
      const FrameworkID& frameworkId,
      const UUID& uuid);

  void executorLaunched(
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const ContainerID& containerId,
      const process::Future<bool>& future);

  void executorTerminated(
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const process::Future<containerizer::Termination>& termination);

  // NOTE: Pulled these to public to make it visible for testing.
  // TODO(vinod): Make tests friends to this class instead.

  // Garbage collects the directories based on the current disk usage.
  // TODO(vinod): Instead of making this function public, we need to
  // mock both GarbageCollector (and pass it through slave's constructor)
  // and os calls.
  void _checkDiskUsage(const process::Future<double>& usage);

  // Invoked whenever the detector detects a change in masters.
  // Made public for testing purposes.
  void detected(const process::Future<Option<MasterInfo>>& pid);

  enum State
  {
    RECOVERING,   // Slave is doing recovery.
    DISCONNECTED, // Slave is not connected to the master.
    RUNNING,      // Slave has (re-)registered.
    TERMINATING,  // Slave is shutting down.
  } state;

  // TODO(benh): Clang requires members to be public in order to take
  // their address which we do in tests (for things like
  // FUTURE_DISPATCH).
// protected:
  virtual void initialize();
  virtual void finalize();
  virtual void exited(const process::UPID& pid);

  // This is called when the resource limits of the container have
  // been updated for the given tasks. If the update is successful, we
  // flush the given tasks to the executor by sending RunTaskMessages.
  // TODO(jieyu): Consider renaming it to '__runTasks' once the slave
  // starts to support launching multiple tasks in one call (i.e.,
  // multi-tasks version of 'runTask').
  void runTasks(
      const process::Future<Nothing>& future,
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const ContainerID& containerId,
      const std::list<TaskInfo>& tasks);

  void fileAttached(const process::Future<Nothing>& result,
                    const std::string& path);

  Nothing detachFile(const std::string& path);

  // Triggers a re-detection of the master when the slave does
  // not receive a ping.
  void pingTimeout(process::Future<Option<MasterInfo>> future);

  void authenticate();

  // Helper routines to lookup a framework/executor.
  Framework* getFramework(const FrameworkID& frameworkId);

  Executor* getExecutor(
      const FrameworkID& frameworkId,
      const ExecutorID& executorId);

  // Returns an ExecutorInfo for a TaskInfo (possibly
  // constructing one if the task has a CommandInfo).
  ExecutorInfo getExecutorInfo(
      const FrameworkInfo& frameworkInfo,
      const TaskInfo& task);

  // Shuts down the executor if it did not register yet.
  void registerExecutorTimeout(
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const ContainerID& containerId);

  // Cleans up all un-reregistered executors during recovery.
  void reregisterExecutorTimeout();

  // This function returns the max age of executor/slave directories allowed,
  // given a disk usage. This value could be used to tune gc.
  Duration age(double usage);

  // Checks the current disk usage and schedules for gc as necessary.
  void checkDiskUsage();

  // Recovers the slave, status update manager and isolator.
  process::Future<Nothing> recover(const Result<state::State>& state);

  // This is called after 'recover()'. If 'flags.reconnect' is
  // 'reconnect', the slave attempts to reconnect to any old live
  // executors. Otherwise, the slave attempts to shutdown/kill them.
  process::Future<Nothing> _recover();

  // This is a helper to call recover() on the containerizer at the end of
  // recover() and before __recover().
  // TODO(idownes): Remove this when we support defers to objects.
  process::Future<Nothing> _recoverContainerizer(
      const Option<state::SlaveState>& state);

  // This is called when recovery finishes.
  // Made 'virtual' for Slave mocking.
  virtual void __recover(const process::Future<Nothing>& future);

  // Helper to recover a framework from the specified state.
  void recoverFramework(const state::FrameworkState& state);

  // Removes and garbage collects the executor.
  void removeExecutor(Framework* framework, Executor* executor);

  // Removes and garbage collects the framework.
  // Made 'virtual' for Slave mocking.
  virtual void removeFramework(Framework* framework);

  // Schedules a 'path' for gc based on its modification time.
  Future<Nothing> garbageCollect(const std::string& path);

  // Called when the slave was signaled from the specified user.
  void signaled(int signal, int uid);

  // Made 'virtual' for Slave mocking.
  virtual void qosCorrections();

  // Made 'virtual' for Slave mocking.
  virtual void _qosCorrections(
      const process::Future<std::list<
          mesos::slave::QoSCorrection>>& correction);

  // Returns the resource usage information for all executors.
  process::Future<ResourceUsage> usage();

private:
  void _authenticate();
  void authenticationTimeout(process::Future<bool> future);

  // Shut down an executor. This is a two phase process. First, an
  // executor receives a shut down message (shut down phase), then
  // after a configurable timeout the slave actually forces a kill
  // (kill phase, via the isolator) if the executor has not
  // exited.
  void _shutdownExecutor(Framework* framework, Executor* executor);

  // Handle the second phase of shutting down an executor for those
  // executors that have not properly shutdown within a timeout.
  void shutdownExecutorTimeout(
      const FrameworkID& frameworkId,
      const ExecutorID& executorId,
      const ContainerID& containerId);

  // Inner class used to namespace HTTP route handlers (see
  // slave/http.cpp for implementations).
  class Http
  {
  public:
    explicit Http(Slave* _slave) : slave(_slave) {}

    // Logs the request, route handlers can compose this with the
    // desired request handler to get consistent request logging.
    static void log(const process::http::Request& request);

    // /slave/api/v1/executor
    process::Future<process::http::Response> executor(
        const process::http::Request& request) const;

    // /slave/flags
    process::Future<process::http::Response> flags(
        const process::http::Request& request) const;

    // /slave/health
    process::Future<process::http::Response> health(
        const process::http::Request& request) const;

    // /slave/state
    process::Future<process::http::Response> state(
        const process::http::Request& request) const;

    static std::string EXECUTOR_HELP();
    static std::string FLAGS_HELP();
    static std::string HEALTH_HELP();
    static std::string STATE_HELP();

  private:
    Slave* slave;
  };

  friend struct Framework;
  friend struct Executor;
  friend struct Metrics;

  Slave(const Slave&);              // No copying.
  Slave& operator=(const Slave&); // No assigning.

  // Gauge methods.
  double _frameworks_active()
  {
    return frameworks.size();
  }

  double _uptime_secs()
  {
    return (Clock::now() - startTime).secs();
  }

  double _registered()
  {
    return master.isSome() ? 1 : 0;
  }

  double _tasks_staging();
  double _tasks_starting();
  double _tasks_running();
  double _tasks_killing();

  double _executors_registering();
  double _executors_running();
  double _executors_terminating();

  double _executor_directory_max_allowed_age_secs();

  void sendExecutorTerminatedStatusUpdate(
      const TaskID& taskId,
      const Future<containerizer::Termination>& termination,
      const FrameworkID& frameworkId,
      const Executor* executor);

  // Forwards the current total of oversubscribed resources.
  void forwardOversubscribed();
  void _forwardOversubscribed(
      const process::Future<Resources>& oversubscribable);

  const Flags flags;

  SlaveInfo info;

  // Resources that are checkpointed by the slave.
  Resources checkpointedResources;

  Option<process::UPID> master;

  hashmap<FrameworkID, Framework*> frameworks;

  boost::circular_buffer<process::Owned<Framework>> completedFrameworks;

  MasterDetector* detector;

  Containerizer* containerizer;

  Files* files;

  Metrics metrics;

  double _resources_total(const std::string& name);
  double _resources_used(const std::string& name);
  double _resources_percent(const std::string& name);

  double _resources_revocable_total(const std::string& name);
  double _resources_revocable_used(const std::string& name);
  double _resources_revocable_percent(const std::string& name);

  process::Time startTime;

  GarbageCollector* gc;

  ResourceMonitor monitor;

  StatusUpdateManager* statusUpdateManager;

  // Master detection future.
  process::Future<Option<MasterInfo>> detection;

  // Master's ping timeout value, updated on reregistration.
  Duration masterPingTimeout;

  // Timer for triggering re-detection when no ping is received from
  // the master.
  process::Timer pingTimer;

  // Flag to indicate if recovery, including reconciling (i.e., reconnect/kill)
  // with executors is finished.
  process::Promise<Nothing> recovered;

  // Root meta directory containing checkpointed data.
  const std::string metaDir;

  // Indicates the number of errors ignored in "--no-strict" recovery mode.
  unsigned int recoveryErrors;

  Option<Credential> credential;

  // Authenticatee name as supplied via flags.
  std::string authenticateeName;

  Authenticatee* authenticatee;

  // Indicates if an authentication attempt is in progress.
  Option<Future<bool>> authenticating;

  // Indicates if the authentication is successful.
  bool authenticated;

  // Indicates if a new authentication attempt should be enforced.
  bool reauthenticate;

  // Maximum age of executor directories. Will be recomputed
  // periodically every flags.disk_watch_interval.
  Duration executorDirectoryMaxAllowedAge;

  mesos::slave::ResourceEstimator* resourceEstimator;

  mesos::slave::QoSController* qosController;

  // The most recent estimate of the total amount of oversubscribed
  // (allocated and oversubscribable) resources.
  Option<Resources> oversubscribedResources;
};


// Represents the streaming HTTP connection to an executor.
struct HttpConnection
{
  HttpConnection(const process::http::Pipe::Writer& _writer,
                 ContentType _contentType)
    : writer(_writer),
      contentType(_contentType),
      encoder(lambda::bind(serialize, contentType, lambda::_1)) {}

  // Converts the message to an Event before sending.
  template <typename Message>
  bool send(const Message& message)
  {
    // We need to evolve the internal 'message' into a
    // 'v1::executor::Event'.
    return writer.write(encoder.encode(evolve(message)));
  }

  bool close()
  {
    return writer.close();
  }

  process::Future<Nothing> closed() const
  {
    return writer.readerClosed();
  }

  process::http::Pipe::Writer writer;
  ContentType contentType;
  ::recordio::Encoder<v1::executor::Event> encoder;
};


std::ostream& operator<<(std::ostream& stream, const Executor& executor);


// Information describing an executor.
struct Executor
{
  Executor(
      Slave* slave,
      const FrameworkID& frameworkId,
      const ExecutorInfo& info,
      const ContainerID& containerId,
      const std::string& directory,
      bool checkpoint);

  ~Executor();

  Task* addTask(const TaskInfo& task);
  void terminateTask(const TaskID& taskId, const mesos::TaskStatus& status);
  void completeTask(const TaskID& taskId);
  void checkpointExecutor();
  void checkpointTask(const TaskInfo& task);
  void recoverTask(const state::TaskState& state);
  void updateTaskState(const TaskStatus& status);

  // Returns true if there are any queued/launched/terminated tasks.
  bool incompleteTasks();

  // Sends a message to the connected executor.
  template <typename Message>
  void send(const Message& message)
  {
    if (state == REGISTERING || state == TERMINATED) {
      LOG(WARNING) << "Attempting to send message to disconnected"
                   << " executor " << *this << " in state " << state;
    }

    if (http.isSome()) {
      if (!http->send(message)) {
        LOG(WARNING) << "Unable to send event to executor " << *this
                     << ": connection closed";
      }
    } else if (pid.isSome()) {
      slave->send(pid.get(), message);
    } else {
      LOG(WARNING) << "Unable to send event to executor " << *this
                   << ": unknown connection type";
    }
  }

  // Returns true if this is a command executor.
  bool isCommandExecutor() const;

  // Closes the HTTP connection.
  void closeHttpConnection();

  friend std::ostream& operator<<(
      std::ostream& stream,
      const Executor& executor);

  enum State
  {
    REGISTERING,  // Executor is launched but not (re-)registered yet.
    RUNNING,      // Executor has (re-)registered.
    TERMINATING,  // Executor is being shutdown/killed.
    TERMINATED,   // Executor has terminated but there might be pending updates.
  } state;

  // We store the pointer to 'Slave' to get access to its methods
  // variables. One could imagine 'Executor' as being an inner class
  // of the 'Slave' class.
  Slave* slave;

  const ExecutorID id;
  const ExecutorInfo info;

  const FrameworkID frameworkId;

  const ContainerID containerId;

  const std::string directory;

  const bool checkpoint;

  // An Executor can either be connected via HTTP or by libprocess
  // message passing. The following are the possible states:
  //
  // Agent State    Executor State       http       pid    Executor Type
  // -----------    --------------       ----      ----    -------------
  //  RECOVERING       REGISTERING       None     UPID()         Unknown
  //                   REGISTERING       None       Some      Libprocess
  //                   REGISTERING       None       None            HTTP
  //
  //           *       REGISTERING       None       None   Not known yet
  //           *                 *       None       Some      Libprocess
  //           *                 *       Some       None            HTTP
  Option<HttpConnection> http;
  Option<process::UPID> pid;

  // Currently consumed resources.
  Resources resources;

  // Tasks can be found in one of the following four data structures:

  // Not yet launched.
  LinkedHashMap<TaskID, TaskInfo> queuedTasks;

  // Running.
  LinkedHashMap<TaskID, Task*> launchedTasks;

  // Terminated but pending updates.
  LinkedHashMap<TaskID, Task*> terminatedTasks;

  // Terminated and updates acked.
  // NOTE: We use a shared pointer for Task because clang doesn't like
  // Boost's implementation of circular_buffer with Task (Boost
  // attempts to do some memset's which are unsafe).
  boost::circular_buffer<std::shared_ptr<Task>> completedTasks;

  // When the slave initiates a destroy of the container, we expect a
  // termination to occur. The 'pendingTermation' indicates why the
  // slave initiated the destruction and will influence the
  // information sent in the status updates for any remaining
  // non-terminal tasks.
  Option<containerizer::Termination> pendingTermination;

private:
  Executor(const Executor&);              // No copying.
  Executor& operator=(const Executor&); // No assigning.

  bool commandExecutor;
};


// Information about a framework.
struct Framework
{
  Framework(
      Slave* slave,
      const FrameworkInfo& info,
      const Option<process::UPID>& pid);

  ~Framework();

  Executor* launchExecutor(
      const ExecutorInfo& executorInfo,
      const TaskInfo& taskInfo);
  void destroyExecutor(const ExecutorID& executorId);
  Executor* getExecutor(const ExecutorID& executorId);
  Executor* getExecutor(const TaskID& taskId);
  void recoverExecutor(const state::ExecutorState& state);
  void checkpointFramework() const;

  const FrameworkID id() const { return info.id(); }

  enum State
  {
    RUNNING,      // First state of a newly created framework.
    TERMINATING,  // Framework is shutting down in the cluster.
  } state;

  // We store the pointer to 'Slave' to get access to its methods
  // variables. One could imagine 'Framework' as being an inner class
  // of the 'Slave' class.
  Slave* slave;

  const FrameworkInfo info;

  // Frameworks using the scheduler driver will have a 'pid',
  // which allows us to send executor messages directly to the
  // driver. Frameworks using the HTTP API (in 0.24.0) will
  // not have a 'pid', in which case executor messages are
  // sent through the master.
  Option<UPID> pid;

  // Executors with pending tasks.
  hashmap<ExecutorID, hashmap<TaskID, TaskInfo>> pending;

  // Current running executors.
  hashmap<ExecutorID, Executor*> executors;

  // Up to MAX_COMPLETED_EXECUTORS_PER_FRAMEWORK completed executors.
  boost::circular_buffer<process::Owned<Executor>> completedExecutors;
private:
  Framework(const Framework&);              // No copying.
  Framework& operator=(const Framework&); // No assigning.
};


std::ostream& operator<<(std::ostream& stream, Slave::State state);
std::ostream& operator<<(std::ostream& stream, Framework::State state);
std::ostream& operator<<(std::ostream& stream, Executor::State state);

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __SLAVE_HPP__
