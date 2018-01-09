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

#include <mesos/slave/isolator.hpp>

#include <mesos/type_utils.hpp>

#include <process/clock.hpp>
#include <process/pid.hpp>

#include <stout/adaptor.hpp>
#include <stout/foreach.hpp>
#include <stout/net.hpp>
#include <stout/stringify.hpp>
#include <stout/uuid.hpp>

#include "common/protobuf_utils.hpp"

#include "messages/messages.hpp"

using std::string;

using google::protobuf::RepeatedPtrField;

using mesos::slave::ContainerLimitation;
using mesos::slave::ContainerState;

using process::UPID;

namespace mesos {
namespace internal {
namespace protobuf {

bool isTerminalState(const TaskState& state)
{
  return (state == TASK_FINISHED ||
          state == TASK_FAILED ||
          state == TASK_KILLED ||
          state == TASK_LOST ||
          state == TASK_ERROR);
}


StatusUpdate createStatusUpdate(
    const FrameworkID& frameworkId,
    const Option<SlaveID>& slaveId,
    const TaskID& taskId,
    const TaskState& state,
    const TaskStatus::Source& source,
    const Option<UUID>& uuid,
    const string& message,
    const Option<TaskStatus::Reason>& reason,
    const Option<ExecutorID>& executorId,
    const Option<bool>& healthy,
    const Option<Labels>& labels,
    const Option<ContainerStatus>& containerStatus)
{
  StatusUpdate update;

  update.set_timestamp(process::Clock::now().secs());
  update.mutable_framework_id()->MergeFrom(frameworkId);

  if (slaveId.isSome()) {
    update.mutable_slave_id()->MergeFrom(slaveId.get());
  }

  if (executorId.isSome()) {
    update.mutable_executor_id()->MergeFrom(executorId.get());
  }

  TaskStatus* status = update.mutable_status();
  status->mutable_task_id()->MergeFrom(taskId);

  if (slaveId.isSome()) {
    status->mutable_slave_id()->MergeFrom(slaveId.get());
  }

  status->set_state(state);
  status->set_source(source);
  status->set_message(message);
  status->set_timestamp(update.timestamp());

  if (uuid.isSome()) {
    update.set_uuid(uuid.get().toBytes());
    status->set_uuid(uuid.get().toBytes());
  }

  if (reason.isSome()) {
    status->set_reason(reason.get());
  }

  if (healthy.isSome()) {
    status->set_healthy(healthy.get());
  }

  if (labels.isSome()) {
    status->mutable_labels()->CopyFrom(labels.get());
  }

  if (containerStatus.isSome()) {
    status->mutable_container_status()->CopyFrom(containerStatus.get());
  }

  return update;
}


StatusUpdate createStatusUpdate(
    const FrameworkID& frameworkId,
    const TaskStatus& status,
    const Option<SlaveID>& slaveId)
{
  StatusUpdate update;

  update.mutable_framework_id()->MergeFrom(frameworkId);

  if (status.has_executor_id()) {
    update.mutable_executor_id()->MergeFrom(status.executor_id());
  }

  if (slaveId.isSome()) {
    update.mutable_slave_id()->MergeFrom(slaveId.get());
  }

  update.mutable_status()->MergeFrom(status);

  if (!status.has_timestamp()) {
    update.set_timestamp(process::Clock::now().secs());
  } else {
    update.set_timestamp(status.timestamp());
  }

  if (status.has_uuid()) {
    update.set_uuid(status.uuid());
  }

  return update;
}


Task createTask(
    const TaskInfo& task,
    const TaskState& state,
    const FrameworkID& frameworkId)
{
  Task t;
  t.mutable_framework_id()->CopyFrom(frameworkId);
  t.set_state(state);
  t.set_name(task.name());
  t.mutable_task_id()->CopyFrom(task.task_id());
  t.mutable_slave_id()->CopyFrom(task.slave_id());
  t.mutable_resources()->CopyFrom(task.resources());

  if (task.has_executor()) {
    t.mutable_executor_id()->CopyFrom(task.executor().executor_id());
  }

  if (task.has_labels()) {
    t.mutable_labels()->CopyFrom(task.labels());
  }

  if (task.has_discovery()) {
    t.mutable_discovery()->CopyFrom(task.discovery());
  }

  if (task.has_container()) {
    t.mutable_container()->CopyFrom(task.container());
  }

  return t;
}


Option<bool> getTaskHealth(const Task& task)
{
  Option<bool> healthy = None();
  if (task.statuses_size() > 0) {
    // The statuses list only keeps the most recent TaskStatus for
    // each state, and appends later states at the end. Thus the last
    // status is either a terminal state (where health is
    // irrelevant), or the latest RUNNING status.
    TaskStatus lastStatus = task.statuses(task.statuses_size() - 1);
    if (lastStatus.has_healthy()) {
      healthy = lastStatus.healthy();
    }
  }
  return healthy;
}


Option<ContainerStatus> getTaskContainerStatus(const Task& task)
{
  // The statuses list only keeps the most recent TaskStatus for
  // each state, and appends later states at the end. Let's find
  // the most recent TaskStatus with a valid container_status.
  foreach (const TaskStatus& status, adaptor::reverse(task.statuses())) {
    if (status.has_container_status()) {
      return status.container_status();
    }
  }
  return None();
}


/**
 * Creates a MasterInfo protobuf from the process's UPID.
 *
 * This is only used by the `StandaloneMasterDetector` (used in tests
 * and outside tests when ZK is not used).
 *
 * For example, when we start a slave with
 * `--master=master@127.0.0.1:5050`, since the slave (and consequently
 * its detector) doesn't have enough information about `MasterInfo`, it
 * tries to construct it based on the only available information
 * (`UPID`).
 *
 * @param pid The process's assigned untyped PID.
 * @return A fully formed `MasterInfo` with the IP/hostname information
 *    as derived from the `UPID`.
 */
MasterInfo createMasterInfo(const UPID& pid)
{
  MasterInfo info;
  info.set_id(stringify(pid) + "-" + UUID::random().toString());

  // NOTE: Currently, we store the ip in network order, which should
  // be fixed. See MESOS-1201 for more details.
  // TODO(marco): `ip` and `port` are deprecated in favor of `address`;
  //     remove them both after the deprecation cycle.
  info.set_ip(pid.address.ip.in().get().s_addr);
  info.set_port(pid.address.port);

  info.mutable_address()->set_ip(stringify(pid.address.ip));
  info.mutable_address()->set_port(pid.address.port);

  info.set_pid(pid);

  Try<string> hostname = net::getHostname(pid.address.ip);
  if (hostname.isSome()) {
    // Hostname is deprecated; but we need to update it
    // to maintain backward compatibility.
    // TODO(marco): Remove once we deprecate it.
    info.set_hostname(hostname.get());
    info.mutable_address()->set_hostname(hostname.get());
  }

  return info;
}


Label createLabel(const string& key, const Option<string>& value)
{
  Label label;
  label.set_key(key);
  if (value.isSome()) {
    label.set_value(value.get());
  }
  return label;
}


TimeInfo getCurrentTime()
{
  TimeInfo timeInfo;
  timeInfo.set_nanoseconds(process::Clock::now().duration().ns());
  return timeInfo;
}

namespace slave {

ContainerLimitation createContainerLimitation(
    const Resources& resources,
    const string& message,
    const TaskStatus::Reason& reason)
{
  ContainerLimitation limitation;
  foreach (Resource resource, resources) {
    limitation.add_resources()->CopyFrom(resource);
  }
  limitation.set_message(message);
  limitation.set_reason(reason);
  return limitation;
}


ContainerState createContainerState(
    const ExecutorInfo& executorInfo,
    const ContainerID& container_id,
    pid_t pid,
    const string& directory)
{
  ContainerState state;
  state.mutable_executor_info()->CopyFrom(executorInfo);
  state.mutable_container_id()->CopyFrom(container_id);
  state.set_pid(pid);
  state.set_directory(directory);
  return state;
}

} // namespace slave {

namespace maintenance {

Unavailability createUnavailability(
    const process::Time& start,
    const Option<Duration>& duration)
{
  Unavailability unavailability;
  unavailability.mutable_start()->set_nanoseconds(start.duration().ns());

  if (duration.isSome()) {
    unavailability.mutable_duration()->set_nanoseconds(duration.get().ns());
  }

  return unavailability;
}


RepeatedPtrField<MachineID> createMachineList(
    std::initializer_list<MachineID> ids)
{
  RepeatedPtrField<MachineID> array;

  foreach (const MachineID& id, ids) {
    array.Add()->CopyFrom(id);
  }

  return array;
}


mesos::maintenance::Window createWindow(
    std::initializer_list<MachineID> ids,
    const Unavailability& unavailability)
{
  mesos::maintenance::Window window;
  window.mutable_unavailability()->CopyFrom(unavailability);

  foreach (const MachineID& id, ids) {
    window.add_machine_ids()->CopyFrom(id);
  }

  return window;
}


mesos::maintenance::Schedule createSchedule(
    std::initializer_list<mesos::maintenance::Window> windows)
{
  mesos::maintenance::Schedule schedule;

  foreach (const mesos::maintenance::Window& window, windows) {
    schedule.add_windows()->CopyFrom(window);
  }

  return schedule;
}

} // namespace maintenance {

} // namespace protobuf {
} // namespace internal {
} // namespace mesos {
