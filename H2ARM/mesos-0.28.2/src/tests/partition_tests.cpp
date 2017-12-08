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

#include <vector>

#include <gmock/gmock.h>

#include <process/clock.hpp>
#include <process/future.hpp>
#include <process/gmock.hpp>
#include <process/pid.hpp>

#include <stout/try.hpp>
#include <stout/uuid.hpp>

#include "common/protobuf_utils.hpp"

#include "master/master.hpp"

#include "master/allocator/mesos/allocator.hpp"

#include "slave/constants.hpp"
#include "slave/flags.hpp"
#include "slave/slave.hpp"

#include "tests/containerizer.hpp"
#include "tests/flags.hpp"
#include "tests/mesos.hpp"

using mesos::internal::master::Master;

using mesos::internal::master::allocator::MesosAllocatorProcess;

using mesos::internal::slave::Slave;

using process::Clock;
using process::Future;
using process::Message;
using process::PID;

using std::vector;

using testing::_;
using testing::AtMost;
using testing::Eq;
using testing::Return;

namespace mesos {
namespace internal {
namespace tests {


class PartitionTest : public MesosTest {};


// This test checks that a scheduler gets a slave lost
// message for a partitioned slave.
TEST_F(PartitionTest, PartitionedSlave)
{
  master::Flags masterFlags = CreateMasterFlags();
  Try<PID<Master>> master = StartMaster(masterFlags);
  ASSERT_SOME(master);

  // Set these expectations up before we spawn the slave so that we
  // don't miss the first PING.
  Future<Message> ping = FUTURE_MESSAGE(
      Eq(PingSlaveMessage().GetTypeName()), _, _);

  // Drop all the PONGs to simulate slave partition.
  DROP_PROTOBUFS(PongSlaveMessage(), _, _);

  Try<PID<Slave>> slave = StartSlave();
  ASSERT_SOME(slave);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  EXPECT_CALL(sched, registered(&driver, _, _));

  Future<Nothing> resourceOffers;
  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillOnce(FutureSatisfy(&resourceOffers))
    .WillRepeatedly(Return()); // Ignore subsequent offers.

  driver.start();

  // Need to make sure the framework AND slave have registered with
  // master. Waiting for resource offers should accomplish both.
  AWAIT_READY(resourceOffers);

  Clock::pause();

  EXPECT_CALL(sched, offerRescinded(&driver, _))
    .Times(AtMost(1));

  Future<Nothing> slaveLost;
  EXPECT_CALL(sched, slaveLost(&driver, _))
    .WillOnce(FutureSatisfy(&slaveLost));

  // Now advance through the PINGs.
  size_t pings = 0;
  while (true) {
    AWAIT_READY(ping);
    pings++;
    if (pings == masterFlags.max_slave_ping_timeouts) {
     break;
    }
    ping = FUTURE_MESSAGE(Eq(PingSlaveMessage().GetTypeName()), _, _);
    Clock::advance(masterFlags.slave_ping_timeout);
  }

  Clock::advance(masterFlags.slave_ping_timeout);

  AWAIT_READY(slaveLost);

  this->Stop(slave.get());

  JSON::Object stats = Metrics();
  EXPECT_EQ(1, stats.values["master/slave_removals"]);
  EXPECT_EQ(1, stats.values["master/slave_removals/reason_unhealthy"]);

  driver.stop();
  driver.join();

  Shutdown();

  Clock::resume();
}


// The purpose of this test is to ensure that when slaves are removed
// from the master, and then attempt to re-register, we deny the
// re-registration by sending a ShutdownMessage to the slave.
// Why? Because during a network partition, the master will remove a
// partitioned slave, thus sending its tasks to LOST. At this point,
// when the partition is removed, the slave will attempt to
// re-register with its running tasks. We've already notified
// frameworks that these tasks were LOST, so we have to have the slave
// slave shut down.
TEST_F(PartitionTest, PartitionedSlaveReregistration)
{
  master::Flags masterFlags = CreateMasterFlags();
  Try<PID<Master>> master = StartMaster(masterFlags);
  ASSERT_SOME(master);

  // Allow the master to PING the slave, but drop all PONG messages
  // from the slave. Note that we don't match on the master / slave
  // PIDs because it's actually the SlaveObserver Process that sends
  // the pings.
  Future<Message> ping = FUTURE_MESSAGE(
      Eq(PingSlaveMessage().GetTypeName()), _, _);

  DROP_PROTOBUFS(PongSlaveMessage(), _, _);

  MockExecutor exec(DEFAULT_EXECUTOR_ID);

  StandaloneMasterDetector detector(master.get());

  Try<PID<Slave>> slave = StartSlave(&exec, &detector);
  ASSERT_SOME(slave);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  EXPECT_CALL(sched, registered(&driver, _, _));

  Future<vector<Offer>> offers;
  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return());

  driver.start();

  AWAIT_READY(offers);
  ASSERT_NE(0u, offers.get().size());

  // Launch a task. This is to ensure the task is killed by the slave,
  // during shutdown.
  TaskID taskId;
  taskId.set_value("1");

  TaskInfo task;
  task.set_name("");
  task.mutable_task_id()->MergeFrom(taskId);
  task.mutable_slave_id()->MergeFrom(offers.get()[0].slave_id());
  task.mutable_resources()->MergeFrom(offers.get()[0].resources());
  task.mutable_executor()->MergeFrom(DEFAULT_EXECUTOR_INFO);
  task.mutable_executor()->mutable_command()->set_value("sleep 60");

  // Set up the expectations for launching the task.
  EXPECT_CALL(exec, registered(_, _, _, _));
  EXPECT_CALL(exec, launchTask(_, _))
    .WillOnce(SendStatusUpdateFromTask(TASK_RUNNING));

  Future<TaskStatus> runningStatus;
  EXPECT_CALL(sched, statusUpdate(&driver, _))
    .WillOnce(FutureArg<1>(&runningStatus));

  Future<Nothing> statusUpdateAck = FUTURE_DISPATCH(
      slave.get(), &Slave::_statusUpdateAcknowledgement);

  driver.launchTasks(offers.get()[0].id(), {task});

  AWAIT_READY(runningStatus);
  EXPECT_EQ(TASK_RUNNING, runningStatus.get().state());

  // Wait for the slave to have handled the acknowledgment prior
  // to pausing the clock.
  AWAIT_READY(statusUpdateAck);

  // Drop the first shutdown message from the master (simulated
  // partition), allow the second shutdown message to pass when
  // the slave re-registers.
  Future<ShutdownMessage> shutdownMessage =
    DROP_PROTOBUF(ShutdownMessage(), _, slave.get());

  Future<TaskStatus> lostStatus;
  EXPECT_CALL(sched, statusUpdate(&driver, _))
    .WillOnce(FutureArg<1>(&lostStatus));

  Future<Nothing> slaveLost;
  EXPECT_CALL(sched, slaveLost(&driver, _))
    .WillOnce(FutureSatisfy(&slaveLost));

  Clock::pause();

  // Now, induce a partition of the slave by having the master
  // timeout the slave.
  size_t pings = 0;
  while (true) {
    AWAIT_READY(ping);
    pings++;
    if (pings == masterFlags.max_slave_ping_timeouts) {
     break;
    }
    ping = FUTURE_MESSAGE(Eq(PingSlaveMessage().GetTypeName()), _, _);
    Clock::advance(masterFlags.slave_ping_timeout);
    Clock::settle();
  }

  Clock::advance(masterFlags.slave_ping_timeout);
  Clock::settle();

  // The master will have notified the framework of the lost task.
  AWAIT_READY(lostStatus);
  EXPECT_EQ(TASK_LOST, lostStatus.get().state());

  // Wait for the master to attempt to shut down the slave.
  AWAIT_READY(shutdownMessage);

  // The master will notify the framework that the slave was lost.
  AWAIT_READY(slaveLost);

  Clock::resume();

  // We now complete the partition on the slave side as well. This
  // is done by simulating a master loss event which would normally
  // occur during a network partition.
  detector.appoint(None());

  Future<Nothing> shutdown;
  EXPECT_CALL(exec, shutdown(_))
    .WillOnce(FutureSatisfy(&shutdown));

  shutdownMessage = FUTURE_PROTOBUF(ShutdownMessage(), _, slave.get());

  // Have the slave re-register with the master.
  detector.appoint(master.get());

  // Upon re-registration, the master will shutdown the slave.
  // The slave will then shut down the executor.
  AWAIT_READY(shutdownMessage);
  AWAIT_READY(shutdown);

  driver.stop();
  driver.join();

  Shutdown();
}


// The purpose of this test is to ensure that when slaves are removed
// from the master, and then attempt to send status updates, we send
// a ShutdownMessage to the slave. Why? Because during a network
// partition, the master will remove a partitioned slave, thus sending
// its tasks to LOST. At this point, when the partition is removed,
// the slave may attempt to send updates if it was unaware that the
// master removed it. We've already notified frameworks that these
// tasks were LOST, so we have to have the slave shut down.
TEST_F(PartitionTest, PartitionedSlaveStatusUpdates)
{
  master::Flags masterFlags = CreateMasterFlags();
  Try<PID<Master>> master = StartMaster(masterFlags);
  ASSERT_SOME(master);

  // Allow the master to PING the slave, but drop all PONG messages
  // from the slave. Note that we don't match on the master / slave
  // PIDs because it's actually the SlaveObserver Process that sends
  // the pings.
  Future<Message> ping = FUTURE_MESSAGE(
      Eq(PingSlaveMessage().GetTypeName()), _, _);

  DROP_PROTOBUFS(PongSlaveMessage(), _, _);

  Future<SlaveRegisteredMessage> slaveRegisteredMessage =
    FUTURE_PROTOBUF(SlaveRegisteredMessage(), _, _);

  MockExecutor exec(DEFAULT_EXECUTOR_ID);

  Try<PID<Slave>> slave = StartSlave(&exec);
  ASSERT_SOME(slave);

  AWAIT_READY(slaveRegisteredMessage);
  SlaveID slaveId = slaveRegisteredMessage.get().slave_id();

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillRepeatedly(Return());

  driver.start();

  AWAIT_READY(frameworkId);

  // Drop the first shutdown message from the master (simulated
  // partition), allow the second shutdown message to pass when
  // the slave sends an update.
  Future<ShutdownMessage> shutdownMessage =
    DROP_PROTOBUF(ShutdownMessage(), _, slave.get());

  EXPECT_CALL(sched, offerRescinded(&driver, _))
    .WillRepeatedly(Return());

  Future<Nothing> slaveLost;
  EXPECT_CALL(sched, slaveLost(&driver, _))
    .WillOnce(FutureSatisfy(&slaveLost));

  Clock::pause();

  // Now, induce a partition of the slave by having the master
  // timeout the slave.
  size_t pings = 0;
  while (true) {
    AWAIT_READY(ping);
    pings++;
    if (pings == masterFlags.max_slave_ping_timeouts) {
     break;
    }
    ping = FUTURE_MESSAGE(Eq(PingSlaveMessage().GetTypeName()), _, _);
    Clock::advance(masterFlags.slave_ping_timeout);
    Clock::settle();
  }

  Clock::advance(masterFlags.slave_ping_timeout);
  Clock::settle();

  // Wait for the master to attempt to shut down the slave.
  AWAIT_READY(shutdownMessage);

  // The master will notify the framework that the slave was lost.
  AWAIT_READY(slaveLost);

  shutdownMessage = FUTURE_PROTOBUF(ShutdownMessage(), _, slave.get());

  // At this point, the slave still thinks it's registered, so we
  // simulate a status update coming from the slave.
  TaskID taskId;
  taskId.set_value("task_id");
  const StatusUpdate& update = protobuf::createStatusUpdate(
      frameworkId.get(),
      slaveId,
      taskId,
      TASK_RUNNING,
      TaskStatus::SOURCE_SLAVE,
      UUID::random());

  StatusUpdateMessage message;
  message.mutable_update()->CopyFrom(update);
  message.set_pid(stringify(slave.get()));

  process::post(master.get(), message);

  // The master should shutdown the slave upon receiving the update.
  AWAIT_READY(shutdownMessage);

  Clock::resume();

  driver.stop();
  driver.join();

  Shutdown();
}


// The purpose of this test is to ensure that when slaves are removed
// from the master, and then attempt to send exited executor messages,
// we send a ShutdownMessage to the slave. Why? Because during a
// network partition, the master will remove a partitioned slave, thus
// sending its tasks to LOST. At this point, when the partition is
// removed, the slave may attempt to send exited executor messages if
// it was unaware that the master removed it. We've already
// notified frameworks that the tasks under the executors were LOST,
// so we have to have the slave shut down.
TEST_F(PartitionTest, PartitionedSlaveExitedExecutor)
{
  master::Flags masterFlags = CreateMasterFlags();
  Try<PID<Master>> master = StartMaster(masterFlags);
  ASSERT_SOME(master);

  // Allow the master to PING the slave, but drop all PONG messages
  // from the slave. Note that we don't match on the master / slave
  // PIDs because it's actually the SlaveObserver Process that sends
  // the pings.
  Future<Message> ping = FUTURE_MESSAGE(
      Eq(PingSlaveMessage().GetTypeName()), _, _);

  DROP_PROTOBUFS(PongSlaveMessage(), _, _);

  MockExecutor exec(DEFAULT_EXECUTOR_ID);
  TestContainerizer containerizer(&exec);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));\

  Future<vector<Offer>> offers;
  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return());

  driver.start();

  AWAIT_READY(frameworkId);
  AWAIT_READY(offers);
  ASSERT_NE(0u, offers.get().size());

  // Launch a task. This allows us to have the slave send an
  // ExitedExecutorMessage.
  TaskID taskId;
  taskId.set_value("1");

  TaskInfo task;
  task.set_name("");
  task.mutable_task_id()->MergeFrom(taskId);
  task.mutable_slave_id()->MergeFrom(offers.get()[0].slave_id());
  task.mutable_resources()->MergeFrom(offers.get()[0].resources());
  task.mutable_executor()->MergeFrom(DEFAULT_EXECUTOR_INFO);
  task.mutable_executor()->mutable_command()->set_value("sleep 60");

  // Set up the expectations for launching the task.
  EXPECT_CALL(exec, registered(_, _, _, _));

  EXPECT_CALL(exec, launchTask(_, _))
    .WillOnce(SendStatusUpdateFromTask(TASK_RUNNING));

  // Drop all the status updates from the slave, so that we can
  // ensure the ExitedExecutorMessage is what triggers the slave
  // shutdown.
  DROP_PROTOBUFS(StatusUpdateMessage(), _, master.get());

  driver.launchTasks(offers.get()[0].id(), {task});

  // Drop the first shutdown message from the master (simulated
  // partition) and allow the second shutdown message to pass when
  // triggered by the ExitedExecutorMessage.
  Future<ShutdownMessage> shutdownMessage =
    DROP_PROTOBUF(ShutdownMessage(), _, slave.get());

  Future<TaskStatus> lostStatus;
  EXPECT_CALL(sched, statusUpdate(&driver, _))
    .WillOnce(FutureArg<1>(&lostStatus));

  Future<Nothing> slaveLost;
  EXPECT_CALL(sched, slaveLost(&driver, _))
    .WillOnce(FutureSatisfy(&slaveLost));

  Clock::pause();

  // Now, induce a partition of the slave by having the master
  // timeout the slave.
  size_t pings = 0;
  while (true) {
    AWAIT_READY(ping);
    pings++;
    if (pings == masterFlags.max_slave_ping_timeouts) {
     break;
    }
    ping = FUTURE_MESSAGE(Eq(PingSlaveMessage().GetTypeName()), _, _);
    Clock::advance(masterFlags.slave_ping_timeout);
    Clock::settle();
  }

  Clock::advance(masterFlags.slave_ping_timeout);
  Clock::settle();

  // The master will have notified the framework of the lost task.
  AWAIT_READY(lostStatus);
  EXPECT_EQ(TASK_LOST, lostStatus.get().state());

  // Wait for the master to attempt to shut down the slave.
  AWAIT_READY(shutdownMessage);

  // The master will notify the framework that the slave was lost.
  AWAIT_READY(slaveLost);

  shutdownMessage = FUTURE_PROTOBUF(ShutdownMessage(), _, slave.get());

  // Induce an ExitedExecutorMessage from the slave.
  containerizer.destroy(
      frameworkId.get(), DEFAULT_EXECUTOR_INFO.executor_id());

  // Upon receiving the message, the master will shutdown the slave.
  AWAIT_READY(shutdownMessage);

  Clock::resume();

  driver.stop();
  driver.join();

  Shutdown();
}


// This test verifies that if master --> slave socket closes and the
// slave is not aware of it (i.e., one way network partition), slave
// will re-register with the master.
TEST_F(PartitionTest, OneWayPartitionMasterToSlave)
{
  // Start a master.
  master::Flags masterFlags = CreateMasterFlags();
  Try<PID<Master>> master = StartMaster(masterFlags);
  ASSERT_SOME(master);

  Future<Message> slaveRegisteredMessage =
    FUTURE_MESSAGE(Eq(SlaveRegisteredMessage().GetTypeName()), _, _);

  // Ensure a ping reaches the slave.
  Future<Message> ping = FUTURE_MESSAGE(
      Eq(PingSlaveMessage().GetTypeName()), _, _);

  Try<PID<Slave>> slave = StartSlave();
  ASSERT_SOME(slave);

  AWAIT_READY(slaveRegisteredMessage);

  AWAIT_READY(ping);

  Future<Nothing> deactivateSlave =
    FUTURE_DISPATCH(_, &MesosAllocatorProcess::deactivateSlave);

  // Inject a slave exited event at the master causing the master
  // to mark the slave as disconnected. The slave should not notice
  // it until the next ping is received.
  process::inject::exited(slaveRegisteredMessage.get().to, master.get());

  // Wait until master deactivates the slave.
  AWAIT_READY(deactivateSlave);

  Future<SlaveReregisteredMessage> slaveReregisteredMessage =
    FUTURE_PROTOBUF(SlaveReregisteredMessage(), _, _);

  // Ensure the slave observer marked the slave as deactivated.
  Clock::pause();
  Clock::settle();

  // Let the slave observer send the next ping.
  Clock::advance(masterFlags.slave_ping_timeout);

  // Slave should re-register.
  AWAIT_READY(slaveReregisteredMessage);

  Shutdown();
}

} // namespace tests {
} // namespace internal {
} // namespace mesos {
