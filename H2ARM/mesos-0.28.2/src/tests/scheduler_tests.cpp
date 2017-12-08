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

#include <string>
#include <queue>

#include <gmock/gmock.h>

#include <mesos/executor.hpp>

#include <mesos/v1/mesos.hpp>
#include <mesos/v1/resources.hpp>
#include <mesos/v1/scheduler.hpp>

#include <mesos/v1/scheduler/scheduler.hpp>

#include <process/clock.hpp>
#include <process/future.hpp>
#include <process/gmock.hpp>
#include <process/gtest.hpp>
#include <process/pid.hpp>
#include <process/queue.hpp>

#include <stout/lambda.hpp>
#include <stout/try.hpp>

#include "internal/devolve.hpp"
#include "internal/evolve.hpp"

#include "master/allocator/mesos/allocator.hpp"

#include "master/master.hpp"

#include "tests/containerizer.hpp"
#include "tests/mesos.hpp"

using mesos::internal::master::allocator::MesosAllocatorProcess;

using mesos::internal::master::Master;

using mesos::internal::slave::Containerizer;
using mesos::internal::slave::Slave;

using mesos::v1::scheduler::Call;
using mesos::v1::scheduler::Event;
using mesos::v1::scheduler::Mesos;

using process::Clock;
using process::Future;
using process::PID;
using process::Queue;

using std::string;

using testing::_;
using testing::AtMost;
using testing::DoAll;
using testing::Return;
using testing::WithParamInterface;

namespace mesos {
namespace internal {
namespace tests {


class SchedulerTest
  : public MesosTest,
    public WithParamInterface<ContentType> {};


// The scheduler library tests are parameterized by the content type
// of the HTTP request.
INSTANTIATE_TEST_CASE_P(
    ContentType,
    SchedulerTest,
    ::testing::Values(ContentType::PROTOBUF, ContentType::JSON));


// This test verifies that a scheduler can subscribe with the master.
TEST_P(SchedulerTest, Subscribe)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);
    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown();
}


// This test verifies that a scheduler can subscribe with the master after
// failing over to another instance.
TEST_P(SchedulerTest, SchedulerFailover)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId = subscribed.get().framework_id();

  auto scheduler2 = std::make_shared<MockV1HTTPScheduler>();

  EXPECT_CALL(*scheduler2, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  // Failover to another scheduler instance.
  scheduler::TestV1Mesos mesos2(master.get(), contentType, scheduler2);

  AWAIT_READY(connected);

  // The previously connected scheduler instance should receive an
  // error/disconnected event.
  Future<Nothing> error;
  EXPECT_CALL(*scheduler, error(_, _))
    .WillOnce(FutureSatisfy(&error));

  Future<Nothing> disconnected;
  EXPECT_CALL(*scheduler, disconnected(_))
    .WillOnce(FutureSatisfy(&disconnected));

  EXPECT_CALL(*scheduler2, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler2, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);
    subscribe->mutable_framework_info()->mutable_id()->CopyFrom(frameworkId);

    mesos2.send(call);
  }

  AWAIT_READY(error);
  AWAIT_READY(disconnected);
  AWAIT_READY(subscribed);

  EXPECT_EQ(frameworkId, subscribed.get().framework_id());

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  EXPECT_CALL(*scheduler2, disconnected(_))
    .Times(AtMost(1));

  Shutdown();
}


// This test verifies that the scheduler can subscribe after a master failover.
TEST_P(SchedulerTest, MasterFailover)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto detector = std::make_shared<StandaloneMasterDetector>(master.get());

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler, detector);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId = subscribed.get().framework_id();

  Future<Nothing> disconnected;
  EXPECT_CALL(*scheduler, disconnected(_))
    .WillOnce(FutureSatisfy(&disconnected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  // Failover the master.
  Stop(master.get());
  master = StartMaster(flags);
  ASSERT_SOME(master);

  AWAIT_READY(disconnected);

  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  detector->appoint(master.get());

  AWAIT_READY(connected);

  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);
    subscribe->mutable_framework_info()->mutable_id()->CopyFrom(frameworkId);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  EXPECT_EQ(frameworkId, subscribed.get().framework_id());

  Shutdown();
}


TEST_P(SchedulerTest, TaskRunning)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed.get().framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_RUNNING));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged));

  Future<Event::Update> statusUpdate;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&statusUpdate));

  Future<Nothing> update;
  EXPECT_CALL(containerizer, update(_, _))
    .WillOnce(DoAll(FutureSatisfy(&update),
                    Return(Nothing())))
    .WillRepeatedly(Return(Future<Nothing>())); // Ignore subsequent calls.

  v1::TaskInfo taskInfo;
  taskInfo.set_name("");
  taskInfo.mutable_task_id()->set_value("1");
  taskInfo.mutable_agent_id()->CopyFrom(
      offers->offers(0).agent_id());
  taskInfo.mutable_resources()->CopyFrom(
      offers->offers(0).resources());
  taskInfo.mutable_executor()->CopyFrom(DEFAULT_V1_EXECUTOR_INFO);

  // TODO(benh): Enable just running a task with a command in the tests:
  //   taskInfo.mutable_command()->set_value("sleep 10");

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offers->offers(0).id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(statusUpdate);

  EXPECT_EQ(v1::TASK_RUNNING, statusUpdate->status().state());
  EXPECT_TRUE(statusUpdate->status().has_executor_id());
  EXPECT_EQ(executorId, devolve(statusUpdate->status().executor_id()));

  AWAIT_READY(update);

  EXPECT_CALL(*executor, shutdown(_))
    .Times(AtMost(1));

  EXPECT_CALL(*executor, disconnected(_))
    .Times(AtMost(1));

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, ReconcileTask)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_RUNNING));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged));

  Future<Event::Update> update1;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update1));

  const v1::Offer& offer = offers->offers(0);

  v1::TaskInfo taskInfo =
    evolve(createTask(devolve(offer), "", DEFAULT_EXECUTOR_ID));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offer.id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(update1);

  EXPECT_EQ(v1::TASK_RUNNING, update1->status().state());

  Future<Event::Update> update2;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update2));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::RECONCILE);

    Call::Reconcile::Task* task = call.mutable_reconcile()->add_tasks();
    task->mutable_task_id()->CopyFrom(taskInfo.task_id());

    mesos.send(call);
  }

  AWAIT_READY(update2);

  EXPECT_FALSE(update2->status().has_uuid());
  EXPECT_EQ(v1::TASK_RUNNING, update2->status().state());
  EXPECT_EQ(v1::TaskStatus::REASON_RECONCILIATION,
            update2->status().reason());

  EXPECT_CALL(*executor, shutdown(_))
    .Times(AtMost(1));

  EXPECT_CALL(*executor, disconnected(_))
    .Times(AtMost(1));

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, KillTask)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed.get().framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_RUNNING));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged))
    .WillRepeatedly(Return());

  Future<Event::Update> update1;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update1));

  const v1::Offer& offer = offers->offers(0);

  v1::TaskInfo taskInfo =
    evolve(createTask(devolve(offer), "", DEFAULT_EXECUTOR_ID));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offer.id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(update1);

  EXPECT_EQ(v1::TASK_RUNNING, update1->status().state());

  {
    // Acknowledge TASK_RUNNING update.
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACKNOWLEDGE);

    Call::Acknowledge* acknowledge = call.mutable_acknowledge();
    acknowledge->mutable_task_id()->CopyFrom(taskInfo.task_id());
    acknowledge->mutable_agent_id()->CopyFrom(offer.agent_id());
    acknowledge->set_uuid(update1->status().uuid());

    mesos.send(call);
  }

  Future<Event::Update> update2;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update2));

  EXPECT_CALL(*executor, kill(_, _))
    .WillOnce(executor::SendUpdateFromTaskID(
        frameworkId, evolve(executorId), v1::TASK_KILLED));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::KILL);

    Call::Kill* kill = call.mutable_kill();
    kill->mutable_task_id()->CopyFrom(taskInfo.task_id());
    kill->mutable_agent_id()->CopyFrom(offer.agent_id());

    mesos.send(call);
  }

  AWAIT_READY(update2);

  EXPECT_EQ(v1::TASK_KILLED, update2->status().state());

  EXPECT_CALL(*executor, shutdown(_))
    .Times(AtMost(1));

  EXPECT_CALL(*executor, disconnected(_))
    .Times(AtMost(1));

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, ShutdownExecutor)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return());

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_FINISHED));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged));

  Future<Event::Update> update;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update));

  const v1::Offer& offer = offers->offers(0);

  v1::TaskInfo taskInfo =
    evolve(createTask(devolve(offer), "", DEFAULT_EXECUTOR_ID));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offer.id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(update);

  EXPECT_EQ(v1::TASK_FINISHED, update->status().state());

  Future<Nothing> shutdown;
  EXPECT_CALL(*executor, shutdown(_))
    .WillOnce(FutureSatisfy(&shutdown));

  Future<Event::Failure> failure;
  EXPECT_CALL(*scheduler, failure(_, _))
    .WillOnce(FutureArg<1>(&failure));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::SHUTDOWN);

    Call::Shutdown* shutdown = call.mutable_shutdown();
    shutdown->mutable_executor_id()->CopyFrom(DEFAULT_V1_EXECUTOR_ID);
    shutdown->mutable_agent_id()->CopyFrom(offer.agent_id());

    mesos.send(call);
  }

  AWAIT_READY(shutdown);
  containerizer.destroy(devolve(frameworkId), executorId);

  // Executor termination results in a 'FAILURE' event.
  AWAIT_READY(failure);
  EXPECT_EQ(executorId, devolve(failure->executor_id()));

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Teardown)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_RUNNING));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged));

  Future<Event::Update> update;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update));

  const v1::Offer& offer = offers->offers(0);

  v1::TaskInfo taskInfo =
    evolve(createTask(devolve(offer), "", DEFAULT_EXECUTOR_ID));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offer.id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(update);

  EXPECT_EQ(v1::TASK_RUNNING, update->status().state());

  Future<Nothing> shutdown;
  EXPECT_CALL(*executor, shutdown(_))
    .WillOnce(FutureSatisfy(&shutdown));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::TEARDOWN);

    mesos.send(call);
  }

  AWAIT_READY(shutdown);

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Decline)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Try<PID<Slave>> slave = StartSlave();
  ASSERT_SOME(slave);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers1;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers1));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers1);
  ASSERT_EQ(1, offers1->offers().size());

  const v1::Offer& offer = offers1->offers(0);

  Future<Event::Offers> offers2;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers2));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::DECLINE);

    Call::Decline* decline = call.mutable_decline();
    decline->add_offer_ids()->CopyFrom(offer.id());

    // Set 0s filter to immediately get another offer.
    v1::Filters filters;
    filters.set_refuse_seconds(0);
    decline->mutable_filters()->CopyFrom(filters);

    mesos.send(call);
  }

  // If the resources were properly declined, the scheduler should
  // get another offer with same amount of resources.
  AWAIT_READY(offers2);
  ASSERT_EQ(1, offers2->offers().size());
  ASSERT_EQ(offer.resources(), offers2->offers(0).resources());

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Revive)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Try<PID<Slave>> slave = StartSlave();
  ASSERT_SOME(slave);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers1;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers1));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers1);
  EXPECT_NE(0, offers1->offers().size());

  const v1::Offer& offer = offers1->offers(0);

  Future<Event::Offers> offers2;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers2));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::DECLINE);

    Call::Decline* decline = call.mutable_decline();
    decline->add_offer_ids()->CopyFrom(offer.id());

    // Set 1hr filter to not immediately get another offer.
    v1::Filters filters;
    filters.set_refuse_seconds(Hours(1).secs());
    decline->mutable_filters()->CopyFrom(filters);

    mesos.send(call);
  }

  // No offers should be sent within 30 mins because we set a filter
  // for 1 hr.
  Clock::pause();
  Clock::advance(Minutes(30));
  Clock::settle();

  ASSERT_TRUE(offers2.isPending());

  // On revival the filters should be cleared and the scheduler should
  // get another offer with same amount of resources.
  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::REVIVE);

    mesos.send(call);
  }

  AWAIT_READY(offers2);
  EXPECT_NE(0, offers2->offers().size());
  ASSERT_EQ(offer.resources(), offers2->offers(0).resources());

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Suppress)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Try<PID<Slave>> slave = StartSlave();
  ASSERT_SOME(slave);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers1;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers1));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers1);
  EXPECT_NE(0, offers1->offers().size());

  const v1::Offer& offer = offers1->offers(0);

  Future<Event::Offers> offers2;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers2));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::DECLINE);

    Call::Decline* decline = call.mutable_decline();
    decline->add_offer_ids()->CopyFrom(offer.id());

    // Set 1hr filter to not immediately get another offer.
    v1::Filters filters;
    filters.set_refuse_seconds(Hours(1).secs());
    decline->mutable_filters()->CopyFrom(filters);

    mesos.send(call);
  }

  Future<Nothing> suppressOffers =
    FUTURE_DISPATCH(_, &MesosAllocatorProcess::suppressOffers);

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::SUPPRESS);

    mesos.send(call);
  }

  AWAIT_READY(suppressOffers);

  // Wait for allocator to finish executing 'suppressOffers()'.
  Clock::pause();
  Clock::settle();

  // No offers should be sent within 100 mins because the framework
  // suppressed offers.
  Clock::advance(Minutes(100));
  Clock::settle();

  ASSERT_TRUE(offers2.isPending());

  // On reviving offers the scheduler should get another offer with same amount
  // of resources.
  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::REVIVE);

    mesos.send(call);
  }

  AWAIT_READY(offers2);

  EXPECT_NE(0, offers2->offers().size());
  ASSERT_EQ(offer.resources(), offers2->offers(0).resources());

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Message)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();
  auto executor = std::make_shared<MockV1HTTPExecutor>();

  ExecutorID executorId = DEFAULT_EXECUTOR_ID;
  TestContainerizer containerizer(executorId, executor);

  Try<PID<Slave>> slave = StartSlave(&containerizer);
  ASSERT_SOME(slave);

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  Future<Event::Offers> offers;
  EXPECT_CALL(*scheduler, offers(_, _))
    .WillOnce(FutureArg<1>(&offers));

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  AWAIT_READY(offers);
  EXPECT_NE(0, offers->offers().size());

  EXPECT_CALL(*executor, connected(_))
    .WillOnce(executor::SendSubscribe(frameworkId, evolve(executorId)));

  EXPECT_CALL(*executor, subscribed(_, _));

  EXPECT_CALL(*executor, launch(_, _))
    .WillOnce(executor::SendUpdateFromTask(
        frameworkId, evolve(executorId), v1::TASK_RUNNING));

  Future<Nothing> acknowledged;
  EXPECT_CALL(*executor, acknowledged(_, _))
    .WillOnce(FutureSatisfy(&acknowledged));

  Future<Event::Update> update;
  EXPECT_CALL(*scheduler, update(_, _))
    .WillOnce(FutureArg<1>(&update));

  const v1::Offer& offer = offers->offers(0);

  v1::TaskInfo taskInfo =
    evolve(createTask(devolve(offer), "", DEFAULT_EXECUTOR_ID));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::ACCEPT);

    Call::Accept* accept = call.mutable_accept();
    accept->add_offer_ids()->CopyFrom(offer.id());

    v1::Offer::Operation* operation = accept->add_operations();
    operation->set_type(v1::Offer::Operation::LAUNCH);
    operation->mutable_launch()->add_task_infos()->CopyFrom(taskInfo);

    mesos.send(call);
  }

  AWAIT_READY(acknowledged);
  AWAIT_READY(update);

  EXPECT_EQ(v1::TASK_RUNNING, update->status().state());

  Future<v1::executor::Event::Message> message;
  EXPECT_CALL(*executor, message(_, _))
    .WillOnce(FutureArg<1>(&message));

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::MESSAGE);

    Call::Message* message = call.mutable_message();
    message->mutable_agent_id()->CopyFrom(offer.agent_id());
    message->mutable_executor_id()->CopyFrom(DEFAULT_V1_EXECUTOR_ID);
    message->set_data("hello world");

    mesos.send(call);
  }

  AWAIT_READY(message);
  ASSERT_EQ("hello world", message->data());

  EXPECT_CALL(*executor, shutdown(_))
    .Times(AtMost(1));

  EXPECT_CALL(*executor, disconnected(_))
    .Times(AtMost(1));

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown(); // Must shutdown before 'containerizer' gets deallocated.
}


TEST_P(SchedulerTest, Request)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  auto scheduler = std::make_shared<MockV1HTTPScheduler>();

  Future<Nothing> connected;
  EXPECT_CALL(*scheduler, connected(_))
    .WillOnce(FutureSatisfy(&connected))
    .WillRepeatedly(Return()); // Ignore future invocations.

  ContentType contentType = GetParam();

  scheduler::TestV1Mesos mesos(master.get(), contentType, scheduler);

  AWAIT_READY(connected);

  Future<Event::Subscribed> subscribed;
  EXPECT_CALL(*scheduler, subscribed(_, _))
    .WillOnce(FutureArg<1>(&subscribed));

  EXPECT_CALL(*scheduler, heartbeat(_))
    .WillRepeatedly(Return()); // Ignore heartbeats.

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    mesos.send(call);
  }

  AWAIT_READY(subscribed);

  v1::FrameworkID frameworkId(subscribed->framework_id());

  Future<Nothing> requestResources =
    FUTURE_DISPATCH(_, &MesosAllocatorProcess::requestResources);

  {
    Call call;
    call.mutable_framework_id()->CopyFrom(frameworkId);
    call.set_type(Call::REQUEST);

    // Create a dummy request.
    Call::Request* request = call.mutable_request();
    request->add_requests();

    mesos.send(call);
  }

  AWAIT_READY(requestResources);

  EXPECT_CALL(*scheduler, disconnected(_))
    .Times(AtMost(1));

  Shutdown();
}


// TODO(benh): Write test for sending Call::Acknowledgement through
// master to slave when Event::Update was generated locally.

} // namespace tests {
} // namespace internal {
} // namespace mesos {
