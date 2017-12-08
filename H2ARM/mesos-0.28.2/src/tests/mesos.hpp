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

#ifndef __TESTS_MESOS_HPP__
#define __TESTS_MESOS_HPP__

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include <mesos/executor.hpp>
#include <mesos/scheduler.hpp>

#include <mesos/v1/executor.hpp>
#include <mesos/v1/scheduler.hpp>

#include <mesos/v1/executor/executor.hpp>

#include <mesos/v1/scheduler/scheduler.hpp>

#include <mesos/authorizer/authorizer.hpp>

#include <mesos/fetcher/fetcher.hpp>

#include <mesos/slave/container_logger.hpp>
#include <mesos/slave/qos_controller.hpp>
#include <mesos/slave/resource_estimator.hpp>

#include <process/future.hpp>
#include <process/gmock.hpp>
#include <process/gtest.hpp>
#include <process/owned.hpp>
#include <process/pid.hpp>
#include <process/process.hpp>
#include <process/queue.hpp>

#include <process/ssl/gtest.hpp>

#include <stout/bytes.hpp>
#include <stout/foreach.hpp>
#include <stout/gtest.hpp>
#include <stout/json.hpp>
#include <stout/lambda.hpp>
#include <stout/none.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/stringify.hpp>
#include <stout/try.hpp>
#include <stout/uuid.hpp>

#include "common/http.hpp"

#include "messages/messages.hpp" // For google::protobuf::Message.

#include "master/detector.hpp"
#include "master/master.hpp"

#include "slave/slave.hpp"

#include "slave/containerizer/containerizer.hpp"
#include "slave/containerizer/docker.hpp"
#include "slave/containerizer/fetcher.hpp"

#include "slave/containerizer/mesos/containerizer.hpp"

#include "tests/cluster.hpp"
#include "tests/limiter.hpp"
#include "tests/utils.hpp"

#ifdef MESOS_HAS_JAVA
#include "tests/zookeeper.hpp"
#endif // MESOS_HAS_JAVA

using ::testing::_;
using ::testing::An;
using ::testing::DoDefault;
using ::testing::Invoke;
using ::testing::Return;

namespace mesos {
namespace internal {
namespace tests {

// Forward declarations.
class MockExecutor;


// NOTE: `SSLTemporaryDirectoryTest` exists even when SSL is not compiled into
// Mesos.  In this case, the class is an alias of `TemporaryDirectoryTest`.
class MesosTest : public SSLTemporaryDirectoryTest
{
protected:
  MesosTest(const Option<zookeeper::URL>& url = None());

  virtual void TearDown();

  // Returns the flags used to create masters.
  virtual master::Flags CreateMasterFlags();

  // Returns the flags used to create slaves.
  virtual slave::Flags CreateSlaveFlags();

  // Starts a master with the specified flags.
  virtual Try<process::PID<master::Master> > StartMaster(
      const Option<master::Flags>& flags = None());

  // Starts a master with the specified allocator process and flags.
  virtual Try<process::PID<master::Master> > StartMaster(
      mesos::master::allocator::Allocator* allocator,
      const Option<master::Flags>& flags = None());

  // Starts a master with the specified authorizer and flags.
  // Waits for the master to detect a leader (could be itself) before
  // returning if 'wait' is set to true.
  virtual Try<process::PID<master::Master> > StartMaster(
      Authorizer* authorizer,
      const Option<master::Flags>& flags = None());

  // Starts a master with a slave removal rate limiter and flags.
  virtual Try<process::PID<master::Master> > StartMaster(
      const std::shared_ptr<MockRateLimiter>& slaveRemovalLimiter,
      const Option<master::Flags>& flags = None());

  // TODO(bmahler): Consider adding a builder style interface, e.g.
  //
  // Try<PID<Slave> > slave =
  //   Slave().With(flags)
  //          .With(executor)
  //          .With(containerizer)
  //          .With(detector)
  //          .With(gc)
  //          .Start();
  //
  // Or options:
  //
  // Injections injections;
  // injections.executor = executor;
  // injections.containerizer = containerizer;
  // injections.detector = detector;
  // injections.gc = gc;
  // Try<PID<Slave> > slave = StartSlave(injections);

  // Starts a slave with the specified flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified mock executor and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      MockExecutor* executor,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified containerizer and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      slave::Containerizer* containerizer,
      const Option<slave::Flags>& flags = None());

  virtual Try<process::PID<slave::Slave> > StartSlave(
      slave::Containerizer* containerizer,
      const std::string& id,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified containerizer, detector and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      slave::Containerizer* containerizer,
      MasterDetector* detector,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified MasterDetector and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      MasterDetector* detector,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified MasterDetector, GC, and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      MasterDetector* detector,
      slave::GarbageCollector* gc,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified mock executor, MasterDetector
  // and flags.
  virtual Try<process::PID<slave::Slave> > StartSlave(
      MockExecutor* executor,
      MasterDetector* detector,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified resource estimator and flags.
  virtual Try<process::PID<slave::Slave>> StartSlave(
      mesos::slave::ResourceEstimator* resourceEstimator,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified mock executor, resource
  // estimator and flags.
  virtual Try<process::PID<slave::Slave>> StartSlave(
      MockExecutor* executor,
      mesos::slave::ResourceEstimator* resourceEstimator,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified resource estimator,
  // containerizer and flags.
  virtual Try<process::PID<slave::Slave>> StartSlave(
      slave::Containerizer* containerizer,
      mesos::slave::ResourceEstimator* resourceEstimator,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified QoS Controller and flags.
  virtual Try<process::PID<slave::Slave>> StartSlave(
      mesos::slave::QoSController* qosController,
      const Option<slave::Flags>& flags = None());

  // Starts a slave with the specified QoS Controller,
  // containerizer and flags.
  virtual Try<process::PID<slave::Slave>> StartSlave(
      slave::Containerizer* containerizer,
      mesos::slave::QoSController* qosController,
      const Option<slave::Flags>& flags = None());

  // Stop the specified master.
  virtual void Stop(
      const process::PID<master::Master>& pid);

  // Stop the specified slave.
  virtual void Stop(
      const process::PID<slave::Slave>& pid,
      bool shutdown = false);

  // Stop all masters and slaves.
  virtual void Shutdown();

  // Stop all masters.
  virtual void ShutdownMasters();

  // Stop all slaves.
  virtual void ShutdownSlaves();

  Cluster cluster;

  // Containerizer(s) created during test that we need to cleanup.
  std::map<process::PID<slave::Slave>, slave::Containerizer*> containerizers;

  const std::string defaultAgentResourcesString{
    "cpus:2;mem:1024;disk:1024;ports:[31000-32000]"};
};


template <typename T>
class ContainerizerTest : public MesosTest {};

#ifdef __linux__
// Cgroups hierarchy used by the cgroups related tests.
const static std::string TEST_CGROUPS_HIERARCHY = "/tmp/mesos_test_cgroup";

// Name of the root cgroup used by the cgroups related tests.
const static std::string TEST_CGROUPS_ROOT = "mesos_test";


template <>
class ContainerizerTest<slave::MesosContainerizer> : public MesosTest
{
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

protected:
  virtual slave::Flags CreateSlaveFlags();
  virtual void SetUp();
  virtual void TearDown();

private:
  // Base hierarchy for separately mounted cgroup controllers, e.g., if the
  // base hierachy is /sys/fs/cgroup then each controller will be mounted to
  // /sys/fs/cgroup/{controller}/.
  std::string baseHierarchy;

  // Set of cgroup subsystems used by the cgroups related tests.
  hashset<std::string> subsystems;
};
#else
template <>
class ContainerizerTest<slave::MesosContainerizer> : public MesosTest
{
protected:
  virtual slave::Flags CreateSlaveFlags();
};
#endif // __linux__


#ifdef MESOS_HAS_JAVA

class MesosZooKeeperTest : public MesosTest
{
public:
  static void SetUpTestCase()
  {
    // Make sure the JVM is created.
    ZooKeeperTest::SetUpTestCase();

    // Launch the ZooKeeper test server.
    server = new ZooKeeperTestServer();
    server->startNetwork();

    Try<zookeeper::URL> parse = zookeeper::URL::parse(
        "zk://" + server->connectString() + "/znode");
    ASSERT_SOME(parse);

    url = parse.get();
  }

  static void TearDownTestCase()
  {
    delete server;
    server = NULL;
  }

  virtual void SetUp()
  {
    MesosTest::SetUp();
    server->startNetwork();
  }

  virtual void TearDown()
  {
    server->shutdownNetwork();
    MesosTest::TearDown();
  }

protected:
  MesosZooKeeperTest() : MesosTest(url) {}

  virtual master::Flags CreateMasterFlags()
  {
    master::Flags flags = MesosTest::CreateMasterFlags();

    // NOTE: Since we are using the replicated log with ZooKeeper
    // (default storage in MesosTest), we need to specify the quorum.
    flags.quorum = 1;

    return flags;
  }

  static ZooKeeperTestServer* server;
  static Option<zookeeper::URL> url;
};

#endif // MESOS_HAS_JAVA


// Macros to get/create (default) ExecutorInfos and FrameworkInfos.
#define DEFAULT_EXECUTOR_INFO                                           \
      ({ ExecutorInfo executor;                                         \
        executor.mutable_executor_id()->set_value("default");           \
        executor.mutable_command()->set_value("exit 1");                \
        executor; })


#define DEFAULT_V1_EXECUTOR_INFO                                        \
     ({ v1::ExecutorInfo executor;                                      \
        executor.mutable_executor_id()->set_value("default");           \
        executor.mutable_command()->set_value("exit 1");                \
        executor; })


#define CREATE_EXECUTOR_INFO(executorId, command)                       \
      ({ ExecutorInfo executor;                                         \
        executor.mutable_executor_id()->set_value(executorId);          \
        executor.mutable_command()->set_value(command);                 \
        executor; })


#define DEFAULT_CREDENTIAL                                             \
     ({ Credential credential;                                         \
        credential.set_principal("test-principal");                    \
        credential.set_secret("test-secret");                          \
        credential; })


#define DEFAULT_CREDENTIAL_2                                           \
     ({ Credential credential;                                         \
        credential.set_principal("test-principal-2");                  \
        credential.set_secret("test-secret-2");                        \
        credential; })


#define DEFAULT_V1_CREDENTIAL                                          \
     ({ v1::Credential credential;                                     \
        credential.set_principal("test-principal");                    \
        credential.set_secret("test-secret");                          \
        credential; })


#define DEFAULT_FRAMEWORK_INFO                                          \
     ({ FrameworkInfo framework;                                        \
        framework.set_name("default");                                  \
        framework.set_user(os::user().get());                           \
        framework.set_principal(DEFAULT_CREDENTIAL.principal());        \
        framework; })


#define DEFAULT_V1_FRAMEWORK_INFO                                       \
     ({ v1::FrameworkInfo framework;                                    \
        framework.set_name("default");                                  \
        framework.set_user(os::user().get());                           \
        framework.set_principal(DEFAULT_CREDENTIAL.principal());        \
        framework; })


#define DEFAULT_EXECUTOR_ID           \
      DEFAULT_EXECUTOR_INFO.executor_id()


#define DEFAULT_V1_EXECUTOR_ID         \
      DEFAULT_V1_EXECUTOR_INFO.executor_id()


#define DEFAULT_CONTAINER_ID                                          \
     ({ ContainerID containerId;                                      \
        containerId.set_value("container");                           \
        containerId; })


#define CREATE_COMMAND_INFO(command)                                  \
  ({ CommandInfo commandInfo;                                         \
     commandInfo.set_value(command);                                  \
     commandInfo; })


// TODO(jieyu): Consider making it a function to support more
// overloads (e.g., createVolumeFromHost, createVolumeFromImage).
#define CREATE_VOLUME(containerPath, hostPath, mode)                  \
      ({ Volume volume;                                               \
         volume.set_container_path(containerPath);                    \
         volume.set_host_path(hostPath);                              \
         volume.set_mode(mode);                                       \
         volume; })


// TODO(bmahler): Refactor this to make the distinction between
// command tasks and executor tasks clearer.
inline TaskInfo createTask(
    const SlaveID& slaveId,
    const Resources& resources,
    const CommandInfo& command,
    const Option<mesos::ExecutorID>& executorId = None(),
    const std::string& name = "test-task",
    const std::string& id = UUID::random().toString())
{
  TaskInfo task;
  task.set_name(name);
  task.mutable_task_id()->set_value(id);
  task.mutable_slave_id()->CopyFrom(slaveId);
  task.mutable_resources()->CopyFrom(resources);
  if (executorId.isSome()) {
    ExecutorInfo executor;
    executor.mutable_executor_id()->CopyFrom(executorId.get());
    executor.mutable_command()->CopyFrom(command);
    task.mutable_executor()->CopyFrom(executor);
  } else {
    task.mutable_command()->CopyFrom(command);
  }

  return task;
}


inline TaskInfo createTask(
    const SlaveID& slaveId,
    const Resources& resources,
    const std::string& command,
    const Option<mesos::ExecutorID>& executorId = None(),
    const std::string& name = "test-task",
    const std::string& id = UUID::random().toString())
{
  return createTask(
      slaveId,
      resources,
      CREATE_COMMAND_INFO(command),
      executorId,
      name,
      id);
}


inline TaskInfo createTask(
    const Offer& offer,
    const std::string& command,
    const Option<mesos::ExecutorID>& executorId = None(),
    const std::string& name = "test-task",
    const std::string& id = UUID::random().toString())
{
  return createTask(
      offer.slave_id(),
      offer.resources(),
      command,
      executorId,
      name,
      id);
}


inline Resource::ReservationInfo createReservationInfo(
    const Option<std::string>& principal = None(),
    const Option<Labels>& labels = None())
{
  Resource::ReservationInfo info;

  if (principal.isSome()) {
    info.set_principal(principal.get());
  }

  if (labels.isSome()) {
    info.mutable_labels()->CopyFrom(labels.get());
  }

  return info;
}


inline Resource createReservedResource(
    const std::string& name,
    const std::string& value,
    const std::string& role,
    const Option<Resource::ReservationInfo>& reservation)
{
  Resource resource = Resources::parse(name, value, role).get();

  if (reservation.isSome()) {
    resource.mutable_reservation()->CopyFrom(reservation.get());
  }

  return resource;
}


// NOTE: We only set the volume in DiskInfo if 'containerPath' is set.
// If volume mode is not specified, Volume::RW will be used (assuming
// 'containerPath' is set).
inline Resource::DiskInfo createDiskInfo(
    const Option<std::string>& persistenceId,
    const Option<std::string>& containerPath,
    const Option<Volume::Mode>& mode = None(),
    const Option<std::string>& hostPath = None(),
    const Option<Resource::DiskInfo::Source>& source = None())
{
  Resource::DiskInfo info;

  if (persistenceId.isSome()) {
    info.mutable_persistence()->set_id(persistenceId.get());
  }

  if (containerPath.isSome()) {
    Volume volume;
    volume.set_container_path(containerPath.get());
    volume.set_mode(mode.isSome() ? mode.get() : Volume::RW);

    if (hostPath.isSome()) {
      volume.set_host_path(hostPath.get());
    }

    info.mutable_volume()->CopyFrom(volume);
  }

  if (source.isSome()) {
    info.mutable_source()->CopyFrom(source.get());
  }

  return info;
}


// Helper for creating a disk source with type `PATH`.
inline Resource::DiskInfo::Source createDiskSourcePath(const std::string& root)
{
  Resource::DiskInfo::Source source;

  source.set_type(Resource::DiskInfo::Source::PATH);
  source.mutable_path()->set_root(root);

  return source;
}


// Helper for creating a disk source with type `MOUNT`.
inline Resource::DiskInfo::Source createDiskSourceMount(const std::string& root)
{
  Resource::DiskInfo::Source source;

  source.set_type(Resource::DiskInfo::Source::MOUNT);
  source.mutable_mount()->set_root(root);

  return source;
}


// Helper for creating a disk resource.
inline Resource createDiskResource(
    const std::string& value,
    const std::string& role,
    const Option<std::string>& persistenceID,
    const Option<std::string>& containerPath,
    const Option<Resource::DiskInfo::Source>& source = None())
{
  Resource resource = Resources::parse("disk", value, role).get();

  if (persistenceID.isSome() || containerPath.isSome() || source.isSome()) {
    resource.mutable_disk()->CopyFrom(
        createDiskInfo(persistenceID, containerPath, None(), None(), source));
  }

  return resource;
}


// Note that `reservationPrincipal` should be specified if and only if
// the volume uses dynamically reserved resources.
inline Resource createPersistentVolume(
    const Bytes& size,
    const std::string& role,
    const std::string& persistenceId,
    const std::string& containerPath,
    const Option<std::string>& reservationPrincipal = None(),
    const Option<Resource::DiskInfo::Source>& source = None())
{
  Resource volume = Resources::parse(
      "disk",
      stringify(size.megabytes()),
      role).get();

  volume.mutable_disk()->CopyFrom(
      createDiskInfo(persistenceId, containerPath, None(), None(), source));

  if (reservationPrincipal.isSome()) {
    volume.mutable_reservation()->set_principal(reservationPrincipal.get());
  }

  return volume;
}


// Note that `reservationPrincipal` should be specified if and only if
// the volume uses dynamically reserved resources.
inline Resource createPersistentVolume(
    Resource volume,
    const std::string& persistenceId,
    const std::string& containerPath,
    const Option<std::string>& reservationPrincipal = None())
{
  Option<Resource::DiskInfo::Source> source = None();
  if (volume.has_disk() && volume.disk().has_source()) {
    source = volume.disk().source();
  }

  volume.mutable_disk()->CopyFrom(
      createDiskInfo(persistenceId, containerPath, None(), None(), source));

  if (reservationPrincipal.isSome()) {
    volume.mutable_reservation()->set_principal(reservationPrincipal.get());
  }

  return volume;
}


inline process::http::Headers createBasicAuthHeaders(
    const Credential& credential)
{
  return process::http::Headers{{
      "Authorization",
      "Basic " +
        base64::encode(credential.principal() + ":" + credential.secret())
  }};
}


// Helpers for creating reserve operations.
inline Offer::Operation RESERVE(const Resources& resources)
{
  Offer::Operation operation;
  operation.set_type(Offer::Operation::RESERVE);
  operation.mutable_reserve()->mutable_resources()->CopyFrom(resources);
  return operation;
}


// Helpers for creating unreserve operations.
inline Offer::Operation UNRESERVE(const Resources& resources)
{
  Offer::Operation operation;
  operation.set_type(Offer::Operation::UNRESERVE);
  operation.mutable_unreserve()->mutable_resources()->CopyFrom(resources);
  return operation;
}


// Helpers for creating offer operations.
inline Offer::Operation CREATE(const Resources& volumes)
{
  Offer::Operation operation;
  operation.set_type(Offer::Operation::CREATE);
  operation.mutable_create()->mutable_volumes()->CopyFrom(volumes);
  return operation;
}


inline Offer::Operation DESTROY(const Resources& volumes)
{
  Offer::Operation operation;
  operation.set_type(Offer::Operation::DESTROY);
  operation.mutable_destroy()->mutable_volumes()->CopyFrom(volumes);
  return operation;
}


inline Offer::Operation LAUNCH(const std::vector<TaskInfo>& tasks)
{
  Offer::Operation operation;
  operation.set_type(Offer::Operation::LAUNCH);

  foreach (const TaskInfo& task, tasks) {
    operation.mutable_launch()->add_task_infos()->CopyFrom(task);
  }

  return operation;
}


// Definition of a mock Scheduler to be used in tests with gmock.
class MockScheduler : public Scheduler
{
public:
  MockScheduler();
  virtual ~MockScheduler();

  MOCK_METHOD3(registered, void(SchedulerDriver*,
                                const FrameworkID&,
                                const MasterInfo&));
  MOCK_METHOD2(reregistered, void(SchedulerDriver*, const MasterInfo&));
  MOCK_METHOD1(disconnected, void(SchedulerDriver*));
  MOCK_METHOD2(resourceOffers, void(SchedulerDriver*,
                                    const std::vector<Offer>&));
  MOCK_METHOD2(offerRescinded, void(SchedulerDriver*, const OfferID&));
  MOCK_METHOD2(statusUpdate, void(SchedulerDriver*, const TaskStatus&));
  MOCK_METHOD4(frameworkMessage, void(SchedulerDriver*,
                                      const ExecutorID&,
                                      const SlaveID&,
                                      const std::string&));
  MOCK_METHOD2(slaveLost, void(SchedulerDriver*, const SlaveID&));
  MOCK_METHOD4(executorLost, void(SchedulerDriver*,
                                  const ExecutorID&,
                                  const SlaveID&,
                                  int));
  MOCK_METHOD2(error, void(SchedulerDriver*, const std::string&));
};

// For use with a MockScheduler, for example:
// EXPECT_CALL(sched, resourceOffers(_, _))
//   .WillOnce(LaunchTasks(EXECUTOR, TASKS, CPUS, MEM, ROLE));
// Launches up to TASKS no-op tasks, if possible,
// each with CPUS cpus and MEM memory and EXECUTOR executor.
ACTION_P5(LaunchTasks, executor, tasks, cpus, mem, role)
{
  SchedulerDriver* driver = arg0;
  std::vector<Offer> offers = arg1;
  int numTasks = tasks;

  int launched = 0;
  for (size_t i = 0; i < offers.size(); i++) {
    const Offer& offer = offers[i];

    const Resources TASK_RESOURCES = Resources::parse(
        "cpus:" + stringify(cpus) + ";mem:" + stringify(mem)).get();

    int nextTaskId = 0;
    std::vector<TaskInfo> tasks;
    Resources remaining = offer.resources();

    while (remaining.flatten().contains(TASK_RESOURCES) &&
           launched < numTasks) {
      TaskInfo task;
      task.set_name("TestTask");
      task.mutable_task_id()->set_value(stringify(nextTaskId++));
      task.mutable_slave_id()->MergeFrom(offer.slave_id());
      task.mutable_executor()->MergeFrom(executor);

      Option<Resources> resources =
        remaining.find(TASK_RESOURCES.flatten(role));

      CHECK_SOME(resources);

      task.mutable_resources()->MergeFrom(resources.get());
      remaining -= resources.get();

      tasks.push_back(task);
      launched++;
    }

    driver->launchTasks(offer.id(), tasks);
  }
}


// Like LaunchTasks, but decline the entire offer and
// don't launch any tasks.
ACTION(DeclineOffers)
{
  SchedulerDriver* driver = arg0;
  std::vector<Offer> offers = arg1;

  for (size_t i = 0; i < offers.size(); i++) {
    driver->declineOffer(offers[i].id());
  }
}


// Like DeclineOffers, but takes a custom filters object.
ACTION_P(DeclineOffers, filters)
{
  SchedulerDriver* driver = arg0;
  std::vector<Offer> offers = arg1;

  for (size_t i = 0; i < offers.size(); i++) {
    driver->declineOffer(offers[i].id(), filters);
  }
}


// For use with a MockScheduler, for example:
// process::Queue<Offer> offers;
// EXPECT_CALL(sched, resourceOffers(_, _))
//   .WillRepeatedly(EnqueueOffers(&offers));
// Enqueues all received offers into the provided queue.
ACTION_P(EnqueueOffers, queue)
{
  std::vector<Offer> offers = arg1;
  foreach (const Offer& offer, offers) {
    queue->put(offer);
  }
}


// Definition of a mock Executor to be used in tests with gmock.
class MockExecutor : public Executor
{
public:
  MockExecutor(const ExecutorID& _id);
  virtual ~MockExecutor();

  MOCK_METHOD4(registered, void(ExecutorDriver*,
                                const ExecutorInfo&,
                                const FrameworkInfo&,
                                const SlaveInfo&));
  MOCK_METHOD2(reregistered, void(ExecutorDriver*, const SlaveInfo&));
  MOCK_METHOD1(disconnected, void(ExecutorDriver*));
  MOCK_METHOD2(launchTask, void(ExecutorDriver*, const TaskInfo&));
  MOCK_METHOD2(killTask, void(ExecutorDriver*, const TaskID&));
  MOCK_METHOD2(frameworkMessage, void(ExecutorDriver*, const std::string&));
  MOCK_METHOD1(shutdown, void(ExecutorDriver*));
  MOCK_METHOD2(error, void(ExecutorDriver*, const std::string&));

  const ExecutorID id;
};


class TestingMesosSchedulerDriver : public MesosSchedulerDriver
{
public:
  TestingMesosSchedulerDriver(
      Scheduler* scheduler,
      MasterDetector* _detector)
    : MesosSchedulerDriver(
          scheduler,
          DEFAULT_FRAMEWORK_INFO,
          "",
          true,
          DEFAULT_CREDENTIAL)
  {
    // No-op destructor as _detector lives on the stack.
    detector =
      std::shared_ptr<MasterDetector>(_detector, [](MasterDetector*) {});
  }

  TestingMesosSchedulerDriver(
      Scheduler* scheduler,
      MasterDetector* _detector,
      const FrameworkInfo& framework,
      bool implicitAcknowledgements = true)
    : MesosSchedulerDriver(
          scheduler,
          framework,
          "",
          implicitAcknowledgements,
          DEFAULT_CREDENTIAL)
  {
    // No-op destructor as _detector lives on the stack.
    detector =
      std::shared_ptr<MasterDetector>(_detector, [](MasterDetector*) {});
  }

  TestingMesosSchedulerDriver(
      Scheduler* scheduler,
      MasterDetector* _detector,
      const FrameworkInfo& framework,
      bool implicitAcknowledgements,
      const Credential& credential)
    : MesosSchedulerDriver(
          scheduler,
          framework,
          "",
          implicitAcknowledgements,
          credential)
  {
    // No-op destructor as _detector lives on the stack.
    detector =
      std::shared_ptr<MasterDetector>(_detector, [](MasterDetector*) {});
  }
};

namespace scheduler {

// A generic mock HTTP scheduler to be used in tests with gmock.
template <typename Mesos, typename Event>
class MockHTTPScheduler
{
public:
  MOCK_METHOD1_T(connected, void(Mesos*));
  MOCK_METHOD1_T(disconnected, void(Mesos*));
  MOCK_METHOD1_T(heartbeat, void(Mesos*));
  MOCK_METHOD2_T(subscribed, void(Mesos*, const typename Event::Subscribed&));
  MOCK_METHOD2_T(offers, void(Mesos*, const typename Event::Offers&));
  MOCK_METHOD2_T(rescind, void(Mesos*, const typename Event::Rescind&));
  MOCK_METHOD2_T(update, void(Mesos*, const typename Event::Update&));
  MOCK_METHOD2_T(message, void(Mesos*, const typename Event::Message&));
  MOCK_METHOD2_T(failure, void(Mesos*, const typename Event::Failure&));
  MOCK_METHOD2_T(error, void(Mesos*, const typename Event::Error&));

  void event(Mesos* mesos, const Event& event)
  {
    switch(event.type()) {
      case Event::SUBSCRIBED:
        subscribed(mesos, event.subscribed());
        break;
      case Event::OFFERS:
        offers(mesos, event.offers());
        break;
      case Event::RESCIND:
        rescind(mesos, event.rescind());
        break;
      case Event::UPDATE:
        update(mesos, event.update());
        break;
      case Event::MESSAGE:
        message(mesos, event.message());
        break;
      case Event::FAILURE:
        failure(mesos, event.failure());
        break;
      case Event::ERROR:
        error(mesos, event.error());
        break;
      case Event::HEARTBEAT:
        heartbeat(mesos);
        break;
    }
  }
};


// A generic testing interface for the scheduler library that can be used to
// test the library across various versions.
template <typename Mesos, typename Event>
class TestMesos : public Mesos
{
public:
  TestMesos(
      const std::string& master,
      ContentType contentType,
      const std::shared_ptr<MockHTTPScheduler<Mesos, Event>>& _scheduler,
      const Option<std::shared_ptr<MasterDetector>>& detector = None())
    : Mesos(
        master,
        contentType,
        // We don't pass the `_scheduler` shared pointer as the library
        // interface expects a `std::function` object.
        lambda::bind(&MockHTTPScheduler<Mesos, Event>::connected,
                     _scheduler.get(),
                     this),
        lambda::bind(&MockHTTPScheduler<Mesos, Event>::disconnected,
                     _scheduler.get(),
                     this),
        lambda::bind(&TestMesos<Mesos, Event>::events,
                     this,
                     lambda::_1),
        detector),
      scheduler(_scheduler) {}

  virtual ~TestMesos()
  {
    // Since the destructor for `TestMesos` is invoked first, the library can
    // make more callbacks to the `scheduler` object before the `Mesos` (base
    // class) destructor is invoked. To prevent this, we invoke `stop()` here
    // to explicitly stop the library.
    this->stop();

    bool paused = process::Clock::paused();

    // Need to settle the Clock to ensure that all the pending async callbacks
    // with references to `this` and `scheduler` queued on libprocess are
    // executed before the object is destructed.
    process::Clock::pause();
    process::Clock::settle();

    // Return the Clock to its original state.
    if (!paused) {
      process::Clock::resume();
    }
  }

protected:
  void events(std::queue<Event> events)
  {
    while(!events.empty()) {
      Event event = std::move(events.front());
      events.pop();
      scheduler->event(this, event);
    }
  }

private:
  std::shared_ptr<MockHTTPScheduler<Mesos, Event>> scheduler;
};


using TestV1Mesos =
  TestMesos<mesos::v1::scheduler::Mesos, mesos::v1::scheduler::Event>;

} // namespace scheduler {


using MockV1HTTPScheduler =
  scheduler::MockHTTPScheduler<
    mesos::v1::scheduler::Mesos, mesos::v1::scheduler::Event>;


namespace executor {

// A generic mock HTTP executor to be used in tests with gmock.
template <typename Mesos, typename Event>
class MockHTTPExecutor
{
public:
  MOCK_METHOD1_T(connected, void(Mesos*));
  MOCK_METHOD1_T(disconnected, void(Mesos*));
  MOCK_METHOD2_T(subscribed, void(Mesos*, const typename Event::Subscribed&));
  MOCK_METHOD2_T(launch, void(Mesos*, const typename Event::Launch&));
  MOCK_METHOD2_T(kill, void(Mesos*, const typename Event::Kill&));
  MOCK_METHOD2_T(message, void(Mesos*, const typename Event::Message&));
  MOCK_METHOD1_T(shutdown, void(Mesos*));
  MOCK_METHOD2_T(error, void(Mesos*, const typename Event::Error&));
  MOCK_METHOD2_T(acknowledged,
                 void(Mesos*, const typename Event::Acknowledged&));

  void event(Mesos* mesos, const Event& event)
  {
    switch(event.type()) {
      case Event::SUBSCRIBED:
        subscribed(mesos, event.subscribed());
        break;
      case Event::LAUNCH:
        launch(mesos, event.launch());
        break;
      case Event::KILL:
        kill(mesos, event.kill());
        break;
      case Event::ACKNOWLEDGED:
        acknowledged(mesos, event.acknowledged());
        break;
      case Event::MESSAGE:
        message(mesos, event.message());
        break;
      case Event::SHUTDOWN:
        shutdown(mesos);
        break;
      case Event::ERROR:
        error(mesos, event.error());
        break;
    }
  }
};


// A generic testing interface for the executor library that can be used to
// test the library across various versions.
template <typename Mesos, typename Event>
class TestMesos : public Mesos
{
public:
  TestMesos(
      ContentType contentType,
      const std::shared_ptr<MockHTTPExecutor<Mesos, Event>>& _executor)
    : Mesos(
        contentType,
        lambda::bind(&MockHTTPExecutor<Mesos, Event>::connected,
                     _executor,
                     this),
        lambda::bind(&MockHTTPExecutor<Mesos, Event>::disconnected,
                     _executor,
                     this),
        lambda::bind(&TestMesos<Mesos, Event>::events,
                     this,
                     lambda::_1)),
      executor(_executor) {}

protected:
  void events(std::queue<Event> events)
  {
    while(!events.empty()) {
      Event event = std::move(events.front());
      events.pop();
      executor->event(this, event);
    }
  }

private:
  std::shared_ptr<MockHTTPExecutor<Mesos, Event>> executor;
};


using TestV1Mesos =
  TestMesos<mesos::v1::executor::Mesos, mesos::v1::executor::Event>;


// TODO(anand): Move these actions to the `v1::executor` namespace.
ACTION_P2(SendSubscribe, frameworkId, executorId)
{
  v1::executor::Call call;
  call.mutable_framework_id()->CopyFrom(frameworkId);
  call.mutable_executor_id()->CopyFrom(executorId);

  call.set_type(v1::executor::Call::SUBSCRIBE);

  call.mutable_subscribe();

  arg0->send(call);
}


ACTION_P3(SendUpdateFromTask, frameworkId, executorId, state)
{
  v1::TaskStatus status;
  status.mutable_task_id()->CopyFrom(arg1.task().task_id());
  status.mutable_executor_id()->CopyFrom(executorId);
  status.set_state(state);
  status.set_source(v1::TaskStatus::SOURCE_EXECUTOR);
  status.set_uuid(UUID::random().toBytes());

  v1::executor::Call call;
  call.mutable_framework_id()->CopyFrom(frameworkId);
  call.mutable_executor_id()->CopyFrom(executorId);

  call.set_type(v1::executor::Call::UPDATE);

  call.mutable_update()->mutable_status()->CopyFrom(status);

  arg0->send(call);
}


ACTION_P3(SendUpdateFromTaskID, frameworkId, executorId, state)
{
  v1::TaskStatus status;
  status.mutable_task_id()->CopyFrom(arg1.task_id());
  status.mutable_executor_id()->CopyFrom(executorId);
  status.set_state(state);
  status.set_source(v1::TaskStatus::SOURCE_EXECUTOR);
  status.set_uuid(UUID::random().toBytes());

  v1::executor::Call call;
  call.mutable_framework_id()->CopyFrom(frameworkId);
  call.mutable_executor_id()->CopyFrom(executorId);

  call.set_type(v1::executor::Call::UPDATE);

  call.mutable_update()->mutable_status()->CopyFrom(status);

  arg0->send(call);
}

} // namespace executor {


using MockV1HTTPExecutor =
  executor::MockHTTPExecutor<
    mesos::v1::executor::Mesos, mesos::v1::executor::Event>;


class MockGarbageCollector : public slave::GarbageCollector
{
public:
  MockGarbageCollector();
  virtual ~MockGarbageCollector();

  MOCK_METHOD2(
      schedule,
      process::Future<Nothing>(const Duration& d, const std::string& path));
  MOCK_METHOD1(
      unschedule,
      process::Future<bool>(const std::string& path));
  MOCK_METHOD1(
      prune,
      void(const Duration& d));
};


class MockResourceEstimator : public mesos::slave::ResourceEstimator
{
public:
  MockResourceEstimator();
  virtual ~MockResourceEstimator();

  MOCK_METHOD1(
      initialize,
      Try<Nothing>(const lambda::function<process::Future<ResourceUsage>()>&));

  MOCK_METHOD0(
      oversubscribable,
      process::Future<Resources>());
};


// The MockQoSController is a stub which lets tests fill the
// correction queue for a slave.
class MockQoSController : public mesos::slave::QoSController
{
public:
  MockQoSController();
  virtual ~MockQoSController();

  MOCK_METHOD1(
      initialize,
      Try<Nothing>(const lambda::function<process::Future<ResourceUsage>()>&));

  MOCK_METHOD0(
      corrections, process::Future<std::list<mesos::slave::QoSCorrection>>());
};


// Definition of a mock Slave to be used in tests with gmock, covering
// potential races between runTask and killTask.
class MockSlave : public slave::Slave
{
public:
  MockSlave(
      const slave::Flags& flags,
      MasterDetector* detector,
      slave::Containerizer* containerizer,
      const Option<mesos::slave::QoSController*>& qosController = None());

  virtual ~MockSlave();

  MOCK_METHOD5(runTask, void(
      const process::UPID& from,
      const FrameworkInfo& frameworkInfo,
      const FrameworkID& frameworkId,
      const process::UPID& pid,
      TaskInfo task));

  void unmocked_runTask(
      const process::UPID& from,
      const FrameworkInfo& frameworkInfo,
      const FrameworkID& frameworkId,
      const process::UPID& pid,
      TaskInfo task);

  MOCK_METHOD3(_runTask, void(
      const process::Future<bool>& future,
      const FrameworkInfo& frameworkInfo,
      const TaskInfo& task));

  void unmocked__runTask(
      const process::Future<bool>& future,
      const FrameworkInfo& frameworkInfo,
      const TaskInfo& task);

  MOCK_METHOD3(killTask, void(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const TaskID& taskId));

  void unmocked_killTask(
      const process::UPID& from,
      const FrameworkID& frameworkId,
      const TaskID& taskId);

  MOCK_METHOD1(removeFramework, void(
      slave::Framework* framework));

  void unmocked_removeFramework(
      slave::Framework* framework);

  MOCK_METHOD1(__recover, void(
      const process::Future<Nothing>& future));

  void unmocked___recover(
      const process::Future<Nothing>& future);

  MOCK_METHOD0(qosCorrections, void());

  void unmocked_qosCorrections();

  MOCK_METHOD1(_qosCorrections, void(
      const process::Future<std::list<
          mesos::slave::QoSCorrection>>& correction));

private:
  Files files;
  MockGarbageCollector gc;
  MockResourceEstimator resourceEstimator;
  MockQoSController qosController;
  slave::StatusUpdateManager* statusUpdateManager;
};


// Definition of a mock FetcherProcess to be used in tests with gmock.
class MockFetcherProcess : public slave::FetcherProcess
{
public:
  MockFetcherProcess();
  virtual ~MockFetcherProcess();

  MOCK_METHOD6(_fetch, process::Future<Nothing>(
      const hashmap<
          CommandInfo::URI,
          Option<process::Future<std::shared_ptr<Cache::Entry>>>>&
        entries,
      const ContainerID& containerId,
      const std::string& sandboxDirectory,
      const std::string& cacheDirectory,
      const Option<std::string>& user,
      const slave::Flags& flags));

  process::Future<Nothing> unmocked__fetch(
      const hashmap<
          CommandInfo::URI,
          Option<process::Future<std::shared_ptr<Cache::Entry>>>>&
        entries,
      const ContainerID& containerId,
      const std::string& sandboxDirectory,
      const std::string& cacheDirectory,
      const Option<std::string>& user,
      const slave::Flags& flags);

  MOCK_METHOD5(run, process::Future<Nothing>(
      const ContainerID& containerId,
      const std::string& sandboxDirectory,
      const Option<std::string>& user,
      const mesos::fetcher::FetcherInfo& info,
      const slave::Flags& flags));

  process::Future<Nothing> unmocked_run(
      const ContainerID& containerId,
      const std::string& sandboxDirectory,
      const Option<std::string>& user,
      const mesos::fetcher::FetcherInfo& info,
      const slave::Flags& flags);
};


// Definition of a mock ContainerLogger to be used in tests with gmock.
class MockContainerLogger : public mesos::slave::ContainerLogger
{
public:
  MockContainerLogger();
  virtual ~MockContainerLogger();

  MOCK_METHOD0(initialize, Try<Nothing>(void));

  MOCK_METHOD2(
      recover,
      process::Future<Nothing>(const ExecutorInfo&, const std::string&));

  MOCK_METHOD2(
      prepare,
      process::Future<mesos::slave::ContainerLogger::SubprocessInfo>(
          const ExecutorInfo&, const std::string&));
};


// Definition of a mock Docker to be used in tests with gmock.
class MockDocker : public Docker
{
public:
  MockDocker(
      const std::string& path,
      const std::string& socket);
  virtual ~MockDocker();

  MOCK_CONST_METHOD9(
      run,
      process::Future<Nothing>(
          const mesos::ContainerInfo&,
          const mesos::CommandInfo&,
          const std::string&,
          const std::string&,
          const std::string&,
          const Option<mesos::Resources>&,
          const Option<std::map<std::string, std::string>>&,
          const process::Subprocess::IO&,
          const process::Subprocess::IO&));

  MOCK_CONST_METHOD2(
      ps,
      process::Future<std::list<Docker::Container>>(
          bool, const Option<std::string>&));

  MOCK_CONST_METHOD3(
      pull,
      process::Future<Docker::Image>(
          const std::string&,
          const std::string&,
          bool));

  MOCK_CONST_METHOD3(
      stop,
      process::Future<Nothing>(
          const std::string&,
          const Duration&,
          bool));

  MOCK_CONST_METHOD2(
      inspect,
      process::Future<Docker::Container>(
          const std::string&,
          const Option<Duration>&));

  process::Future<Nothing> _run(
      const mesos::ContainerInfo& containerInfo,
      const mesos::CommandInfo& commandInfo,
      const std::string& name,
      const std::string& sandboxDirectory,
      const std::string& mappedDirectory,
      const Option<mesos::Resources>& resources,
      const Option<std::map<std::string, std::string>>& env,
      const process::Subprocess::IO& stdout,
      const process::Subprocess::IO& stderr) const
  {
    return Docker::run(
        containerInfo,
        commandInfo,
        name,
        sandboxDirectory,
        mappedDirectory,
        resources,
        env,
        stdout,
        stderr);
  }

  process::Future<std::list<Docker::Container>> _ps(
      bool all,
      const Option<std::string>& prefix) const
  {
    return Docker::ps(all, prefix);
  }

  process::Future<Docker::Image> _pull(
      const std::string& directory,
      const std::string& image,
      bool force) const
  {
    return Docker::pull(directory, image, force);
  }

  process::Future<Nothing> _stop(
      const std::string& containerName,
      const Duration& timeout,
      bool remove) const
  {
    return Docker::stop(containerName, timeout, remove);
  }

  process::Future<Docker::Container> _inspect(
      const std::string& containerName,
      const Option<Duration>& retryInterval)
  {
    return Docker::inspect(containerName, retryInterval);
  }
};


// Definition of a mock DockerContainerizer to be used in tests with gmock.
class MockDockerContainerizer : public slave::DockerContainerizer {
public:
  MockDockerContainerizer(
      const slave::Flags& flags,
      slave::Fetcher* fetcher,
      const process::Owned<mesos::slave::ContainerLogger>& logger,
      process::Shared<Docker> docker);

  MockDockerContainerizer(
      const process::Owned<slave::DockerContainerizerProcess>& process);

  virtual ~MockDockerContainerizer();

  void initialize()
  {
    // NOTE: See TestContainerizer::setup for why we use
    // 'EXPECT_CALL' and 'WillRepeatedly' here instead of
    // 'ON_CALL' and 'WillByDefault'.
    EXPECT_CALL(*this, launch(_, _, _, _, _, _, _))
      .WillRepeatedly(Invoke(this, &MockDockerContainerizer::_launchExecutor));

    EXPECT_CALL(*this, launch(_, _, _, _, _, _, _, _))
      .WillRepeatedly(Invoke(this, &MockDockerContainerizer::_launch));

    EXPECT_CALL(*this, update(_, _))
      .WillRepeatedly(Invoke(this, &MockDockerContainerizer::_update));
  }

  MOCK_METHOD7(
      launch,
      process::Future<bool>(
          const ContainerID&,
          const ExecutorInfo&,
          const std::string&,
          const Option<std::string>&,
          const SlaveID&,
          const process::PID<slave::Slave>&,
          bool checkpoint));

  MOCK_METHOD8(
      launch,
      process::Future<bool>(
          const ContainerID&,
          const TaskInfo&,
          const ExecutorInfo&,
          const std::string&,
          const Option<std::string>&,
          const SlaveID&,
          const process::PID<slave::Slave>&,
          bool checkpoint));

  MOCK_METHOD2(
      update,
      process::Future<Nothing>(
          const ContainerID&,
          const Resources&));

  // Default 'launch' implementation (necessary because we can't just
  // use &slave::DockerContainerizer::launch with 'Invoke').
  process::Future<bool> _launch(
      const ContainerID& containerId,
      const TaskInfo& taskInfo,
      const ExecutorInfo& executorInfo,
      const std::string& directory,
      const Option<std::string>& user,
      const SlaveID& slaveId,
      const slave::PID<slave::Slave>& slavePid,
      bool checkpoint)
  {
    return slave::DockerContainerizer::launch(
        containerId,
        taskInfo,
        executorInfo,
        directory,
        user,
        slaveId,
        slavePid,
        checkpoint);
  }

  process::Future<bool> _launchExecutor(
      const ContainerID& containerId,
      const ExecutorInfo& executorInfo,
      const std::string& directory,
      const Option<std::string>& user,
      const SlaveID& slaveId,
      const slave::PID<slave::Slave>& slavePid,
      bool checkpoint)
  {
    return slave::DockerContainerizer::launch(
        containerId,
        executorInfo,
        directory,
        user,
        slaveId,
        slavePid,
        checkpoint);
  }

  process::Future<Nothing> _update(
      const ContainerID& containerId,
      const Resources& resources)
  {
    return slave::DockerContainerizer::update(
        containerId,
        resources);
  }
};


// Definition of a mock DockerContainerizerProcess to be used in tests
// with gmock.
class MockDockerContainerizerProcess : public slave::DockerContainerizerProcess
{
public:
  MockDockerContainerizerProcess(
      const slave::Flags& flags,
      slave::Fetcher* fetcher,
      const process::Owned<mesos::slave::ContainerLogger>& logger,
      const process::Shared<Docker>& docker);

  virtual ~MockDockerContainerizerProcess();

  MOCK_METHOD2(
      fetch,
      process::Future<Nothing>(
          const ContainerID& containerId,
          const SlaveID& slaveId));

  MOCK_METHOD1(
      pull,
      process::Future<Nothing>(const ContainerID& containerId));

  process::Future<Nothing> _fetch(
      const ContainerID& containerId,
      const SlaveID& slaveId)
  {
    return slave::DockerContainerizerProcess::fetch(containerId, slaveId);
  }

  process::Future<Nothing> _pull(const ContainerID& containerId)
  {
    return slave::DockerContainerizerProcess::pull(containerId);
  }
};


// Definition of a MockAuthorizer that can be used in tests with gmock.
class MockAuthorizer : public Authorizer
{
public:
  MockAuthorizer();
  virtual ~MockAuthorizer();

  MOCK_METHOD1(
      initialize, Try<Nothing>(const Option<ACLs>& acls));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::RegisterFramework& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::RunTask& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::ShutdownFramework& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::ReserveResources& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::UnreserveResources& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::CreateVolume& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::DestroyVolume& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::SetQuota& request));
  MOCK_METHOD1(
      authorize, process::Future<bool>(const ACL::RemoveQuota& request));
};


class OfferEqMatcher
  : public ::testing::MatcherInterface<const std::vector<Offer>& >
{
public:
  OfferEqMatcher(int _cpus, int _mem)
    : cpus(_cpus), mem(_mem) {}

  virtual bool MatchAndExplain(const std::vector<Offer>& offers,
                               ::testing::MatchResultListener* listener) const
  {
    double totalCpus = 0;
    double totalMem = 0;

    foreach (const Offer& offer, offers) {
      foreach (const Resource& resource, offer.resources()) {
        if (resource.name() == "cpus") {
          totalCpus += resource.scalar().value();
        } else if (resource.name() == "mem") {
          totalMem += resource.scalar().value();
        }
      }
    }

    bool matches = totalCpus == cpus && totalMem == mem;

    if (!matches) {
      *listener << totalCpus << " cpus and " << totalMem << "mem";
    }

    return matches;
  }

  virtual void DescribeTo(::std::ostream* os) const
  {
    *os << "contains " << cpus << " cpus and " << mem << " mem";
  }

  virtual void DescribeNegationTo(::std::ostream* os) const
  {
    *os << "does not contain " << cpus << " cpus and "  << mem << " mem";
  }

private:
  int cpus;
  int mem;
};


inline
const ::testing::Matcher<const std::vector<Offer>& > OfferEq(int cpus, int mem)
{
  return MakeMatcher(new OfferEqMatcher(cpus, mem));
}


ACTION_P(SendStatusUpdateFromTask, state)
{
  TaskStatus status;
  status.mutable_task_id()->MergeFrom(arg1.task_id());
  status.set_state(state);
  arg0->sendStatusUpdate(status);
}


ACTION_P(SendStatusUpdateFromTaskID, state)
{
  TaskStatus status;
  status.mutable_task_id()->MergeFrom(arg1);
  status.set_state(state);
  arg0->sendStatusUpdate(status);
}


ACTION_P(SendFrameworkMessage, data)
{
  arg0->sendFrameworkMessage(data);
}


#define FUTURE_PROTOBUF(message, from, to)              \
  FutureProtobuf(message, from, to)


#define DROP_PROTOBUF(message, from, to)              \
  FutureProtobuf(message, from, to, true)


#define DROP_PROTOBUFS(message, from, to)              \
  DropProtobufs(message, from, to)


#define EXPECT_NO_FUTURE_PROTOBUFS(message, from, to)              \
  ExpectNoFutureProtobufs(message, from, to)


#define FUTURE_HTTP_PROTOBUF(message, path, contentType)   \
  FutureHttp(message, path, contentType)


#define DROP_HTTP_PROTOBUF(message, path, contentType)     \
  FutureHttp(message, path, contentType, true)


#define DROP_HTTP_PROTOBUFS(message, path, contentType)    \
  DropHttpProtobufs(message, path, contentType)


#define EXPECT_NO_FUTURE_HTTP_PROTOBUFS(message, path, contentType)  \
  ExpectNoFutureHttpProtobufs(message, path, contentType)


// These are specialized versions of {FUTURE,DROP}_PROTOBUF that
// capture a scheduler/executor Call protobuf of the given 'type'.
// Note that we name methods as '*ProtobufUnion()' because these could
// be reused for macros that capture any protobufs that are described
// using the standard protocol buffer "union" trick (e.g.,
// FUTURE_EVENT to capture scheduler::Event), see
// https://developers.google.com/protocol-buffers/docs/techniques#union.

#define FUTURE_CALL(message, unionType, from, to)              \
  FutureUnionProtobuf(message, unionType, from, to)


#define DROP_CALL(message, unionType, from, to)                \
  FutureUnionProtobuf(message, unionType, from, to, true)


#define DROP_CALLS(message, unionType, from, to)               \
  DropUnionProtobufs(message, unionType, from, to)


#define EXPECT_NO_FUTURE_CALLS(message, unionType, from, to)   \
  ExpectNoFutureUnionProtobufs(message, unionType, from, to)


#define FUTURE_CALL_MESSAGE(message, unionType, from, to)          \
  process::FutureUnionMessage(message, unionType, from, to)


#define DROP_CALL_MESSAGE(message, unionType, from, to)            \
  process::FutureUnionMessage(message, unionType, from, to, true)


#define FUTURE_HTTP_CALL(message, unionType, path, contentType)  \
  FutureUnionHttp(message, unionType, path, contentType)


#define DROP_HTTP_CALL(message, unionType, path, contentType)    \
  FutureUnionHttp(message, unionType, path, contentType, true)


#define DROP_HTTP_CALLS(message, unionType, path, contentType)   \
  DropUnionHttpProtobufs(message, unionType, path, contentType)


#define EXPECT_NO_FUTURE_HTTP_CALLS(message, unionType, path, contentType)   \
  ExpectNoFutureUnionHttpProtobufs(message, unionType, path, contentType)


// Forward declaration.
template <typename T>
T _FutureProtobuf(const process::Message& message);


template <typename T, typename From, typename To>
process::Future<T> FutureProtobuf(T t, From from, To to, bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &t; (void) m; }

  return process::FutureMessage(testing::Eq(t.GetTypeName()), from, to, drop)
    .then(lambda::bind(&_FutureProtobuf<T>, lambda::_1));
}


template <typename Message, typename UnionType, typename From, typename To>
process::Future<Message> FutureUnionProtobuf(
    Message message, UnionType unionType, From from, To to, bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  return process::FutureUnionMessage(message, unionType, from, to, drop)
    .then(lambda::bind(&_FutureProtobuf<Message>, lambda::_1));
}


template <typename Message, typename Path>
process::Future<Message> FutureHttp(
    Message message,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  return process::FutureHttpRequest(message, path, deserializer, drop)
    .then([deserializer](const process::http::Request& request) {
      return deserializer(request.body).get();
    });
}


template <typename Message, typename UnionType, typename Path>
process::Future<Message> FutureUnionHttp(
    Message message,
    UnionType unionType,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  return process::FutureUnionHttpRequest(
      message, unionType, path, deserializer, drop)
    .then([deserializer](const process::http::Request& request) {
      return deserializer(request.body).get();
    });
}


template <typename T>
T _FutureProtobuf(const process::Message& message)
{
  T t;
  t.ParseFromString(message.body);
  return t;
}


template <typename T, typename From, typename To>
void DropProtobufs(T t, From from, To to)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &t; (void) m; }

  process::DropMessages(testing::Eq(t.GetTypeName()), from, to);
}


template <typename Message, typename UnionType, typename From, typename To>
void DropUnionProtobufs(Message message, UnionType unionType, From from, To to)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  process::DropUnionMessages(message, unionType, from, to);
}


template <typename Message, typename Path>
void DropHttpProtobufs(
    Message message,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  process::DropHttpRequests(message, path, deserializer);
}


template <typename Message, typename UnionType, typename Path>
void DropUnionHttpProtobufs(
    Message message,
    UnionType unionType,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  process::DropUnionHttpRequests(message, unionType, path, deserializer);
}


template <typename T, typename From, typename To>
void ExpectNoFutureProtobufs(T t, From from, To to)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &t; (void) m; }

  process::ExpectNoFutureMessages(testing::Eq(t.GetTypeName()), from, to);
}


template <typename Message, typename UnionType, typename From, typename To>
void ExpectNoFutureUnionProtobufs(
    Message message, UnionType unionType, From from, To to)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  process::ExpectNoFutureUnionMessages(message, unionType, from, to);
}


template <typename Message, typename Path>
void ExpectNoFutureHttpProtobufs(
    Message message,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  process::ExpectNoFutureHttpRequests(message, path, deserializer);
}


template <typename Message, typename UnionType, typename Path>
void ExpectNoFutureUnionHttpProtobufs(
    Message message,
    UnionType unionType,
    Path path,
    ContentType contentType,
    bool drop = false)
{
  // Help debugging by adding some "type constraints".
  { google::protobuf::Message* m = &message; (void) m; }

  auto deserializer =
    lambda::bind(&deserialize<Message>, contentType, lambda::_1);

  process::ExpectNoFutureUnionHttpRequests(
      message, unionType, path, deserializer);
}


// This matcher is used to match the task ids of TaskStatus messages.
// Suppose we set up N futures for LaunchTasks and N futures for StatusUpdates.
// (This is a common pattern). We get into a situation where all StatusUpdates
// are satisfied before the LaunchTasks if the master re-sends StatusUpdates.
// We use this matcher to only satisfy the StatusUpdate future if the
// StatusUpdate came from the corresponding task.
MATCHER_P(TaskStatusEq, task, "") { return arg.task_id() == task.task_id(); }

} // namespace tests {
} // namespace internal {
} // namespace mesos {

#endif // __TESTS_MESOS_HPP__
