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

#include <gmock/gmock.h>

#include <mesos/executor.hpp>
#include <mesos/scheduler.hpp>

#include <process/future.hpp>
#include <process/gmock.hpp>
#include <process/http.hpp>
#include <process/pid.hpp>

#include <stout/hashmap.hpp>
#include <stout/option.hpp>

#include "master/flags.hpp"
#include "master/master.hpp"

#include "tests/mesos.hpp"
#include "tests/utils.hpp"

using mesos::internal::master::Master;
using mesos::internal::slave::Slave;

using process::Future;
using process::PID;

using process::http::BadRequest;
using process::http::Forbidden;
using process::http::OK;
using process::http::Response;
using process::http::Unauthorized;

using testing::_;

namespace mesos {
namespace internal {
namespace tests {


class TeardownTest : public MesosTest {};


// Testing /master/teardown to validate that this endpoint shuts down
// the designated framework or returns an appropriate error.

// Testing route with authorization header and good credentials.
TEST_F(TeardownTest, TeardownEndpoint)
{
  Try<PID<Master>> master = StartMaster();
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(DEFAULT_CREDENTIAL),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route with bad credentials.
TEST_F(TeardownTest, TeardownEndpointBadCredentials)
{
  Try<PID<Master>> master = StartMaster();
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Credential badCredential;
  badCredential.set_principal("badPrincipal");
  badCredential.set_secret("badSecret");

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(badCredential),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(Unauthorized({}).status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route with good ACLs.
TEST_F(TeardownTest, TeardownEndpointGoodACLs)
{
  // Setup ACLs so that the default principal can teardown the
  // framework.
  ACLs acls;
  mesos::ACL::TeardownFramework* acl = acls.add_teardown_frameworks();
  acl->mutable_principals()->add_values(DEFAULT_CREDENTIAL.principal());
  acl->mutable_framework_principals()->add_values(
      DEFAULT_CREDENTIAL.principal());

  master::Flags flags = CreateMasterFlags();
  flags.acls = acls;
  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(DEFAULT_CREDENTIAL),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route with deprecated (but still good) ACLs.
// This ACL/test will be removed at the end of the deprecation cycle on 0.27.
TEST_F(TeardownTest, TeardownEndpointGoodDeprecatedACLs)
{
  // Setup ACLs so that the default principal can teardown the
  // framework.
  ACLs acls;
  mesos::ACL::ShutdownFramework* acl = acls.add_shutdown_frameworks();
  acl->mutable_principals()->add_values(DEFAULT_CREDENTIAL.principal());
  acl->mutable_framework_principals()->add_values(
      DEFAULT_CREDENTIAL.principal());

  master::Flags flags = CreateMasterFlags();
  flags.acls = acls;
  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(DEFAULT_CREDENTIAL),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route with bad ACLs.
TEST_F(TeardownTest, TeardownEndpointBadACLs)
{
  // Setup ACLs so that no principal can do teardown the framework.
  ACLs acls;
  mesos::ACL::ShutdownFramework* acl = acls.add_shutdown_frameworks();
  acl->mutable_principals()->set_type(mesos::ACL::Entity::NONE);
  acl->mutable_framework_principals()->add_values(
      DEFAULT_CREDENTIAL.principal());

  master::Flags flags = CreateMasterFlags();
  flags.acls = acls;
  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(DEFAULT_CREDENTIAL),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(Forbidden().status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route without frameworkId value.
TEST_F(TeardownTest, TeardownEndpointNoFrameworkId)
{
  Try<PID<Master>> master = StartMaster();
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      createBasicAuthHeaders(DEFAULT_CREDENTIAL),
      "");

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);

  driver.stop();
  driver.join();

  Shutdown();
}


// Testing route without authorization header.
TEST_F(TeardownTest, TeardownEndpointNoHeader)
{
  Try<PID<Master>> master = StartMaster();
  ASSERT_SOME(master);

  MockScheduler sched;
  MesosSchedulerDriver driver(
      &sched, DEFAULT_FRAMEWORK_INFO, master.get(), DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  ASSERT_EQ(DRIVER_RUNNING, driver.start());

  AWAIT_READY(frameworkId);

  Future<Response> response = process::http::post(
      master.get(),
      "teardown",
      None(),
      "frameworkId=" + frameworkId.get().value());

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(Unauthorized({}).status, response);

  driver.stop();
  driver.join();

  Shutdown();
}

} // namespace tests {
} // namespace internal {
} // namespace mesos {
