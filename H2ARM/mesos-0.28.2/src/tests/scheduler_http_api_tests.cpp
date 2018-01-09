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

#include <mesos/v1/mesos.hpp>
#include <mesos/v1/scheduler.hpp>

#include <process/clock.hpp>
#include <process/future.hpp>
#include <process/gtest.hpp>
#include <process/http.hpp>
#include <process/pid.hpp>

#include <stout/gtest.hpp>
#include <stout/json.hpp>
#include <stout/lambda.hpp>
#include <stout/recordio.hpp>
#include <stout/uuid.hpp>

#include "common/http.hpp"
#include "common/recordio.hpp"

#include "master/constants.hpp"
#include "master/master.hpp"

#include "tests/mesos.hpp"
#include "tests/utils.hpp"

using mesos::internal::master::DEFAULT_HEARTBEAT_INTERVAL;
using mesos::internal::master::Master;

using mesos::internal::recordio::Reader;

using mesos::v1::scheduler::Call;
using mesos::v1::scheduler::Event;

using process::Clock;
using process::Future;
using process::PID;

using process::http::BadRequest;
using process::http::Forbidden;
using process::http::MethodNotAllowed;
using process::http::NotAcceptable;
using process::http::OK;
using process::http::Pipe;
using process::http::Response;
using process::http::UnsupportedMediaType;

using recordio::Decoder;

using std::string;

using testing::WithParamInterface;

namespace mesos {
namespace internal {
namespace tests {


class SchedulerHttpApiTest
  : public MesosTest,
    public WithParamInterface<string>
{
public:
  // TODO(anand): Use the serialize/deserialize from common/http.hpp
  // when they are available.
  Try<Event> deserialize(const string& contentType, const string& body)
  {
    if (contentType == APPLICATION_PROTOBUF) {
      Event event;
      if (!event.ParseFromString(body)) {
        return Error("Failed to parse body into Event protobuf");
      }
      return event;
    }

    Try<JSON::Value> value = JSON::parse(body);
    Try<Event> parse = ::protobuf::parse<Event>(value.get());
    return parse;
  }

  string serialize(const Call& call, const string& contentType)
  {
    if (contentType == APPLICATION_PROTOBUF) {
      return call.SerializeAsString();
    }

    return stringify(JSON::protobuf(call));
  }
};


// The HttpApi tests are parameterized by the content type.
INSTANTIATE_TEST_CASE_P(
    ContentType,
    SchedulerHttpApiTest,
    ::testing::Values(APPLICATION_PROTOBUF, APPLICATION_JSON));


// TODO(anand): Add tests for:
// - A subscribed scheduler closes it's reader and then tries to
//  subscribe again before the framework failover timeout and should
//  succeed.
//
// - A subscribed PID scheduler disconnects and then tries to
//  subscribe again as a HTTP framework before the framework failover
//  timeout and should succeed.


TEST_F(SchedulerHttpApiTest, AuthenticationRequired)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = true;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Future<Response> response = process::http::post(
      master.get(),
      "api/v1/scheduler",
      None(),
      None());

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(Forbidden().status, response);
}


// TODO(anand): Add additional tests for validation.
TEST_F(SchedulerHttpApiTest, NoContentType)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  // Expect a BadRequest when 'Content-Type' is omitted.
  //
  // TODO(anand): Send a valid call here to ensure that
  // the BadRequest is only due to the missing header.
  Future<Response> response = process::http::post(
      master.get(),
      "api/v1/scheduler",
      None(),
      None());

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);
}


// This test sends a valid JSON blob that cannot be deserialized
// into a valid protobuf resulting in a BadRequest.
TEST_F(SchedulerHttpApiTest, ValidJsonButInvalidProtobuf)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  JSON::Object object;
  object.values["string"] = "valid_json";

  process::http::Headers headers;
  headers["Accept"] = APPLICATION_JSON;

  Future<Response> response = process::http::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      stringify(object),
      APPLICATION_JSON);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);
}


// This test sends a malformed body that cannot be deserialized
// into a valid protobuf resulting in a BadRequest.
TEST_P(SchedulerHttpApiTest, MalformedContent)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  const string body = "MALFORMED_CONTENT";

  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  Future<Response> response = process::http::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      body,
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);
}


// This test sets an unsupported media type as Content-Type. This
// should result in a 415 (UnsupportedMediaType) response.
TEST_P(SchedulerHttpApiTest, UnsupportedContentMediaType)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  const string unknownMediaType = "application/unknown-media-type";

  Future<Response> response = process::http::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      unknownMediaType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(UnsupportedMediaType().status, response);
}


// This test verifies if the scheduler is able to receive a Subscribed
// event and heartbeat events on the stream in response to a Subscribe
// call request.
TEST_P(SchedulerHttpApiTest, Subscribe)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
  AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
  ASSERT_EQ(Response::PIPE, response.get().type);
  ASSERT_TRUE(response.get().headers.contains("Mesos-Stream-Id"));
  EXPECT_NE("", response.get().headers.at("Mesos-Stream-Id"));

  Option<Pipe::Reader> reader = response.get().reader;
  ASSERT_SOME(reader);

  auto deserializer = lambda::bind(
      &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

  Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

  Future<Result<Event>> event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  // Check event type is subscribed and the framework id is set.
  ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
  EXPECT_NE("", event.get().get().subscribed().framework_id().value());

  // Make sure it receives a heartbeat.
  event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  ASSERT_EQ(Event::HEARTBEAT, event.get().get().type());

  // Advance the clock to receive another heartbeat.
  Clock::pause();
  Clock::advance(DEFAULT_HEARTBEAT_INTERVAL);

  event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  ASSERT_EQ(Event::HEARTBEAT, event.get().get().type());

  Shutdown();
}


// This test verifies that the client will receive a `BadRequest` response if it
// includes a stream ID header with a subscribe call.
TEST_P(SchedulerHttpApiTest, SubscribeWithStreamId)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;
  headers["Mesos-Stream-Id"] = UUID::random().toString();

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);

  Shutdown();
}


// This test verifies if the scheduler can subscribe on retrying,
// e.g. after a ZK blip.
TEST_P(SchedulerHttpApiTest, SubscribedOnRetry)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  auto deserializer = lambda::bind(
      &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

  v1::FrameworkID frameworkId;

  {
    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
    ASSERT_EQ(Response::PIPE, response.get().type);

    Option<Pipe::Reader> reader = response.get().reader;
    ASSERT_SOME(reader);

    Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

    Future<Result<Event>> event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    frameworkId = event.get().get().subscribed().framework_id();

    // Check event type is subscribed and the framework id is set.
    ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
    EXPECT_NE("", event.get().get().subscribed().framework_id().value());
  }

  {
    call.mutable_framework_id()->CopyFrom(frameworkId);
    subscribe->mutable_framework_info()->mutable_id()->CopyFrom(frameworkId);

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    Option<Pipe::Reader> reader = response.get().reader;
    ASSERT_SOME(reader);

    Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

    // Check if we were successfully able to subscribe after the blip.
    Future<Result<Event>> event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    // Check event type is subscribed and the same framework id is set.
    ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
    EXPECT_EQ(frameworkId, event.get().get().subscribed().framework_id());

    // Make sure it receives a heartbeat.
    event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    ASSERT_EQ(Event::HEARTBEAT, event.get().get().type());
  }

  Shutdown();
}


// This test verifies if we are able to upgrade from a PID based
// scheduler to HTTP scheduler.
TEST_P(SchedulerHttpApiTest, UpdatePidToHttpScheduler)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  v1::FrameworkInfo frameworkInfo = DEFAULT_V1_FRAMEWORK_INFO;
  frameworkInfo.set_failover_timeout(Weeks(2).secs());

  // Start the scheduler without credentials.
  MockScheduler sched;
  StandaloneMasterDetector detector(master.get());
  TestingMesosSchedulerDriver driver(&sched, &detector, devolve(frameworkInfo));

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  // Check that driver is notified with an error when the http
  // framework is connected.
  Future<FrameworkErrorMessage> errorMessage =
    FUTURE_PROTOBUF(FrameworkErrorMessage(), _, _);

  EXPECT_CALL(sched, error(_, _));

  driver.start();

  AWAIT_READY(frameworkId);
  EXPECT_NE("", frameworkId.get().value());

  // Now try to subscribe as an HTTP framework.
  Call call;
  call.set_type(Call::SUBSCRIBE);
  call.mutable_framework_id()->CopyFrom(evolve(frameworkId.get()));

  Call::Subscribe* subscribe = call.mutable_subscribe();

  subscribe->mutable_framework_info()->CopyFrom(frameworkInfo);
  subscribe->mutable_framework_info()->mutable_id()->
    CopyFrom(evolve(frameworkId.get()));

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
  AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
  ASSERT_EQ(Response::PIPE, response.get().type);

  Option<Pipe::Reader> reader = response.get().reader;
  ASSERT_SOME(reader);

  auto deserializer = lambda::bind(
      &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

  Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

  Future<Result<Event>> event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  // Check event type is subscribed and the framework id is set.
  ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
  EXPECT_EQ(evolve(frameworkId.get()),
            event.get().get().subscribed().framework_id());

  // Make sure it receives a heartbeat.
  event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  ASSERT_EQ(Event::HEARTBEAT, event.get().get().type());

  driver.stop();
  driver.join();

  Shutdown();
}


// This test verifies that we are able to downgrade from a HTTP based
// framework to PID.
TEST_P(SchedulerHttpApiTest, UpdateHttpToPidScheduler)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  v1::FrameworkInfo frameworkInfo = DEFAULT_V1_FRAMEWORK_INFO;

  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(frameworkInfo);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
  AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
  ASSERT_EQ(Response::PIPE, response.get().type);

  Option<Pipe::Reader> reader = response.get().reader;
  ASSERT_SOME(reader);

  auto deserializer = lambda::bind(
      &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

  Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

  Future<Result<Event>> event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  // Check event type is subscribed and the framework id is set.
  ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
  frameworkInfo.mutable_id()->
    CopyFrom(event.get().get().subscribed().framework_id());

  // Make sure it receives a heartbeat.
  event = responseDecoder.read();
  AWAIT_READY(event);
  ASSERT_SOME(event.get());

  ASSERT_EQ(Event::HEARTBEAT, event.get().get().type());

  // Start PID based scheduler without credentials.
  MockScheduler sched;
  MesosSchedulerDriver driver(&sched, devolve(frameworkInfo), master.get());

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  driver.start();

  AWAIT_READY(frameworkId);
  ASSERT_EQ(evolve(frameworkId.get()), frameworkInfo.id());

  driver.stop();
  driver.join();

  Shutdown();
}


TEST_P(SchedulerHttpApiTest, NotAcceptable)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master> > master = StartMaster(flags);
  ASSERT_SOME(master);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();

  process::http::Headers headers;
  headers["Accept"] = "foo";

  // Only subscribe needs to 'Accept' json or protobuf.
  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(NotAcceptable().status, response);
}


TEST_P(SchedulerHttpApiTest, NoAcceptHeader)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master> > master = StartMaster(flags);
  ASSERT_SOME(master);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();

  // No 'Accept' header leads to all media types considered
  // acceptable. JSON will be chosen by default.
  process::http::Headers headers;

  // Only subscribe needs to 'Accept' json or protobuf.
  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
  AWAIT_EXPECT_RESPONSE_HEADER_EQ(APPLICATION_JSON, "Content-Type", response);
}


TEST_P(SchedulerHttpApiTest, DefaultAccept)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master> > master = StartMaster(flags);
  ASSERT_SOME(master);

  process::http::Headers headers;
  headers["Accept"] = "*/*";

  // Only subscribe needs to 'Accept' json or protobuf.
  Call call;
  call.set_type(Call::SUBSCRIBE);

  Call::Subscribe* subscribe = call.mutable_subscribe();
  subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();

  Future<Response> response = process::http::streaming::post(
      master.get(),
      "api/v1/scheduler",
      headers,
      serialize(call, contentType),
      contentType);

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
  AWAIT_EXPECT_RESPONSE_HEADER_EQ(APPLICATION_JSON, "Content-Type", response);
}


TEST_F(SchedulerHttpApiTest, GetRequest)
{
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master> > master = StartMaster(flags);
  ASSERT_SOME(master);

  Future<Response> response = process::http::get(
      master.get(),
      "api/v1/scheduler");

  AWAIT_READY(response);
  AWAIT_EXPECT_RESPONSE_STATUS_EQ(MethodNotAllowed({"POST"}).status, response);
}


// This test verifies that the scheduler will receive a `BadRequest` response
// when a teardown call is made without including a stream ID header.
TEST_P(SchedulerHttpApiTest, TeardownWithoutStreamId)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  v1::FrameworkID frameworkId;

  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
    AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
    ASSERT_EQ(Response::PIPE, response.get().type);
    ASSERT_TRUE(response.get().headers.contains("Mesos-Stream-Id"));

    Option<Pipe::Reader> reader = response.get().reader;
    ASSERT_SOME(reader);

    auto deserializer = lambda::bind(
        &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

    Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

    Future<Result<Event>> event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    // Check event type is subscribed and the framework ID is set.
    ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
    EXPECT_NE("", event.get().get().subscribed().framework_id().value());

    frameworkId = event.get().get().subscribed().framework_id();
  }

  {
    // Send a TEARDOWN call without a stream ID.
    Call call;
    call.set_type(Call::TEARDOWN);
    call.mutable_framework_id()->CopyFrom(frameworkId);

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);
  }

  Shutdown();
}


// This test verifies that the scheduler will receive a `BadRequest` response
// when a teardown call is made with an incorrect stream ID header.
TEST_P(SchedulerHttpApiTest, TeardownWrongStreamId)
{
  // HTTP schedulers cannot yet authenticate.
  master::Flags flags = CreateMasterFlags();
  flags.authenticate_frameworks = false;

  Try<PID<Master>> master = StartMaster(flags);
  ASSERT_SOME(master);

  // Retrieve the parameter passed as content type to this test.
  const string contentType = GetParam();
  process::http::Headers headers;
  headers["Accept"] = contentType;

  v1::FrameworkID frameworkId;
  string streamId;

  // Subscribe once to get a valid stream ID.
  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
    AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
    ASSERT_EQ(Response::PIPE, response.get().type);
    ASSERT_TRUE(response.get().headers.contains("Mesos-Stream-Id"));

    streamId = response.get().headers.at("Mesos-Stream-Id");

    Option<Pipe::Reader> reader = response.get().reader;
    ASSERT_SOME(reader);

    auto deserializer = lambda::bind(
        &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

    Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

    Future<Result<Event>> event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    // Check that the event type is subscribed and the framework ID is set.
    ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
    EXPECT_NE("", event.get().get().subscribed().framework_id().value());

    frameworkId = event.get().get().subscribed().framework_id();
  }

  // Subscribe again to invalidate the first stream ID and acquire another one.
  {
    Call call;
    call.set_type(Call::SUBSCRIBE);

    Call::Subscribe* subscribe = call.mutable_subscribe();
    subscribe->mutable_framework_info()->CopyFrom(DEFAULT_V1_FRAMEWORK_INFO);

    // Set the framework ID in the subscribe call.
    call.mutable_framework_id()->CopyFrom(frameworkId);
    subscribe->mutable_framework_info()->mutable_id()->CopyFrom(frameworkId);

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(OK().status, response);
    AWAIT_EXPECT_RESPONSE_HEADER_EQ("chunked", "Transfer-Encoding", response);
    ASSERT_EQ(Response::PIPE, response.get().type);
    ASSERT_TRUE(response.get().headers.contains("Mesos-Stream-Id"));

    // Make sure that the new stream ID is different.
    ASSERT_NE(streamId, response.get().headers.at("Mesos-Stream-Id"));

    Option<Pipe::Reader> reader = response.get().reader;
    ASSERT_SOME(reader);

    auto deserializer = lambda::bind(
        &SchedulerHttpApiTest::deserialize, this, contentType, lambda::_1);

    Reader<Event> responseDecoder(Decoder<Event>(deserializer), reader.get());

    Future<Result<Event>> event = responseDecoder.read();
    AWAIT_READY(event);
    ASSERT_SOME(event.get());

    ASSERT_EQ(Event::SUBSCRIBED, event.get().get().type());
    EXPECT_NE("", event.get().get().subscribed().framework_id().value());
  }

  {
    Call call;
    call.set_type(Call::TEARDOWN);
    call.mutable_framework_id()->CopyFrom(frameworkId);

    // Send the first (now incorrect) stream ID with the teardown call.
    headers["Mesos-Stream-Id"] = streamId;

    Future<Response> response = process::http::streaming::post(
        master.get(),
        "api/v1/scheduler",
        headers,
        serialize(call, contentType),
        contentType);

    AWAIT_EXPECT_RESPONSE_STATUS_EQ(BadRequest().status, response);
  }

  Shutdown();
}

} // namespace tests {
} // namespace internal {
} // namespace mesos {
