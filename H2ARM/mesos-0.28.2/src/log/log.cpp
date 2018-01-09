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

#include <stdint.h>

#include <process/defer.hpp>
#include <process/dispatch.hpp>
#include <process/future.hpp>
#include <process/id.hpp>
#include <process/owned.hpp>
#include <process/process.hpp>
#include <process/shared.hpp>

#include <stout/check.hpp>
#include <stout/error.hpp>
#include <stout/foreach.hpp>
#include <stout/lambda.hpp>
#include <stout/nothing.hpp>
#include <stout/set.hpp>

#include "log/coordinator.hpp"
#include "log/log.hpp"
#include "log/network.hpp"
#include "log/recover.hpp"
#include "log/replica.hpp"

using namespace process;

using std::list;
using std::set;
using std::string;

namespace mesos {
namespace internal {
namespace log {

class LogProcess : public Process<LogProcess>
{
public:
  LogProcess(
      size_t _quorum,
      const string& path,
      const set<UPID>& pids,
      bool _autoInitialize);

  LogProcess(
      size_t _quorum,
      const string& path,
      const string& servers,
      const Duration& timeout,
      const string& znode,
      const Option<zookeeper::Authentication>& auth,
      bool _autoInitialize);

  // Recovers the log by catching up if needed. Returns a shared
  // pointer to the local replica if the recovery succeeds.
  Future<Shared<Replica> > recover();

protected:
  virtual void initialize();
  virtual void finalize();

private:
  friend class LogReaderProcess;
  friend class LogWriterProcess;

  // Continuations.
  void _recover();

  // TODO(benh): Factor this out into "membership renewer".
  void watch(
      const UPID& pid,
      const set<zookeeper::Group::Membership>& memberships);

  void failed(const string& message);
  void discarded();

  const size_t quorum;
  Shared<Replica> replica;
  Shared<Network> network;
  const bool autoInitialize;

  // For replica recovery.
  Option<Future<Owned<Replica> > > recovering;
  process::Promise<Nothing> recovered;
  list<process::Promise<Shared<Replica> >*> promises;

  // For renewing membership. We store a Group instance in order to
  // continually renew the replicas membership (when using ZooKeeper).
  zookeeper::Group* group;
  Future<zookeeper::Group::Membership> membership;
};


class LogReaderProcess : public Process<LogReaderProcess>
{
public:
  explicit LogReaderProcess(Log* log);

  Future<Log::Position> beginning();
  Future<Log::Position> ending();

  Future<list<Log::Entry> > read(
      const Log::Position& from,
      const Log::Position& to);

protected:
  virtual void initialize();
  virtual void finalize();

private:
  // Returns a position from a raw value.
  static Log::Position position(uint64_t value);

  // Returns a future which gets set when the log recovery has
  // finished (either succeeded or failed).
  Future<Nothing> recover();

  // Continuations.
  void _recover();

  Future<Log::Position> _beginning();
  Future<Log::Position> _ending();

  Future<list<Log::Entry> > _read(
      const Log::Position& from,
      const Log::Position& to);

  Future<list<Log::Entry> > __read(
      const Log::Position& from,
      const Log::Position& to,
      const list<Action>& actions);

  Future<Shared<Replica> > recovering;
  list<process::Promise<Nothing>*> promises;
};


class LogWriterProcess : public Process<LogWriterProcess>
{
public:
  explicit LogWriterProcess(Log* log);

  Future<Option<Log::Position> > start();
  Future<Option<Log::Position> > append(const string& bytes);
  Future<Option<Log::Position> > truncate(const Log::Position& to);

protected:
  virtual void initialize();
  virtual void finalize();

private:
  // Helper for converting an optional position returned from the
  // coordinator into a Log::Position.
  static Option<Log::Position> position(const Option<uint64_t>& position);

  // Returns a future which gets set when the log recovery has
  // finished (either succeeded or failed).
  Future<Nothing> recover();

  // Continuations.
  void _recover();

  Future<Option<Log::Position> > _start();
  Option<Log::Position> __start(const Option<uint64_t>& position);

  void failed(const string& message, const string& reason);

  const size_t quorum;
  const Shared<Network> network;

  Future<Shared<Replica> > recovering;
  list<process::Promise<Nothing>*> promises;

  Coordinator* coordinator;
  Option<string> error;
};


/////////////////////////////////////////////////
// Implementation of LogProcess.
/////////////////////////////////////////////////


LogProcess::LogProcess(
    size_t _quorum,
    const string& path,
    const set<UPID>& pids,
    bool _autoInitialize)
  : ProcessBase(ID::generate("log")),
    quorum(_quorum),
    replica(new Replica(path)),
    network(new Network(pids + (UPID) replica->pid())),
    autoInitialize(_autoInitialize),
    group(NULL) {}


LogProcess::LogProcess(
    size_t _quorum,
    const string& path,
    const string& servers,
    const Duration& timeout,
    const string& znode,
    const Option<zookeeper::Authentication>& auth,
    bool _autoInitialize)
  : ProcessBase(ID::generate("log")),
    quorum(_quorum),
    replica(new Replica(path)),
    network(new ZooKeeperNetwork(
        servers,
        timeout,
        znode,
        auth,
        Set<UPID>((UPID) replica->pid()))),
    autoInitialize(_autoInitialize),
    group(new zookeeper::Group(servers, timeout, znode, auth)) {}


void LogProcess::initialize()
{
  if (group != NULL) {
    // Need to add our replica to the ZooKeeper group!
    LOG(INFO) << "Attempting to join replica to ZooKeeper group";

    membership = group->join(replica->pid())
      .onFailed(defer(self(), &Self::failed, lambda::_1))
      .onDiscarded(defer(self(), &Self::discarded));

    // We save and pass the pid of the replica to the 'watch' function
    // because the field member 'replica' is not available during
    // recovery. We need the pid to renew the replicas membership.
    group->watch()
      .onReady(defer(self(), &Self::watch, replica->pid(), lambda::_1))
      .onFailed(defer(self(), &Self::failed, lambda::_1))
      .onDiscarded(defer(self(), &Self::discarded));
  }

  // Start the recovery.
  recover();
}


void LogProcess::finalize()
{
  if (recovering.isSome()) {
    // Stop the recovery if it is still pending.
    Future<Owned<Replica> > future = recovering.get();
    future.discard();
  }

  // If there exist operations that are gated by the recovery, we fail
  // all of them because the log is being deleted.
  foreach (process::Promise<Shared<Replica> >* promise, promises) {
    promise->fail("Log is being deleted");
    delete promise;
  }
  promises.clear();

  delete group;

  // Wait for the shared pointers 'network' and 'replica' to become
  // unique (i.e., no other reference to them). These calls should not
  // be blocking for too long because at this moment, all operations
  // should have been cancelled or are being cancelled. We do this
  // because we want to make sure that after the log is deleted, all
  // operations associated with this log are terminated.
  network.own().await();
  replica.own().await();
}


Future<Shared<Replica> > LogProcess::recover()
{
  // The future 'recovered' is used to mark the success (or the
  // failure) of the recovery. We do not use the future 'recovering'
  // to do that because it can be set in other process and thus has a
  // race condition which we want to avoid. We deliberately do not
  // save replica in 'recovered' because it will complicate our
  // deleting logic (see 'finalize').
  Future<Nothing> future = recovered.future();

  if (future.isDiscarded()) {
    return Failure("Not expecting discarded future");
  } else if (future.isFailed()) {
    return Failure(future.failure());
  } else if (future.isReady()) {
    return replica;
  }

  // Recovery has not finished yet. Create a promise and queue it such
  // that it can get notified once the recovery has finished (either
  // succeeded or failed).
  process::Promise<Shared<Replica> >* promise =
    new process::Promise<Shared<Replica> >();

  promises.push_back(promise);

  if (recovering.isNone()) {
    // TODO(jieyu): At this moment, we haven't shared 'replica' to
    // others yet. Therefore, the following 'replica.own()' call
    // should not be blocking. In the future, we may wanna support
    // 'release' in Shared which will provide this CHECK internally.
    CHECK(replica.unique());

    recovering =
      log::recover(
          quorum,
          replica.own().get(),
          network,
          autoInitialize)
      .onAny(defer(self(), &Self::_recover));
  }

  // TODO(benh): Add 'onDiscard' callback to our returned future.

  return promise->future();
}


void LogProcess::_recover()
{
  CHECK_SOME(recovering);

  Future<Owned<Replica> > future = recovering.get();

  if (!future.isReady()) {
    VLOG(2) << "Log recovery failed";

    // The 'future' here can only be discarded in 'finalize'.
    string failure = future.isFailed() ?
      future.failure() :
      "The future 'recovering' is unexpectedly discarded";

    // Mark the failure of the recovery.
    recovered.fail(failure);

    foreach (process::Promise<Shared<Replica> >* promise, promises) {
      promise->fail(failure);
      delete promise;
    }
    promises.clear();
  } else {
    VLOG(2) << "Log recovery completed";

    // Pull out the replica but need to make a copy since we get a
    // 'const &' from 'Try::get'.
    replica = Owned<Replica>(future.get()).share();

    // Mark the success of the recovery.
    recovered.set(Nothing());

    foreach (process::Promise<Shared<Replica> >* promise, promises) {
      promise->set(replica);
      delete promise;
    }
    promises.clear();
  }
}


void LogProcess::watch(
    const UPID& pid,
    const set<zookeeper::Group::Membership>& memberships)
{
  if (membership.isReady() && memberships.count(membership.get()) == 0) {
    // Our replica's membership must have expired, join back up.
    LOG(INFO) << "Renewing replica group membership";

    membership = group->join(pid)
      .onFailed(defer(self(), &Self::failed, lambda::_1))
      .onDiscarded(defer(self(), &Self::discarded));
  }

  group->watch(memberships)
    .onReady(defer(self(), &Self::watch, pid, lambda::_1))
    .onFailed(defer(self(), &Self::failed, lambda::_1))
    .onDiscarded(defer(self(), &Self::discarded));
}


void LogProcess::failed(const string& message)
{
  LOG(FATAL) << "Failed to participate in ZooKeeper group: " << message;
}


void LogProcess::discarded()
{
  LOG(FATAL) << "Not expecting future to get discarded!";
}


/////////////////////////////////////////////////
// Implementation of LogReaderProcess.
/////////////////////////////////////////////////


LogReaderProcess::LogReaderProcess(Log* log)
  : ProcessBase(ID::generate("log-reader")),
    recovering(dispatch(log->process, &LogProcess::recover)) {}


void LogReaderProcess::initialize()
{
  recovering.onAny(defer(self(), &Self::_recover));
}


void LogReaderProcess::finalize()
{
  foreach (process::Promise<Nothing>* promise, promises) {
    promise->fail("Log reader is being deleted");
    delete promise;
  }
  promises.clear();
}


Future<Nothing> LogReaderProcess::recover()
{
  if (recovering.isReady()) {
    return Nothing();
  } else if (recovering.isFailed()) {
    return Failure(recovering.failure());
  } else if (recovering.isDiscarded()) {
    return Failure("The future 'recovering' is unexpectedly discarded");
  }

  // At this moment, the future 'recovering' should most likely be
  // pending. But it is also likely that it gets set after the above
  // checks. Either way, we know that the continuation '_recover' has
  // not been called yet (otherwise, we should not be able to reach
  // here). The promise we are creating below will be properly
  // set/failed when '_recover' is called.
  process::Promise<Nothing>* promise = new process::Promise<Nothing>();
  promises.push_back(promise);

  // TODO(benh): Add 'onDiscard' callback to our returned future.

  return promise->future();
}


void LogReaderProcess::_recover()
{
  if (!recovering.isReady()) {
    foreach (process::Promise<Nothing>* promise, promises) {
      promise->fail(
          recovering.isFailed() ?
          recovering.failure() :
          "The future 'recovering' is unexpectedly discarded");
      delete promise;
    }
    promises.clear();
  } else {
    foreach (process::Promise<Nothing>* promise, promises) {
      promise->set(Nothing());
      delete promise;
    }
    promises.clear();
  }
}


Future<Log::Position> LogReaderProcess::beginning()
{
  return recover().then(defer(self(), &Self::_beginning));
}


Future<Log::Position> LogReaderProcess::_beginning()
{
  CHECK_READY(recovering);

  return recovering.get()->beginning()
    .then(lambda::bind(&Self::position, lambda::_1));
}


Future<Log::Position> LogReaderProcess::ending()
{
  return recover().then(defer(self(), &Self::_ending));
}


Future<Log::Position> LogReaderProcess::_ending()
{
  CHECK_READY(recovering);

  return recovering.get()->ending()
    .then(lambda::bind(&Self::position, lambda::_1));
}


Future<list<Log::Entry> > LogReaderProcess::read(
    const Log::Position& from,
    const Log::Position& to)
{
  return recover().then(defer(self(), &Self::_read, from, to));
}


Future<list<Log::Entry> > LogReaderProcess::_read(
    const Log::Position& from,
    const Log::Position& to)
{
  CHECK_READY(recovering);

  return recovering.get()->read(from.value, to.value)
    .then(defer(self(), &Self::__read, from, to, lambda::_1));
}


Future<list<Log::Entry> > LogReaderProcess::__read(
    const Log::Position& from,
    const Log::Position& to,
    const list<Action>& actions)
{
  list<Log::Entry> entries;

  uint64_t position = from.value;

  foreach (const Action& action, actions) {
    // Ensure read range is valid.
    if (!action.has_performed() ||
        !action.has_learned() ||
        !action.learned()) {
      return Failure("Bad read range (includes pending entries)");
    } else if (position++ != action.position()) {
      return Failure("Bad read range (includes missing entries)");
    }

    // And only return appends.
    CHECK(action.has_type());
    if (action.type() == Action::APPEND) {
      entries.push_back(Log::Entry(action.position(), action.append().bytes()));
    }
  }

  return entries;
}


Log::Position LogReaderProcess::position(uint64_t value)
{
  return Log::Position(value);
}


/////////////////////////////////////////////////
// Implementation of LogWriterProcess.
/////////////////////////////////////////////////


LogWriterProcess::LogWriterProcess(Log* log)
  : ProcessBase(ID::generate("log-writer")),
    quorum(log->process->quorum),
    network(log->process->network),
    recovering(dispatch(log->process, &LogProcess::recover)),
    coordinator(NULL),
    error(None()) {}


void LogWriterProcess::initialize()
{
  recovering.onAny(defer(self(), &Self::_recover));
}


void LogWriterProcess::finalize()
{
  foreach (process::Promise<Nothing>* promise, promises) {
    promise->fail("Log writer is being deleted");
    delete promise;
  }
  promises.clear();

  delete coordinator;
}


Future<Nothing> LogWriterProcess::recover()
{
  if (recovering.isReady()) {
    return Nothing();
  } else if (recovering.isFailed()) {
    return Failure(recovering.failure());
  } else if (recovering.isDiscarded()) {
    return Failure("The future 'recovering' is unexpectedly discarded");
  }

  // At this moment, the future 'recovering' should most likely be
  // pending. But it is also likely that it gets set after the above
  // checks. Either way, we know that the continuation '_recover' has
  // not been called yet (otherwise, we should not be able to reach
  // here). The promise we are creating below will be properly
  // set/failed when '_recover' is called.
  process::Promise<Nothing>* promise = new process::Promise<Nothing>();
  promises.push_back(promise);

  // TODO(benh): Add 'onDiscard' callback to our returned future.

  return promise->future();
}


void LogWriterProcess::_recover()
{
  if (!recovering.isReady()) {
    foreach (process::Promise<Nothing>* promise, promises) {
      promise->fail(
          recovering.isFailed() ?
          recovering.failure() :
          "The future 'recovering' is unexpectedly discarded");
      delete promise;
    }
    promises.clear();
  } else {
    foreach (process::Promise<Nothing>* promise, promises) {
      promise->set(Nothing());
      delete promise;
    }
    promises.clear();
  }
}


Future<Option<Log::Position> > LogWriterProcess::start()
{
  return recover().then(defer(self(), &Self::_start));
}


Future<Option<Log::Position> > LogWriterProcess::_start()
{
  // We delete the existing coordinator (if exists) and create a new
  // coordinator each time 'start' is called.
  // TODO(benh): We shouldn't need to delete the coordinator everytime.
  delete coordinator;
  error = None();

  CHECK_READY(recovering);

  coordinator = new Coordinator(quorum, recovering.get(), network);

  LOG(INFO) << "Attempting to start the writer";

  return coordinator->elect()
    .then(defer(self(), &Self::__start, lambda::_1))
    .onFailed(defer(self(), &Self::failed, "Failed to start", lambda::_1));
}


Option<Log::Position> LogWriterProcess::__start(
    const Option<uint64_t>& position)
{
  if (position.isNone()) {
    LOG(INFO) << "Could not start the writer, but can be retried";
    return None();
  }

  LOG(INFO) << "Writer started with ending position " << position.get();

  return Log::Position(position.get());
}


Future<Option<Log::Position> > LogWriterProcess::append(const string& bytes)
{
  LOG(INFO) << "Attempting to append " << bytes.size() << " bytes to the log";

  if (coordinator == NULL) {
    return Failure("No election has been performed");
  }

  if (error.isSome()) {
    return Failure(error.get());
  }

  return coordinator->append(bytes)
    .then(lambda::bind(&Self::position, lambda::_1))
    .onFailed(defer(self(), &Self::failed, "Failed to append", lambda::_1));
}


Future<Option<Log::Position> > LogWriterProcess::truncate(
    const Log::Position& to)
{
  LOG(INFO) << "Attempting to truncate the log to " << to.value;

  if (coordinator == NULL) {
    return Failure("No election has been performed");
  }

  if (error.isSome()) {
    return Failure(error.get());
  }

  return coordinator->truncate(to.value)
    .then(lambda::bind(&Self::position, lambda::_1))
    .onFailed(defer(self(), &Self::failed, "Failed to truncate", lambda::_1));
}


Option<Log::Position> LogWriterProcess::position(
    const Option<uint64_t>& position)
{
  if (position.isNone()) {
    return None();
  }

  return Log::Position(position.get());
}


void LogWriterProcess::failed(const string& message, const string& reason)
{
  error = message + ": " + reason;
}


/////////////////////////////////////////////////
// Public interfaces for Log.
/////////////////////////////////////////////////


Log::Log(
    int quorum,
    const string& path,
    const set<UPID>& pids,
    bool autoInitialize)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  process =
    new LogProcess(
        quorum,
        path,
        pids,
        autoInitialize);

  spawn(process);
}

Log::Log(
    int quorum,
    const string& path,
    const string& servers,
    const Duration& timeout,
    const string& znode,
    const Option<zookeeper::Authentication>& auth,
    bool autoInitialize)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  process =
    new LogProcess(
        quorum,
        path,
        servers,
        timeout,
        znode,
        auth,
        autoInitialize);

  spawn(process);
}


Log::~Log()
{
  terminate(process);
  process::wait(process);
  delete process;
}


/////////////////////////////////////////////////
// Public interfaces for Log::Reader.
/////////////////////////////////////////////////


Log::Reader::Reader(Log* log)
{
  process = new LogReaderProcess(log);
  spawn(process);
}


Log::Reader::~Reader()
{
  terminate(process);
  process::wait(process);
  delete process;
}


Future<list<Log::Entry> > Log::Reader::read(
    const Log::Position& from,
    const Log::Position& to)
{
  return dispatch(process, &LogReaderProcess::read, from, to);
}


Future<Log::Position> Log::Reader::beginning()
{
  return dispatch(process, &LogReaderProcess::beginning);
}


Future<Log::Position> Log::Reader::ending()
{
  return dispatch(process, &LogReaderProcess::ending);
}


/////////////////////////////////////////////////
// Public interfaces for Log::Writer.
/////////////////////////////////////////////////


Log::Writer::Writer(Log* log)
{
  process = new LogWriterProcess(log);
  spawn(process);
}


Log::Writer::~Writer()
{
  terminate(process);
  process::wait(process);
  delete process;
}


Future<Option<Log::Position> > Log::Writer::start()
{
  return dispatch(process, &LogWriterProcess::start);
}


Future<Option<Log::Position> > Log::Writer::append(const string& data)
{
  return dispatch(process, &LogWriterProcess::append, data);
}


Future<Option<Log::Position> > Log::Writer::truncate(const Log::Position& to)
{
  return dispatch(process, &LogWriterProcess::truncate, to);
}

} // namespace log {
} // namespace internal {
} // namespace mesos {
