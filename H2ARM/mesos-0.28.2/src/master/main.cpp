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

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <mesos/mesos.hpp>

#include <mesos/authorizer/authorizer.hpp>

#include <mesos/master/allocator.hpp>

#include <mesos/module/anonymous.hpp>
#include <mesos/module/authorizer.hpp>

#include <process/limiter.hpp>
#include <process/owned.hpp>
#include <process/pid.hpp>

#include <stout/check.hpp>
#include <stout/duration.hpp>
#include <stout/exit.hpp>
#include <stout/flags.hpp>
#include <stout/nothing.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/path.hpp>
#include <stout/stringify.hpp>
#include <stout/strings.hpp>
#include <stout/try.hpp>

#include "common/build.hpp"
#include "common/protobuf_utils.hpp"

#include "hook/manager.hpp"

#include "logging/flags.hpp"
#include "logging/logging.hpp"

#include "master/contender.hpp"
#include "master/detector.hpp"
#include "master/master.hpp"
#include "master/registrar.hpp"
#include "master/repairer.hpp"

#include "master/allocator/mesos/hierarchical.hpp"

#include "module/manager.hpp"

#include "state/in_memory.hpp"
#include "state/log.hpp"
#include "state/protobuf.hpp"
#include "state/storage.hpp"

#include "version/version.hpp"

#include "zookeeper/detector.hpp"

using namespace mesos::internal;
using namespace mesos::internal::log;
using namespace mesos::internal::master;
using namespace zookeeper;

using mesos::Authorizer;
using mesos::MasterInfo;

using mesos::master::allocator::Allocator;

using mesos::modules::Anonymous;
using mesos::modules::ModuleManager;

using process::Owned;
using process::RateLimiter;
using process::UPID;

using process::firewall::DisabledEndpointsFirewallRule;
using process::firewall::FirewallRule;

using std::cerr;
using std::cout;
using std::endl;
using std::move;
using std::ostringstream;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;


void version()
{
  cout << "mesos" << " " << MESOS_VERSION << endl;
}


int main(int argc, char** argv)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  master::Flags flags;

  // The following flags are executable specific (e.g., since we only
  // have one instance of libprocess per execution, we only want to
  // advertise the IP and port option once, here).
  Option<string> ip;
  flags.add(&ip,
            "ip",
            "IP address to listen on. This cannot be used in conjunction\n"
            "with `--ip_discovery_command`.");

  uint16_t port;
  flags.add(&port,
            "port",
            "Port to listen on.",
            MasterInfo().port());

  Option<string> advertise_ip;
  flags.add(&advertise_ip,
            "advertise_ip",
            "IP address advertised to reach mesos master.\n"
            "Mesos master does not bind using this IP address.\n"
            "However, this IP address may be used to access Mesos master.");

  Option<string> advertise_port;
  flags.add(&advertise_port,
            "advertise_port",
            "Port advertised to reach mesos master (along with\n"
            "`advertise_ip`). Mesos master does not bind using this port.\n"
            "However, this port (along with `advertise_ip`) may be used to\n"
            "access Mesos master.");

  Option<string> zk;
  flags.add(&zk,
            "zk",
            "ZooKeeper URL (used for leader election amongst masters)\n"
            "May be one of:\n"
            "  `zk://host1:port1,host2:port2,.../path`\n"
            "  `zk://username:password@host1:port1,host2:port2,.../path`\n"
            "  `file:///path/to/file` (where file contains one of the above)\n"
            "NOTE: Not required if master is run in standalone mode (non-HA).");

  // Optional IP discover script that will set the Master IP.
  // If set, its output is expected to be a valid parseable IP string.
  Option<string> ip_discovery_command;
  flags.add(&ip_discovery_command,
            "ip_discovery_command",
            "Optional IP discovery binary: if set, it is expected to emit\n"
            "the IP address which the master will try to bind to.\n"
            "Cannot be used in conjunction with `--ip`.");

  Try<Nothing> load = flags.load("MESOS_", argc, argv);

  if (load.isError()) {
    cerr << flags.usage(load.error()) << endl;
    return EXIT_FAILURE;
  }

  if (flags.version) {
    version();
    return EXIT_SUCCESS;
  }

  if (flags.help) {
    cout << flags.usage() << endl;
    return EXIT_SUCCESS;
  }

  // Initialize modules. Note that since other subsystems may depend
  // upon modules, we should initialize modules before anything else.
  if (flags.modules.isSome()) {
    Try<Nothing> result = ModuleManager::load(flags.modules.get());
    if (result.isError()) {
      EXIT(EXIT_FAILURE) << "Error loading modules: " << result.error();
    }
  }

  // Initialize hooks.
  if (flags.hooks.isSome()) {
    Try<Nothing> result = HookManager::initialize(flags.hooks.get());
    if (result.isError()) {
      EXIT(EXIT_FAILURE) << "Error installing hooks: " << result.error();
    }
  }

  if (ip_discovery_command.isSome() && ip.isSome()) {
    EXIT(EXIT_FAILURE) << flags.usage(
        "Only one of `--ip` or `--ip_discovery_command` should be specified");
  }

  if (ip_discovery_command.isSome()) {
    Try<string> ipAddress = os::shell(ip_discovery_command.get());

    if (ipAddress.isError()) {
      EXIT(EXIT_FAILURE) << ipAddress.error();
    }

    os::setenv("LIBPROCESS_IP", strings::trim(ipAddress.get()));
  } else if (ip.isSome()) {
    os::setenv("LIBPROCESS_IP", ip.get());
  }

  os::setenv("LIBPROCESS_PORT", stringify(port));

  if (advertise_ip.isSome()) {
    os::setenv("LIBPROCESS_ADVERTISE_IP", advertise_ip.get());
  }

  if (advertise_port.isSome()) {
    os::setenv("LIBPROCESS_ADVERTISE_PORT", advertise_port.get());
  }

  // Initialize libprocess.
  process::initialize("master");

  logging::initialize(argv[0], flags, true); // Catch signals.

  spawn(new VersionProcess(), true);

  LOG(INFO) << "Build: " << build::DATE << " by " << build::USER;

  LOG(INFO) << "Version: " << MESOS_VERSION;

  if (build::GIT_TAG.isSome()) {
    LOG(INFO) << "Git tag: " << build::GIT_TAG.get();
  }

  if (build::GIT_SHA.isSome()) {
    LOG(INFO) << "Git SHA: " << build::GIT_SHA.get();
  }

  // Create an instance of allocator.
  const string allocatorName = flags.allocator;
  Try<Allocator*> allocator = Allocator::create(allocatorName);

  if (allocator.isError()) {
    EXIT(EXIT_FAILURE)
      << "Failed to create '" << allocatorName
      << "' allocator: " << allocator.error();
  }

  CHECK_NOTNULL(allocator.get());
  LOG(INFO) << "Using '" << allocatorName << "' allocator";

  state::Storage* storage = NULL;
  Log* log = NULL;

  if (flags.registry == "in_memory") {
    if (flags.registry_strict) {
      EXIT(EXIT_FAILURE)
        << "Cannot use '--registry_strict' when using in-memory storage"
        << " based registry";
    }
    storage = new state::InMemoryStorage();
  } else if (flags.registry == "replicated_log" ||
             flags.registry == "log_storage") {
    // TODO(bmahler): "log_storage" is present for backwards
    // compatibility, can be removed before 0.19.0.
    if (flags.work_dir.isNone()) {
      EXIT(EXIT_FAILURE)
        << "--work_dir needed for replicated log based registry";
    }

    Try<Nothing> mkdir = os::mkdir(flags.work_dir.get());
    if (mkdir.isError()) {
      EXIT(EXIT_FAILURE)
        << "Failed to create work directory '" << flags.work_dir.get()
        << "': " << mkdir.error();
    }

    if (zk.isSome()) {
      // Use replicated log with ZooKeeper.
      if (flags.quorum.isNone()) {
        EXIT(EXIT_FAILURE)
          << "Need to specify --quorum for replicated log based"
          << " registry when using ZooKeeper";
      }

      Try<zookeeper::URL> url = zookeeper::URL::parse(zk.get());
      if (url.isError()) {
        EXIT(EXIT_FAILURE) << "Error parsing ZooKeeper URL: " << url.error();
      }

      log = new Log(
          flags.quorum.get(),
          path::join(flags.work_dir.get(), "replicated_log"),
          url.get().servers,
          flags.zk_session_timeout,
          path::join(url.get().path, "log_replicas"),
          url.get().authentication,
          flags.log_auto_initialize);
    } else {
      // Use replicated log without ZooKeeper.
      log = new Log(
          1,
          path::join(flags.work_dir.get(), "replicated_log"),
          set<UPID>(),
          flags.log_auto_initialize);
    }
    storage = new state::LogStorage(log);
  } else {
    EXIT(EXIT_FAILURE)
      << "'" << flags.registry << "' is not a supported"
      << " option for registry persistence";
  }

  CHECK_NOTNULL(storage);

  state::protobuf::State* state = new state::protobuf::State(storage);
  Registrar* registrar = new Registrar(flags, state);
  Repairer* repairer = new Repairer();

  Files files;

  MasterContender* contender;
  MasterDetector* detector;

  Try<MasterContender*> contender_ = MasterContender::create(zk);
  if (contender_.isError()) {
    EXIT(EXIT_FAILURE)
      << "Failed to create a master contender: " << contender_.error();
  }
  contender = contender_.get();

  Try<MasterDetector*> detector_ = MasterDetector::create(zk);
  if (detector_.isError()) {
    EXIT(EXIT_FAILURE)
      << "Failed to create a master detector: " << detector_.error();
  }
  detector = detector_.get();

  Option<Authorizer*> authorizer = None();

  auto authorizerNames = strings::split(flags.authorizers, ",");
  if (authorizerNames.empty()) {
    EXIT(EXIT_FAILURE) << "No authorizer specified";
  }
  if (authorizerNames.size() > 1) {
    EXIT(EXIT_FAILURE) << "Multiple authorizers not supported";
  }
  string authorizerName = authorizerNames[0];

  // NOTE: The flag --authorizers overrides the flag --acls, i.e. if
  // a non default authorizer is requested, it will be used and
  // the contents of --acls will be ignored.
  // TODO(arojas): Add support for multiple authorizers.
  if (authorizerName != master::DEFAULT_AUTHORIZER ||
      flags.acls.isSome()) {
    Try<Authorizer*> create = Authorizer::create(authorizerName);

    if (create.isError()) {
      EXIT(EXIT_FAILURE) << "Could not create '" << authorizerName
                         << "' authorizer: " << create.error();
    }

    authorizer = create.get();

    LOG(INFO) << "Using '" << authorizerName << "' authorizer";

    // Only default authorizer requires initialization, see the comment
    // for `initialize()` in "mesos/authorizer/authorizer.hpp".
    if (authorizerName == master::DEFAULT_AUTHORIZER) {
      Try<Nothing> initialize = authorizer.get()->initialize(flags.acls.get());

      if (initialize.isError()) {
        // Failing to initialize the authorizer leads to undefined
        // behavior, therefore we default to skip authorization
        // altogether.
        LOG(WARNING) << "Authorization disabled: Failed to initialize '"
                     << authorizerName << "' authorizer: "
                     << initialize.error();

        delete authorizer.get();
        authorizer = None();
      }
    } else if (flags.acls.isSome()) {
      LOG(WARNING) << "Ignoring contents of --acls flag, because '"
                   << authorizerName << "' authorizer will be used instead "
                   << " of the default.";
    }
  }

  Option<shared_ptr<RateLimiter>> slaveRemovalLimiter = None();
  if (flags.slave_removal_rate_limit.isSome()) {
    // Parse the flag value.
    // TODO(vinod): Move this parsing logic to flags once we have a
    // 'Rate' abstraction in stout.
    vector<string> tokens =
      strings::tokenize(flags.slave_removal_rate_limit.get(), "/");

    if (tokens.size() != 2) {
      EXIT(EXIT_FAILURE)
        << "Invalid slave_removal_rate_limit: "
        << flags.slave_removal_rate_limit.get()
        << ". Format is <Number of slaves>/<Duration>";
    }

    Try<int> permits = numify<int>(tokens[0]);
    if (permits.isError()) {
      EXIT(EXIT_FAILURE)
        << "Invalid slave_removal_rate_limit: "
        << flags.slave_removal_rate_limit.get()
        << ". Format is <Number of slaves>/<Duration>"
        << ": " << permits.error();
    }

    Try<Duration> duration = Duration::parse(tokens[1]);
    if (duration.isError()) {
      EXIT(EXIT_FAILURE)
        << "Invalid slave_removal_rate_limit: "
        << flags.slave_removal_rate_limit.get()
        << ". Format is <Number of slaves>/<Duration>"
        << ": " << duration.error();
    }

    slaveRemovalLimiter = new RateLimiter(permits.get(), duration.get());
  }

  if (flags.firewall_rules.isSome()) {
    vector<Owned<FirewallRule>> rules;

    const Firewall firewall = flags.firewall_rules.get();

    if (firewall.has_disabled_endpoints()) {
      hashset<string> paths;

      foreach (const string& path, firewall.disabled_endpoints().paths()) {
        paths.insert(path);
      }

      rules.emplace_back(new DisabledEndpointsFirewallRule(paths));
    }

    process::firewall::install(move(rules));
  }

  // Create anonymous modules.
  foreach (const string& name, ModuleManager::find<Anonymous>()) {
    Try<Anonymous*> create = ModuleManager::create<Anonymous>(name);
    if (create.isError()) {
      EXIT(EXIT_FAILURE)
        << "Failed to create anonymous module named '" << name << "'";
    }

    // We don't bother keeping around the pointer to this anonymous
    // module, when we exit that will effectively free it's memory.
    //
    // TODO(benh): We might want to add explicit finalization (and
    // maybe explicit initialization too) in order to let the module
    // do any housekeeping necessary when the master is cleanly
    // terminating.
  }

  LOG(INFO) << "Starting Mesos master";

  Master* master =
    new Master(
      allocator.get(),
      registrar,
      repairer,
      &files,
      contender,
      detector,
      authorizer,
      slaveRemovalLimiter,
      flags);

  if (zk.isNone()) {
    // It means we are using the standalone detector so we need to
    // appoint this Master as the leader.
    dynamic_cast<StandaloneMasterDetector*>(detector)->appoint(master->info());
  }

  process::spawn(master);
  process::wait(master->self());

  delete master;
  delete allocator.get();

  delete registrar;
  delete repairer;
  delete state;
  delete storage;
  delete log;

  delete contender;
  delete detector;

  if (authorizer.isSome()) {
    delete authorizer.get();
  }

  return EXIT_SUCCESS;
}
