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

#include <gtest/gtest.h>

#include <process/process.hpp>

#include <stout/flags.hpp>
#include <stout/nothing.hpp>
#include <stout/os.hpp>
#include <stout/try.hpp>

#include "logging/logging.hpp"

#include "messages/messages.hpp" // For GOOGLE_PROTOBUF_VERIFY_VERSION.

#include "tests/environment.hpp"
#include "tests/flags.hpp"
#include "tests/module.hpp"

using namespace mesos::internal;
using namespace mesos::internal::tests;

using std::cerr;
using std::cout;
using std::endl;
using std::string;


int main(int argc, char** argv)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  using mesos::internal::tests::flags; // Needed to disabmiguate.

  // Load flags from environment and command line but allow unknown
  // flags (since we might have gtest/gmock flags as well).
  Try<Nothing> load = flags.load("MESOS_", argc, argv, true);

  if (load.isError()) {
    cerr << flags.usage(load.error()) << endl;
    return EXIT_FAILURE;
  }

  if (flags.help) {
    cout << flags.usage() << endl;
    testing::InitGoogleTest(&argc, argv); // Get usage from gtest too.
    return EXIT_SUCCESS;
  }

  // Initialize Modules.
  Try<Nothing> result = tests::initModules(flags.modules);
  if (result.isError()) {
    cerr << "Error initializing modules: " << result.error() << endl;
    return EXIT_FAILURE;
  }

  // Disable /metrics/snapshot rate limiting, but do not
  // overwrite whatever the user set.
  os::setenv("LIBPROCESS_METRICS_SNAPSHOT_ENDPOINT_RATE_LIMIT", "", false);

  // Initialize libprocess.
  process::initialize();

  // Be quiet by default!
  if (!flags.verbose) {
    flags.quiet = true;
  }

  // Initialize logging.
  logging::initialize(argv[0], flags, true);

  // Initialize gmock/gtest.
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  cout << "Source directory: " << flags.source_dir << endl;
  cout << "Build directory: " << flags.build_dir << endl;

  // Instantiate our environment. Note that it will be managed by
  // gtest after we add it via testing::AddGlobalTestEnvironment.
  environment = new Environment(flags);

  testing::AddGlobalTestEnvironment(environment);

  return RUN_ALL_TESTS();
}
