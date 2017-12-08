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

#include <set>
#include <string>

#include <stout/foreach.hpp>
#include <stout/subcommand.hpp>
#include <stout/try.hpp>

#include "linux/ns.hpp"

#include "tests/containerizer/setns_test_helper.hpp"

using std::set;
using std::string;

const char SetnsTestHelper::NAME[] = "SetnsTestHelper";

int SetnsTestHelper::execute()
{
  // Get all the available namespaces.
  set<string> namespaces = ns::namespaces();

  // Note: /proc has not been remounted so we can look up pid 1's
  // namespaces, even if we're in a separate pid namespace.
  foreach (const string& ns, namespaces) {
    if (ns == "pid") {
      // ns::setns() does not (currently) support pid namespaces so
      // this should return an error.
      Try<Nothing> setns = ns::setns(1, ns);
      if (!setns.isError()) {
        return 1;
      }
    } else if (ns == "user") {
      // ns::setns() will also fail with user namespaces, so we skip
      // for now. See MESOS-3083.
      continue;
    } else {
      Try<Nothing> setns = ns::setns(1, ns);
      if (!setns.isSome()) {
        return 1;
      }
    }
  }

  return 0;
}
