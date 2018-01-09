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

#include <mesos/master/allocator.hpp>

#include <mesos/module/allocator.hpp>

#include "master/constants.hpp"

#include "master/allocator/mesos/hierarchical.hpp"

#include "module/manager.hpp"

using std::string;

using mesos::internal::master::allocator::HierarchicalDRFAllocator;

namespace mesos {
namespace master {
namespace allocator {

Try<Allocator*> Allocator::create(const string& name)
{
  // Create an instance of the default allocator. If other than the
  // default allocator is requested, search for it in loaded modules.
  // NOTE: We do not need an extra not-null check, because both
  // ModuleManager and built-in allocator factory do that already.
  if (name == mesos::internal::master::DEFAULT_ALLOCATOR) {
    return HierarchicalDRFAllocator::create();
  }

  return modules::ModuleManager::create<Allocator>(name);
}

} // namespace allocator {
} // namespace master {
} // namespace mesos {
