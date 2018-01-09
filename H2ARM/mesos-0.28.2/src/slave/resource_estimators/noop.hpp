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

#ifndef __SLAVE_RESOURCE_ESTIMATORS_NOOP_HPP__
#define __SLAVE_RESOURCE_ESTIMATORS_NOOP_HPP__

#include <mesos/slave/resource_estimator.hpp>

#include <stout/lambda.hpp>

#include <process/owned.hpp>

namespace mesos {
namespace internal {
namespace slave {

// Forward declaration.
class NoopResourceEstimatorProcess;


// A noop resource estimator which tells the master that no resource
// on the slave can be oversubscribed. Using this resource estimator
// will effectively turn off the oversubscription on the slave.
class NoopResourceEstimator : public mesos::slave::ResourceEstimator
{
public:
  virtual ~NoopResourceEstimator();

  virtual Try<Nothing> initialize(
      const lambda::function<process::Future<ResourceUsage>()>& usage);

  virtual process::Future<Resources> oversubscribable();

protected:
  process::Owned<NoopResourceEstimatorProcess> process;
};


} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __SLAVE_RESOURCE_ESTIMATORS_NOOP_HPP__
