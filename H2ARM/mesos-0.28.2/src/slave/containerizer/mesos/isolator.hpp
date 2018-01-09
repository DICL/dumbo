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

#ifndef __ISOLATOR_HPP__
#define __ISOLATOR_HPP__

#include <mesos/slave/isolator.hpp>

#include <process/owned.hpp>
#include <process/process.hpp>

#include <stout/none.hpp>

namespace mesos {
namespace internal {
namespace slave {

// Forward declaration.
class MesosIsolatorProcess;


// A wrapper class that implements the 'Isolator' interface which is
// backed by an 'MesosIsolatorProcess'. This helper class is useful
// for programmers to write asynchronous isolators.
class MesosIsolator : public mesos::slave::Isolator
{
public:
  explicit MesosIsolator(process::Owned<MesosIsolatorProcess> process);
  virtual ~MesosIsolator();

  virtual process::Future<Nothing> recover(
      const std::list<mesos::slave::ContainerState>& states,
      const hashset<ContainerID>& orphans);

  virtual process::Future<Option<mesos::slave::ContainerLaunchInfo>> prepare(
      const ContainerID& containerId,
      const mesos::slave::ContainerConfig& containerConfig);

  virtual process::Future<Nothing> isolate(
      const ContainerID& containerId,
      pid_t pid);

  virtual process::Future<mesos::slave::ContainerLimitation> watch(
      const ContainerID& containerId);

  virtual process::Future<Nothing> update(
      const ContainerID& containerId,
      const Resources& resources);

  virtual process::Future<ResourceStatistics> usage(
      const ContainerID& containerId);

  virtual process::Future<ContainerStatus> status(
      const ContainerID& containerId);

  virtual process::Future<Nothing> cleanup(
      const ContainerID& containerId);

private:
  process::Owned<MesosIsolatorProcess> process;
};


class MesosIsolatorProcess : public process::Process<MesosIsolatorProcess>
{
public:
  virtual ~MesosIsolatorProcess() {}

  virtual process::Future<Nothing> recover(
      const std::list<mesos::slave::ContainerState>& states,
      const hashset<ContainerID>& orphans) = 0;

  virtual process::Future<Option<mesos::slave::ContainerLaunchInfo>> prepare(
      const ContainerID& containerId,
      const mesos::slave::ContainerConfig& containerConfig) = 0;

  virtual process::Future<Nothing> isolate(
      const ContainerID& containerId,
      pid_t pid) = 0;

  virtual process::Future<mesos::slave::ContainerLimitation> watch(
      const ContainerID& containerId) = 0;

  virtual process::Future<Nothing> update(
      const ContainerID& containerId,
      const Resources& resources) = 0;

  virtual process::Future<ResourceStatistics> usage(
      const ContainerID& containerId) = 0;

  // A method to query the isolator processes about the run-time state
  // of isolator specific properties associated with a container.
  // Unlike other methods in this class we are not making this pure
  // virtual, since isolators do not necessarily need to provide the
  // status of isolated containers.
  virtual process::Future<ContainerStatus> status(
      const ContainerID& containerId)
  {
    return ContainerStatus();
  };

  virtual process::Future<Nothing> cleanup(
      const ContainerID& containerId) = 0;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __ISOLATOR_HPP__
