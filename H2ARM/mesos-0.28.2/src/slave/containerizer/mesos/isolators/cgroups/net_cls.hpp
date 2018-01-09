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

#ifndef __CGROUPS_NET_CLS_ISOLATOR_HPP__
#define __CGROUPS_NET_CLS_ISOLATOR_HPP__

#include <stdint.h>

#include <bitset>
#include <iostream>
#include <string>

#include <stout/hashmap.hpp>
#include <stout/interval.hpp>
#include <stout/none.hpp>
#include <stout/nothing.hpp>
#include <stout/result.hpp>
#include <stout/try.hpp>

#include "slave/containerizer/mesos/isolator.hpp"

namespace mesos {
namespace internal {
namespace slave {

// This defines the net_cls handle. The handle is composed of two
// parts, a 16-bit primary handle and a 16-bit secondary handle.
//
// TODO(asridharan): Currently we need to define the net_cls handle
// here, since we cannot use the definitions in
// `src/linux/routing/handle.hpp` due to its dependency on `libnl`,
// which is under GPL. Once we have been able to resolve these issues
// we should remove this definition and use the definition presented
// in `src/linux/routing/handle.hpp`.
struct NetClsHandle
{
  NetClsHandle(uint16_t _primary, uint16_t _secondary)
    : primary(_primary), secondary(_secondary) {};

  explicit NetClsHandle(uint32_t handle)
  {
    primary = handle >> 16;
    secondary = handle & 0xffff;
  };

  // Get the 32-bit representation of the handle in the form of
  // 0xAAAABBBB. Where 0xAAAA is the primary handle and 0xBBBB is the
  // secondary handle.
  uint32_t get() const
  {
    uint32_t handle = primary;

    handle <<= 16;
    handle |= secondary;

    return handle;
  };

  uint16_t primary;
  uint16_t secondary;
};


std::ostream& operator<<(std::ostream& stream, const NetClsHandle& obj);


// This manages the net_cls handles for the `cgroup/net_cls` isolator.
// The isolator can use this with a range of primary handles, which
// will be managed by this class. For each primary handle there are
// 64K possible secondary handles. For a given primary handle the
// isolator can get a secondary handle by calling `alloc` and release
// an allocated handle by calling `free` on the secondary handle. For
// a given primary handle, the isolator can also explicitly reserve a
// secondary handle by calling `reserve`.
class NetClsHandleManager
{
public:
  NetClsHandleManager(
      const IntervalSet<uint32_t>& _primaries,
      const IntervalSet<uint32_t>& _secondaries = IntervalSet<uint32_t>());

  ~NetClsHandleManager() {};

  // Allocates a primary handle from the given interval set.
  Try<uint16_t> allocPrimary() { return Error("Not Implemented"); }
  Try<NetClsHandle> alloc(const Option<uint16_t>& primary = None());

  Try<Nothing> reserve(const NetClsHandle& handle);
  Try<Nothing> free(const NetClsHandle& handle);

  // Check if a handle is used.
  Try<bool> isUsed(const NetClsHandle& handle);

private:
  // The key to this hashmap is the 16-bit primary handle.
  hashmap<uint16_t, std::bitset<0x10000>> used;

  // NOTE: Though the primary and secondary handles are 16 bit, we
  // cannot use an `IntervalSet` specialization of type `uint16_t`
  // since the intervals are stored in right openf format -- [x,y) --
  // and setting the type to `uint16_t` would lead to overflow errors.
  // For e.g., we would not be able to store the range [0xffff,0xffff]
  // in `IntervalSet<uint16_t>` due to overflow error.
  IntervalSet<uint32_t> primaries;
  IntervalSet<uint32_t> secondaries;
};


// Uses the Linux net_cls subsystem for allocating network handles to
// containers. The network handles of a net_cls cgroup will be used
// for tagging packets originating from containers belonging to that
// cgroup. The tags on the packets can then be used by applications,
// such as traffic-controllers (tc) and firewalls (iptables), to
// provide network performance isolation. A more detailed explanation
// can be found at:
// https://www.kernel.org/doc/Documentation/cgroups/net_cls.txt
class CgroupsNetClsIsolatorProcess : public MesosIsolatorProcess
{
public:
  static Try<mesos::slave::Isolator*> create(const Flags& flags);

  virtual ~CgroupsNetClsIsolatorProcess();

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
  struct Info
  {
    Info(const std::string& _cgroup)
      : cgroup(_cgroup) {}

    Info(const std::string& _cgroup, const NetClsHandle &_handle)
      : cgroup(_cgroup), handle(_handle) {}

    const std::string cgroup;
    const Option<NetClsHandle> handle;
  };

  CgroupsNetClsIsolatorProcess(
      const Flags& _flags,
      const std::string& _hierarchy,
      const IntervalSet<uint32_t>& primaries,
      const IntervalSet<uint32_t>& secondaries);

  process::Future<Nothing> _cleanup(
      const ContainerID& containerId);

  Result<NetClsHandle> recoverHandle(
      const std::string& hierarchy,
      const std::string& cgroup);

  const Flags flags;
  const std::string hierarchy;

  hashmap<ContainerID, Info> infos;
  Option<NetClsHandleManager> handleManager;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {
#endif // __CGROUPS_NET_CLS_ISOLATOR_HPP__
