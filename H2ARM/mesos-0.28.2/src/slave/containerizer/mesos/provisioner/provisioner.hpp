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

#ifndef __PROVISIONER_HPP__
#define __PROVISIONER_HPP__

#include <list>

#include <mesos/resources.hpp>

#include <mesos/docker/v1.hpp>

#include <mesos/slave/isolator.hpp> // For ContainerState.

#include <stout/nothing.hpp>
#include <stout/try.hpp>

#include <process/future.hpp>
#include <process/owned.hpp>

#include <process/metrics/counter.hpp>
#include <process/metrics/metrics.hpp>

#include "slave/flags.hpp"

#include "slave/containerizer/fetcher.hpp"

#include "slave/containerizer/mesos/provisioner/store.hpp"

namespace mesos {
namespace internal {
namespace slave {

// Forward declaration.
class Backend;
class ProvisionerProcess;
class Store;

// Provision info struct includes root filesystem for the container
// with specified image, all image manifests that include runtime
// configurations from the image will be passed to Mesos Containerizer.
struct ProvisionInfo
{
  std::string rootfs;

  // Docker v1 image manifest.
  Option<::docker::spec::v1::ImageManifest> dockerManifest;
};


class Provisioner
{
public:
  // Create the provisioner based on the specified flags.
  static Try<process::Owned<Provisioner>> create(const Flags& flags);

  // Available only for testing.
  explicit Provisioner(process::Owned<ProvisionerProcess> process);

  // NOTE: Made 'virtual' for mocking and testing.
  virtual ~Provisioner();

  // Recover root filesystems for containers from the run states and
  // the orphan containers (known to the launcher but not known to the
  // slave) detected by the launcher. This function is also
  // responsible for cleaning up any intermediate artifacts (e.g.
  // directories) to not leak anything.
  virtual process::Future<Nothing> recover(
      const std::list<mesos::slave::ContainerState>& states,
      const hashset<ContainerID>& orphans);

  // Provision a root filesystem for the container using the specified
  // image and return the absolute path to the root filesystem.
  virtual process::Future<ProvisionInfo> provision(
      const ContainerID& containerId,
      const Image& image);

  // Destroy a previously provisioned root filesystem. Assumes that
  // all references (e.g., mounts, open files) to the provisioned
  // filesystem have been removed. Return false if there is no
  // provisioned root filesystem for the given container.
  virtual process::Future<bool> destroy(const ContainerID& containerId);

protected:
  Provisioner() {} // For creating mock object.

private:
  Provisioner(const Provisioner&) = delete; // Not copyable.
  Provisioner& operator=(const Provisioner&) = delete; // Not assignable.

  process::Owned<ProvisionerProcess> process;
};


// Expose this class for testing only.
class ProvisionerProcess : public process::Process<ProvisionerProcess>
{
public:
  ProvisionerProcess(
      const Flags& flags,
      const std::string& rootDir,
      const hashmap<Image::Type, process::Owned<Store>>& stores,
      const hashmap<std::string, process::Owned<Backend>>& backends);

  process::Future<Nothing> recover(
      const std::list<mesos::slave::ContainerState>& states,
      const hashset<ContainerID>& orphans);

  process::Future<ProvisionInfo> provision(
      const ContainerID& containerId,
      const Image& image);

  process::Future<bool> destroy(const ContainerID& containerId);

private:
  process::Future<ProvisionInfo> _provision(
      const ContainerID& containerId,
      const ImageInfo& layers);

  process::Future<bool> _destroy(const ContainerID& containerId);

  const Flags flags;

  // Absolute path to the provisioner root directory. It can be
  // derived from '--work_dir' but we keep a separate copy here
  // because we converted it into an absolute path so managed rootfs
  // paths match the ones in 'mountinfo' (important if mount-based
  // backends are used).
  const std::string rootDir;

  const hashmap<Image::Type, process::Owned<Store>> stores;
  const hashmap<std::string, process::Owned<Backend>> backends;

  struct Info
  {
    // Mappings: backend -> {rootfsId, ...}
    hashmap<std::string, hashset<std::string>> rootfses;
  };

  hashmap<ContainerID, process::Owned<Info>> infos;

  struct Metrics
  {
    Metrics();
    ~Metrics();

    process::metrics::Counter remove_container_errors;
  } metrics;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __PROVISIONER_HPP__
