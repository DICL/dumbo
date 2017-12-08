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

#ifndef __URI_FETCHERS_DOCKER_HPP__
#define __URI_FETCHERS_DOCKER_HPP__

#include <process/owned.hpp>

#include <stout/flags.hpp>
#include <stout/try.hpp>

#include <mesos/uri/fetcher.hpp>

namespace mesos {
namespace uri {

// Forward declarations.
class DockerFetcherPluginProcess;


class DockerFetcherPlugin : public Fetcher::Plugin
{
public:
  class Flags : public virtual flags::FlagsBase {};

  static Try<process::Owned<Fetcher::Plugin>> create(const Flags& flags);

  virtual ~DockerFetcherPlugin();

  virtual std::set<std::string> schemes();

  virtual process::Future<Nothing> fetch(
      const URI& uri,
      const std::string& directory);

private:
  explicit DockerFetcherPlugin(
      process::Owned<DockerFetcherPluginProcess> _process);

  process::Owned<DockerFetcherPluginProcess> process;
};

} // namespace uri {
} // namespace mesos {

#endif // __URI_FETCHERS_DOCKER_HPP__
