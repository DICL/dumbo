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
// limitations under the License

#ifndef __STATE_LEVELDB_HPP__
#define __STATE_LEVELDB_HPP__

#include <set>
#include <string>

#include <process/future.hpp>

#include <stout/option.hpp>
#include <stout/try.hpp>
#include <stout/uuid.hpp>

#include "messages/state.hpp"

#include "state/storage.hpp"

namespace mesos {
namespace internal {
namespace state {

// More forward declarations.
class LevelDBStorageProcess;


class LevelDBStorage : public Storage
{
public:
  explicit LevelDBStorage(const std::string& path);
  virtual ~LevelDBStorage();

  // Storage implementation.
  virtual process::Future<Option<Entry> > get(const std::string& name);
  virtual process::Future<bool> set(const Entry& entry, const UUID& uuid);
  virtual process::Future<bool> expunge(const Entry& entry);
  virtual process::Future<std::set<std::string> > names();

private:
  LevelDBStorageProcess* process;
};

} // namespace state {
} // namespace internal {
} // namespace mesos {

#endif // __STATE_LEVELDB_HPP__
