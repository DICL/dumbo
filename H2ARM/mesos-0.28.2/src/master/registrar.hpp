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

#ifndef __MASTER_REGISTRAR_HPP__
#define __MASTER_REGISTRAR_HPP__

#include <mesos/mesos.hpp>

#include <stout/hashset.hpp>

#include <process/future.hpp>
#include <process/owned.hpp>

#include "master/flags.hpp"
#include "master/registry.hpp"

#include "state/protobuf.hpp"

namespace mesos {
namespace internal {
namespace master {

// Forward declaration.
class RegistrarProcess;

// Defines an abstraction for operations that can be applied on the
// Registry.
// TODO(xujyan): Make Operation generic so that we can apply them
// against a generic "batch operation applier" abstraction, see TODO
// below for more details.
class Operation : public process::Promise<bool>
{
public:
  Operation() : success(false) {}
  virtual ~Operation() {}

  // Attempts to invoke the operation on the registry object.
  // Aided by accumulator(s):
  //   slaveIDs - is the set of registered slaves.
  //
  // NOTE: the "strict" parameter only applies to operations that
  // affect slaves (i.e. registration).  See Flags::registry_strict
  // in master/flags.cpp for more information.
  //
  // Returns whether the operation mutates 'registry', or an error if
  // the operation cannot be applied successfully.
  Try<bool> operator()(
      Registry* registry,
      hashset<SlaveID>* slaveIDs,
      bool strict)
  {
    const Try<bool> result = perform(registry, slaveIDs, strict);

    success = !result.isError();

    return result;
  }

  // Sets the promise based on whether the operation was successful.
  bool set() { return process::Promise<bool>::set(success); }

protected:
  virtual Try<bool> perform(
      Registry* registry,
      hashset<SlaveID>* slaveIDs,
      bool strict) = 0;

private:
  bool success;
};


// TODO(xujyan): Add a generic abstraction for applying batches of
// operations against State Variables. The Registrar and other
// components could leverage this. This abstraction would be
// templatized to take the type, along with any accumulators:
// template <typename T,
//           typename X = None,
//           typename Y = None,
//           typename Z = None>
// T: the data type that the batch operations can be applied on.
// X, Y, Z: zero to 3 generic accumulators that facilitate the batch
// of operations.
// This allows us to reuse the idea of "doing batches of operations"
// on other potential new state variables (i.e. repair state, offer
// reservations, etc).
class Registrar
{
public:
  // If flags.registry_strict is true, all operations will be
  // permitted.
  Registrar(const Flags& flags, state::protobuf::State* state);
  ~Registrar();

  // Recovers the Registry, persisting the new Master information.
  // The Registrar must be recovered to allow other operations to
  // proceed.
  // TODO(bmahler): Consider a "factory" for constructing the
  // Registrar, to eliminate the need for passing 'MasterInfo'.
  // This is required as the Registrar is injected into the Master,
  // and therefore MasterInfo is unknown during construction.
  process::Future<Registry> recover(const MasterInfo& info);

  // Applies an operation on the Registry.
  // Returns:
  //   true if the operation is permitted.
  //   false if the operation is not permitted.
  //   Failure if the operation fails (possibly lost log leadership),
  //     or recovery failed.
  process::Future<bool> apply(process::Owned<Operation> operation);

private:
  RegistrarProcess* process;
};

} // namespace master {
} // namespace internal {
} // namespace mesos {

#endif // __MASTER_REGISTRAR_HPP__
