// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#ifndef __PROCESS_NETWORK_HPP__
#define __PROCESS_NETWORK_HPP__

#include <process/address.hpp>

#include <stout/net.hpp>
#include <stout/try.hpp>

namespace process {
namespace network {

/**
 * Returns a socket file descriptor for the specified options.
 *
 * **NOTE:** on OS X, the returned socket will have the SO_NOSIGPIPE
 * option set.
 */
inline Try<int> socket(int family, int type, int protocol)
{
  int s;
  if ((s = ::socket(family, type, protocol)) == -1) {
    return ErrnoError();
  }

#ifdef __APPLE__
  // Disable SIGPIPE via setsockopt because OS X does not support
  // the MSG_NOSIGNAL flag on send(2).
  const int enable = 1;
  if (setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &enable, sizeof(int)) == -1) {
    return ErrnoError();
  }
#endif // __APPLE__

  return s;
}


// TODO(benh): Remove and defer to Socket::accept.
inline Try<int> accept(int s)
{
  struct sockaddr_storage storage;
  socklen_t storagelen = sizeof(storage);

  int accepted = ::accept(s, (struct sockaddr*) &storage, &storagelen);
  if (accepted < 0) {
    return ErrnoError("Failed to accept");
  }

  return accepted;
}


// TODO(benh): Remove and defer to Socket::bind.
inline Try<int> bind(int s, const Address& address)
{
  struct sockaddr_storage storage =
    net::createSockaddrStorage(address.ip, address.port);

  int error = ::bind(s, (struct sockaddr*) &storage, address.size());
  if (error < 0) {
    return ErrnoError("Failed to bind on " + stringify(address));
  }

  return error;
}


// TODO(benh): Remove and defer to Socket::connect.
inline Try<int> connect(int s, const Address& address)
{
  struct sockaddr_storage storage =
    net::createSockaddrStorage(address.ip, address.port);

  int error = ::connect(s, (struct sockaddr*) &storage, address.size());
  if (error < 0) {
    return ErrnoError("Failed to connect to " + stringify(address));
  }

  return error;
}


/**
 * Returns the `Address` with the assigned ip and assigned port.
 *
 * @return An `Address` or an error if the `getsockname` system call
 *     fails or the family type is not supported.
 */
inline Try<Address> address(int s)
{
  struct sockaddr_storage storage;
  socklen_t storagelen = sizeof(storage);

  if (::getsockname(s, (struct sockaddr*) &storage, &storagelen) < 0) {
    return ErrnoError("Failed to getsockname");
  }

  return Address::create(storage);
}


/**
 * Returns the peer's `Address` for the accepted or connected socket.
 *
 * @return An `Address` or an error if the `getpeername` system call
 *     fails or the family type is not supported.
 */
inline Try<Address> peer(int s)
{
  struct sockaddr_storage storage;
  socklen_t storagelen = sizeof(storage);

  if (::getpeername(s, (struct sockaddr*) &storage, &storagelen) < 0) {
    return ErrnoError("Failed to getpeername");
  }

  return Address::create(storage);
}

} // namespace network {
} // namespace process {

#endif // __PROCESS_NETWORK_HPP__
