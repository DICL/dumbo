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

#ifndef __PROCESS_LOGGING_HPP__
#define __PROCESS_LOGGING_HPP__

#include <glog/logging.h>

#include <process/future.hpp>
#include <process/http.hpp>
#include <process/process.hpp>
#include <process/timeout.hpp>

namespace process {

class Logging : public Process<Logging>
{
public:
  Logging()
    : ProcessBase("logging"),
      original(FLAGS_v)
  {
    // Make sure all reads/writes can be done atomically (i.e., to
    // make sure VLOG(*) statements don't read partial writes).
    // TODO(benh): Use "atomics" primitives for doing reads/writes of
    // FLAGS_v anyway to account for proper memory barriers.
    CHECK(sizeof(FLAGS_v) == sizeof(int32_t));
  }

  virtual ~Logging() {}

protected:
  virtual void initialize()
  {
    route("/toggle", TOGGLE_HELP(), &This::toggle);
  }

private:
  Future<http::Response> toggle(const http::Request& request);

  void set(int v)
  {
    if (FLAGS_v != v) {
      VLOG(FLAGS_v) << "Setting verbose logging level to " << v;
      FLAGS_v = v;
      __sync_synchronize(); // Ensure 'FLAGS_v' visible in other threads.
    }
  }

  void revert()
  {
    if (timeout.remaining() == Seconds(0)) {
      set(original);
    }
  }

  static const std::string TOGGLE_HELP();

  Timeout timeout;

  const int32_t original; // Original value of FLAGS_v.
};

} // namespace process {

#endif // __PROCESS_LOGGING_HPP__
