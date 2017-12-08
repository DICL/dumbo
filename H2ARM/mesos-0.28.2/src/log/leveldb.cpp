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

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <glog/logging.h>

#include <leveldb/comparator.h>
#include <leveldb/write_batch.h>

#include <stdint.h>

#include <stout/check.hpp>
#include <stout/error.hpp>
#include <stout/numify.hpp>
#include <stout/stopwatch.hpp>
#include <stout/strings.hpp>
#include <stout/unreachable.hpp>

#include "log/leveldb.hpp"

using std::string;

namespace mesos {
namespace internal {
namespace log {

class Varint64Comparator : public leveldb::Comparator
{
public:
  virtual int Compare(
      const leveldb::Slice& a,
      const leveldb::Slice& b) const
  {
    // TODO(benh): Use varint comparator.
    LOG(FATAL) << "Unimplemented";
    // uint64_t left = position(a);
    // uint64_t right = position(b);
    // if (left < right) return -1;
    // if (left == right) return 0;
    // if (left > right) return 1;
    UNREACHABLE();
  }

  virtual const char* Name() const
  {
    // Note that this name MUST NOT CHANGE across uses of this
    // comparator with the same DB (the semantics of doing so are
    // undefined if the database doesn't catch this first).
    return "varint64";
  }

  virtual void FindShortestSeparator(
      string* start,
      const leveldb::Slice& limit) const
  {
    // Intentional no-op.
  }

  virtual void FindShortSuccessor(string* key) const
  {
    // Intentional no-op.
  }
};


// TODO(benh): Use varint comparator.
// static Varint64Comparator comparator;


// Returns a string representing the specified position. Note that we
// adjust the actual position by incrementing it by 1 because we
// reserve 0 for storing the promise record (Record::Promise,
// DEPRECATED!), or the metadata (Record::Metadata).
static string encode(uint64_t position, bool adjust = true)
{
  // Adjusted stringified represenation is plus 1 of actual position.
  position = adjust ? position + 1 : position;

  // TODO(benh): Use varint encoding for VarInt64Comparator!
  // string s;
  // google::protobuf::io::StringOutputStream _stream(&s);
  // google::protobuf::io::CodedOutputStream stream(&_stream);
  // position = adjust ? position + 1 : position;
  // stream.WriteVarint64(position);
  // return s;

  Try<string> s = strings::format("%.*d", 10, position);
  CHECK_SOME(s);
  return s.get();
}


// Returns the position as represented in the specified slice
// (performing a decrement as necessary to determine the actual
// position represented).
// TODO(jieyu): This function is not used (see RB-18252). However, we
// still want to keep this function in case we need it in the future.
// We comment it out to silence the warning (unused static function)
// from the compiler.
// static uint64_t decode(const leveldb::Slice& s)
// {
//   // TODO(benh): Use varint decoding for VarInt64Comparator!
//   // uint64_t position;
//   // google::protobuf::io::ArrayInputStream _stream(s.data(), s.size());
//   // google::protobuf::io::CodedInputStream stream(&_stream);
//   // bool success = stream.ReadVarint64(&position);
//   // CHECK(success);
//   // return position - 1; // Actual position is less 1 of stringified.
//   Try<uint64_t> position = numify<uint64_t>(string(s.data(), s.size()));
//   CHECK_SOME(position);
//   return position.get() - 1; // Actual position is less 1 of stringified.
// }


LevelDBStorage::LevelDBStorage()
  : db(NULL), first(None())
{
  // Nothing to see here.
}


LevelDBStorage::~LevelDBStorage()
{
  delete db; // Might be null if open failed in LevelDBStorage::restore.
}


Try<Storage::State> LevelDBStorage::restore(const string& path)
{
  leveldb::Options options;
  options.create_if_missing = true;

  // TODO(benh): Can't use varint comparator until bug discussed at
  // groups.google.com/group/leveldb/browse_thread/thread/17eac39168909ba7
  // gets fixed. For now, we are using the default byte-wise
  // comparator and *assuming* that the encoding from unsigned long to
  // string produces a stable ordering. Checks below.
  // options.comparator = &comparator;

  const string& one = encode(1);
  const string& two = encode(2);
  const string& ten = encode(10);

  CHECK(leveldb::BytewiseComparator()->Compare(one, two) < 0);
  CHECK(leveldb::BytewiseComparator()->Compare(two, one) > 0);
  CHECK(leveldb::BytewiseComparator()->Compare(one, ten) < 0);
  CHECK(leveldb::BytewiseComparator()->Compare(ten, two) > 0);
  CHECK(leveldb::BytewiseComparator()->Compare(ten, ten) == 0);

  Stopwatch stopwatch;
  stopwatch.start();

  leveldb::Status status = leveldb::DB::Open(options, path, &db);

  if (!status.ok()) {
    // TODO(benh): Consider trying to repair the DB.
    return Error(status.ToString());
  }

  LOG(INFO) << "Opened db in " << stopwatch.elapsed();

  stopwatch.start(); // Restart the stopwatch.

  // TODO(benh): Conditionally compact to avoid long recovery times?
  db->CompactRange(NULL, NULL);

  LOG(INFO) << "Compacted db in " << stopwatch.elapsed();

  State state;
  state.begin = 0;
  state.end = 0;

  // TODO(benh): Consider just reading the "promise" record (e.g.,
  // 'encode(0, false)') and then iterating over the rest of the
  // records and confirming that they are all indeed of type
  // Record::Action.

  stopwatch.start(); // Restart the stopwatch.

  leveldb::Iterator* iterator = db->NewIterator(leveldb::ReadOptions());

  LOG(INFO) << "Created db iterator in " << stopwatch.elapsed();

  stopwatch.start(); // Restart the stopwatch.

  iterator->SeekToFirst();

  LOG(INFO) << "Seeked to beginning of db in " << stopwatch.elapsed();

  stopwatch.start(); // Restart the stopwatch.

  uint64_t keys = 0;

  while (iterator->Valid()) {
    keys++;
    const leveldb::Slice& slice = iterator->value();

    google::protobuf::io::ArrayInputStream stream(slice.data(), slice.size());

    Record record;

    if (!record.ParseFromZeroCopyStream(&stream)) {
      return Error("Failed to deserialize record");
    }

    switch (record.type()) {
      case Record::METADATA: {
        CHECK(record.has_metadata());
        state.metadata.CopyFrom(record.metadata());
        break;
      }

      // DEPRECATED!
      case Record::PROMISE: {
        CHECK(record.has_promise());
        // This replica is in old format. Set its status to VOTING
        // since there is no catch-up logic in the old code and this
        // replica is obviously not empty.
        state.metadata.set_status(Metadata::VOTING);
        state.metadata.set_promised(record.promise().proposal());
        break;
      }

      case Record::ACTION: {
        CHECK(record.has_action());
        const Action& action = record.action();
        if (action.has_learned() && action.learned()) {
          state.learned.insert(action.position());
          state.unlearned.erase(action.position());
          if (action.has_type() && action.type() == Action::TRUNCATE) {
            state.begin = std::max(state.begin, action.truncate().to());
          }
        } else {
          state.learned.erase(action.position());
          state.unlearned.insert(action.position());
        }
        state.end = std::max(state.end, action.position());

        // Cache the first position in this replica so during a
        // truncation, we can attempt to delete all positions from the
        // first position up to the truncate position. Note that this
        // is not the beginning position of the log, but rather the
        // first position that remains (i.e., hasn't been deleted) in
        // leveldb.
        first = min(first, action.position());
        break;
      }

      default: {
        return Error("Bad record");
      }
    }

    iterator->Next();
  }

  LOG(INFO) << "Iterated through " << keys
            << " keys in the db in " << stopwatch.elapsed();

  delete iterator;

  return state;
}


Try<Nothing> LevelDBStorage::persist(const Metadata& metadata)
{
  Stopwatch stopwatch;
  stopwatch.start();

  leveldb::WriteOptions options;
  options.sync = true;

  Record record;
  record.set_type(Record::METADATA);
  record.mutable_metadata()->CopyFrom(metadata);

  string value;

  if (!record.SerializeToString(&value)) {
    return Error("Failed to serialize record");
  }

  leveldb::Status status = db->Put(options, encode(0, false), value);

  if (!status.ok()) {
    return Error(status.ToString());
  }

  LOG(INFO) << "Persisting metadata (" << value.size()
            << " bytes) to leveldb took " << stopwatch.elapsed();

  return Nothing();
}


Try<Nothing> LevelDBStorage::persist(const Action& action)
{
  Stopwatch stopwatch;
  stopwatch.start();

  Record record;
  record.set_type(Record::ACTION);
  record.mutable_action()->MergeFrom(action);

  string value;

  if (!record.SerializeToString(&value)) {
    return Error("Failed to serialize record");
  }

  leveldb::WriteOptions options;
  options.sync = true;

  leveldb::Status status = db->Put(options, encode(action.position()), value);

  if (!status.ok()) {
    return Error(status.ToString());
  }

  // Updated the first position. Notice that we use 'min' here instead
  // of checking 'isNone()' because it's likely that log entries are
  // written out of order during catch-up (e.g. if a random bulk
  // catch-up policy is used).
  first = min(first, action.position());

  LOG(INFO) << "Persisting action (" << value.size()
            << " bytes) to leveldb took " << stopwatch.elapsed();

  // Delete positions if a truncate action has been *learned*. Note
  // that we do this in a best-effort fashion (i.e., we ignore any
  // failures to the database since we can always try again).
  if (action.has_type() && action.type() == Action::TRUNCATE &&
      action.has_learned() && action.learned()) {
    CHECK(action.has_truncate());

    stopwatch.start(); // Restart the stopwatch.

    // To actually perform the truncation in leveldb we need to remove
    // all the keys that represent positions no longer in the log. We
    // do this by attempting to delete all keys that represent the
    // first position we know is still in leveldb up to (but
    // excluding) the truncate position. Note that this works because
    // the semantics of WriteBatch are such that even if the position
    // doesn't exist (which is possible because this replica has some
    // holes), we can attempt to delete the key that represents it and
    // it will just ignore that key. This is *much* cheaper than
    // actually iterating through the entire database instead (which
    // was, for posterity, the original implementation). In addition,
    // caching the "first" position we know is in the database is
    // cheaper than using an iterator to determine the first position
    // (which was, for posterity, the second implementation).

    leveldb::WriteBatch batch;

    CHECK_SOME(first);

    // Add positions up to (but excluding) the truncate position to
    // the batch starting at the first position still in leveldb. It's
    // likely that the first position is greater than the truncate
    // position (e.g., during catch-up). In that case, we do nothing
    // because there is nothing we can truncate.
    // TODO(jieyu): We might miss a truncation if we do random (i.e.,
    // out of order) bulk catch-up and the truncate operation is
    // caught up first.
    uint64_t index = 0;
    while ((first.get() + index) < action.truncate().to()) {
      batch.Delete(encode(first.get() + index));
      index++;
    }

    // If we added any positions, attempt to delete them!
    if (index > 0) {
      // We do this write asynchronously (e.g., using default options).
      leveldb::Status status = db->Write(leveldb::WriteOptions(), &batch);

      if (!status.ok()) {
        LOG(WARNING) << "Ignoring leveldb batch delete failure: "
                     << status.ToString();
      } else {
        // Save the new first position!
        CHECK_LT(first.get(), action.truncate().to());
        first = action.truncate().to();

        LOG(INFO) << "Deleting ~" << index
                  << " keys from leveldb took " << stopwatch.elapsed();
      }
    }
  }

  return Nothing();
}


Try<Action> LevelDBStorage::read(uint64_t position)
{
  Stopwatch stopwatch;
  stopwatch.start();

  leveldb::ReadOptions options;

  string value;

  leveldb::Status status = db->Get(options, encode(position), &value);

  if (!status.ok()) {
    return Error(status.ToString());
  }

  google::protobuf::io::ArrayInputStream stream(value.data(), value.size());

  Record record;

  if (!record.ParseFromZeroCopyStream(&stream)) {
    return Error("Failed to deserialize record");
  }

  if (record.type() != Record::ACTION) {
    return Error("Bad record");
  }

  LOG(INFO) << "Reading position from leveldb took " << stopwatch.elapsed();

  return record.action();
}

} // namespace log {
} // namespace internal {
} // namespace mesos {
