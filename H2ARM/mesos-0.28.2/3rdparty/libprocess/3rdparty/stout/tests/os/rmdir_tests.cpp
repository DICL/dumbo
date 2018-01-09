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

#include <list>
#include <map>
#include <set>
#include <string>

#include <stout/path.hpp>

#include <stout/os/getcwd.hpp>
#include <stout/os/ls.hpp>
#include <stout/os/mkdir.hpp>
#include <stout/os/touch.hpp>

#include <stout/tests/utils.hpp>

using std::list;
using std::set;
using std::string;


const hashset<string>& EMPTY = hashset<string>::EMPTY;


static hashset<string> listfiles(const string& directory)
{
  hashset<string> fileset;
  Try<list<string>> entries = os::ls(directory);
  if (entries.isSome()) {
    foreach (const string& entry, entries.get()) {
      fileset.insert(entry);
    }
  }
  return fileset;
}


class RmdirTest : public TemporaryDirectoryTest {};


// TODO(hausdorff): This test is almost copy-pasted from
// `TrivialRemoveEmptyDirectoryRelativePath`; we should parameterize them to
// reduce redundancy.
TEST_F(RmdirTest, TrivialRemoveEmptyDirectoryAbsolutePath)
{
  const string tmpdir = os::getcwd();
  hashset<string> expectedListing = EMPTY;

  // Directory is initially empty.
  EXPECT_EQ(expectedListing, listfiles(tmpdir));

  // Successfully make directory using absolute path.
  const string newDirectoryName = "newDirectory";
  const string newDirectoryAbsolutePath = path::join(tmpdir, newDirectoryName);
  expectedListing.insert(newDirectoryName);
  EXPECT_SOME(os::mkdir(newDirectoryAbsolutePath));
  EXPECT_EQ(expectedListing, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryAbsolutePath));

  // Successfully remove.
  EXPECT_SOME(os::rmdir(newDirectoryAbsolutePath));
  EXPECT_EQ(EMPTY, listfiles(tmpdir));
}


TEST_F(RmdirTest, TrivialRemoveEmptyDirectoryRelativePath)
{
  const string tmpdir = os::getcwd();
  hashset<string> expectedListing = EMPTY;

  // Directory is initially empty.
  EXPECT_EQ(expectedListing, listfiles(tmpdir));

  // Successfully make directory using relative path.
  const string newDirectoryName = "newDirectory";
  expectedListing.insert(newDirectoryName);
  EXPECT_SOME(os::mkdir(newDirectoryName));
  EXPECT_EQ(expectedListing, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryName));

  // Successfully remove.
  EXPECT_SOME(os::rmdir(newDirectoryName));
  EXPECT_EQ(EMPTY, listfiles(tmpdir));
}


TEST_F(RmdirTest, RemoveRecursiveByDefault)
{
  const string tmpdir = os::getcwd();
  hashset<string> expectedRootListing = EMPTY;
  hashset<string> expectedSubListing = EMPTY;

  // Directory is initially empty.
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));

  // Successfully make directory using absolute path, and then `touch` a file
  // in that folder.
  const string newDirectoryName = "newDirectory";
  const string newDirectoryAbsolutePath = path::join(tmpdir, newDirectoryName);
  const string newFileName = "newFile";
  const string newFileAbsolutePath = path::join(
      newDirectoryAbsolutePath,
      newFileName);

  expectedRootListing.insert(newDirectoryName);
  expectedSubListing.insert(newFileName);

  EXPECT_SOME(os::mkdir(newDirectoryAbsolutePath));
  EXPECT_SOME(os::touch(newFileAbsolutePath));
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));
  EXPECT_EQ(expectedSubListing, listfiles(newDirectoryAbsolutePath));

  // Successfully remove.
  EXPECT_SOME(os::rmdir(newDirectoryAbsolutePath));
  EXPECT_EQ(EMPTY, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryAbsolutePath));
}


TEST_F(RmdirTest, TrivialFailToRemoveInvalidPath)
{
  // Directory is initially empty.
  EXPECT_EQ(EMPTY, listfiles(os::getcwd()));

  // Removing fake relative paths should error out.
  EXPECT_ERROR(os::rmdir("fakeRelativePath", false));
  EXPECT_ERROR(os::rmdir("fakeRelativePath", true));

  // Directory still empty.
  EXPECT_EQ(EMPTY, listfiles(os::getcwd()));
}


TEST_F(RmdirTest, FailTRemoveNestedInvalidPath)
{
  const string tmpdir = os::getcwd();
  hashset<string> expectedRootListing = EMPTY;

  // Directory is initially empty.
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));

  // Successfully make directory using absolute path.
  const string newDirectoryName = "newDirectory";
  const string newDirectoryAbsolutePath = path::join(tmpdir, newDirectoryName);

  expectedRootListing.insert(newDirectoryName);

  EXPECT_SOME(os::mkdir(newDirectoryAbsolutePath));
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryAbsolutePath));

  // Fail to remove a path to an invalid folder inside the
  // `newDirectoryAbsolutePath`.
  const string fakeAbsolutePath = path::join(newDirectoryAbsolutePath, "fake");
  EXPECT_ERROR(os::rmdir(fakeAbsolutePath, false));
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryAbsolutePath));

  // Test the same thing, but using the `recursive` flag.
  EXPECT_ERROR(os::rmdir(fakeAbsolutePath, true));
  EXPECT_EQ(expectedRootListing, listfiles(tmpdir));
  EXPECT_EQ(EMPTY, listfiles(newDirectoryAbsolutePath));
}
