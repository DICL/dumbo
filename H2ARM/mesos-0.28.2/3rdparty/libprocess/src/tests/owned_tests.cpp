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

#include <gmock/gmock.h>

#include <process/gtest.hpp>
#include <process/owned.hpp>
#include <process/shared.hpp>

using process::Owned;
using process::Shared;

class Foo
{
public:
  int get() const { return value; }
  void set(int _value) { value = _value; }

private:
  int value;
};


TEST(OwnedTest, Access)
{
  Foo* foo = new Foo();
  foo->set(42);

  Owned<Foo> owned(foo);

  EXPECT_EQ(42, owned->get());
  EXPECT_EQ(42, (*owned).get());
  EXPECT_EQ(42, owned.get()->get());

  owned->set(10);

  EXPECT_EQ(10, owned->get());
  EXPECT_EQ(10, (*owned).get());
  EXPECT_EQ(10, owned.get()->get());
}


TEST(OwnedTest, Null)
{
  Owned<Foo> owned;
  Owned<Foo> owned2(NULL);

  EXPECT_EQ(NULL, owned.get());
  EXPECT_EQ(NULL, owned2.get());
}


TEST(OwnedTest, Share)
{
  Foo* foo = new Foo();
  foo->set(42);

  Owned<Foo> owned(foo);

  EXPECT_EQ(42, owned->get());
  EXPECT_EQ(42, (*owned).get());
  EXPECT_EQ(42, owned.get()->get());

  Shared<Foo> shared = owned.share();

  EXPECT_EQ(NULL, owned.get());
  EXPECT_TRUE(shared.unique());

  EXPECT_EQ(42, shared->get());
  EXPECT_EQ(42, (*shared).get());
  EXPECT_EQ(42, shared.get()->get());

  {
    Shared<Foo> shared2(shared);

    EXPECT_EQ(42, shared2->get());
    EXPECT_EQ(42, (*shared2).get());
    EXPECT_EQ(42, shared2.get()->get());
    EXPECT_FALSE(shared.unique());
    EXPECT_FALSE(shared2.unique());
  }

  EXPECT_TRUE(shared.unique());
}


TEST(OwnedTest, Release)
{
  Foo* foo = new Foo();
  foo->set(42);

  Owned<Foo> owned(foo);

  EXPECT_EQ(42, owned->get());
  EXPECT_EQ(42, (*owned).get());
  EXPECT_EQ(42, owned.get()->get());

  Foo* raw = owned.release();
  EXPECT_EQ(NULL, owned.get());
  EXPECT_EQ(42, raw->get());

  delete raw;
  EXPECT_EQ(NULL, owned.get());
}
