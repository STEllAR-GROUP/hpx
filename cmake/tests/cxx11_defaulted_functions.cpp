////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct foo {
    foo() = delete;
    foo(const foo&) = default;
    foo(foo&&) = default;
    foo& operator=(const foo&) = default;
    foo& operator=(foo&&) = default;
    ~foo() = default;
};

// GCC4.6 doesn't get it quite right
struct bar {
  bar() {}
  bar(bar const&) {};
};

template <typename T>
struct baz : bar {
  baz() {}
  baz(baz const&) = default;
  baz(baz&&) = default; // deleted
};

int main() {
  baz<int> x, y(static_cast<baz<int>&&>(x));
}
