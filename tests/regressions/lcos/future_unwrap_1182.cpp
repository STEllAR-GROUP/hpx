//  Copyright (c) 2014 Erik Schnetter
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <iostream>
#include <utility>

using namespace hpx;

future<void> nested_future() {
  return make_ready_future();
}

int main() {
  std::cout << "Starting...\n";

  future<future<void> > f1 =
      async(launch::deferred, &nested_future);

  future<void> f2(std::move(f1));
  f2.wait();

  std::cout << "Done.\n";
  return 0;
}
