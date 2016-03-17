//  Copyright (c) 2016 Hadrian G. (a.k.a. Neolander)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This compile-only test case verifies that #2035 remains fixed

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <hpx/hpx_main.hpp>
#include <hpx/include/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <array>

int main()
{
   std::array<hpx::shared_future<int>, 1> future_array { hpx::make_ready_future(0) };
   hpx::wait_all(future_array.cbegin(), future_array.cend());

   return 0;
}
#endif
