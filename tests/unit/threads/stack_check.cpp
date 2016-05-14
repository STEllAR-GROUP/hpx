// Copyright (C) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <stack>

typedef std::tuple<std::size_t, std::ptrdiff_t, std::ptrdiff_t> info;
typedef std::stack<info> info_stack;

void stack_remaining(const char *txt, info_stack &stack) {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
    std::size_t stack_ptr = hpx::threads::coroutines::detail::get_stack_ptr();
#else
    std::size_t stack_ptr = 0x00000000;
#endif
    std::ptrdiff_t stacksize = hpx::this_thread::get_stack_size();
    std::ptrdiff_t remaining_stack = hpx::this_thread::get_available_stack_space();
    //
    std::cout << txt << " stacksize       : 0x" << std::hex << stacksize << "\n";
    std::cout << txt << " stack pointer   : 0x" << std::hex << stack_ptr << "\n";
    std::cout << txt << " stack remaining : 0x" << std::hex << remaining_stack << "\n\n";

    stack.push(std::make_tuple(stack_ptr, stacksize, remaining_stack));
}
//

void stack_waste(int N, info_stack &stack) {
  // declare 1 MB of stack vars
  char bytes[1<<10];
  // prevent the compiler optimizing it away
  std::fill_n(&bytes[0], 32, 0);
  std::stringstream dummy;
  dummy << bytes[45] << std::ends;
  //
  std::stringstream temp;
  temp << "stack_waste " << N;
  stack_remaining(temp.str().c_str(), stack);
  //
  if (N>0) stack_waste(N-1, stack);
}
//

int hpx_main(int argc, char *argv[])
{
    info_stack my_stack_info;
    // just for curiosity
    stack_remaining("hpx_main", my_stack_info);

    // test stack vars
    stack_waste(20, my_stack_info);

    std::ptrdiff_t current_stack = 0;
    while (!my_stack_info.empty()) {
        info i = my_stack_info.top();
        std::ptrdiff_t stack_now = std::get<2>(i);
        std::cout << "stack remaining 0x" << std::hex << stack_now << "\n";
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        HPX_TEST_LT(current_stack, stack_now);
#endif
        current_stack = stack_now;
        my_stack_info.pop();
    }
    return 0;
}

int main(int argc, char *argv[])
{
    //
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
