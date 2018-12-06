//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <exception>

std::atomic<std::size_t> count_error_handler(0);

///////////////////////////////////////////////////////////////////////////////
bool on_thread_error(std::size_t num, std::exception_ptr const& e)
{
    ++count_error_handler;
    return false;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    HPX_THROW_EXCEPTION(hpx::invalid_status, "test", "test");
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    auto on_stop = hpx::register_thread_on_error_func(&on_thread_error);
    HPX_TEST(on_stop.empty());

    bool caught_exception = false;
    try
    {
        hpx::init(argc, argv);
        HPX_TEST(false);
    }
    catch (...)
    {
        caught_exception = true;
    }

    HPX_TEST(caught_exception);
    HPX_TEST(count_error_handler == 1);

    return hpx::util::report_errors();
}
