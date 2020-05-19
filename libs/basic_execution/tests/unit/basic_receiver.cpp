//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/basic_execution/receiver.hpp>
#include <hpx/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

bool done_called = false;
bool error_called = false;
bool value_called = false;

namespace mylib {
    struct receiver_1
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) noexcept
        {
            error_called = true;
        }

        void set_value(int)
        {
            value_called = true;
        }
    };

    struct receiver_2
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(int) noexcept
        {
            error_called = true;
        }
    };

    struct receiver_3
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) noexcept
        {
            error_called = true;
        }

        void set_value(int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_1
    {
        void set_done() & noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) noexcept
        {
            error_called = true;
        }

        void set_value(int)
        {
            value_called = true;
        }
    };

    struct non_receiver_2
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) & noexcept
        {
            error_called = true;
        }

        void set_value(int)
        {
            value_called = true;
        }
    };

    struct non_receiver_3
    {
        void set_done() & noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) & noexcept
        {
            error_called = true;
        }

        void set_value(int)
        {
            value_called = true;
        }
    };

    struct non_receiver_4
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) noexcept
        {
            error_called = true;
        }

        void set_value(int) &
        {
            value_called = true;
        }
    };

    struct non_receiver_5
    {
        void set_done()
        {
            done_called = true;
        }

        void set_error(std::exception_ptr) noexcept
        {
            error_called = true;
        }
    };

    struct non_receiver_6
    {
        void set_done() noexcept
        {
            done_called = true;
        }

        void set_error(std::exception_ptr)
        {
            error_called = true;
        }
    };

    struct non_receiver_7
    {
        void set_done()
        {
            done_called = true;
        }

        void set_error(std::exception_ptr)
        {
            error_called = true;
        }
    };
}    // namespace mylib

int main()
{
    static_assert(hpx::basic_execution::is_receiver<mylib::receiver_1>::value,
        "mylib::receiver_1 should be a receiver");
    static_assert(
        hpx::basic_execution::is_receiver_of<mylib::receiver_1, int>::value,
        "mylib::receiver_1 should be a receiver of an int");
    static_assert(
        !hpx::basic_execution::is_nothrow_receiver_of<mylib::receiver_1,
            int>::value,
        "mylib::receiver_1 should not be a nothrow receiver of an int");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::receiver_1,
                      std::string>::value,
        "mylib::receiver_1 should not be a receiver of a std::string");

    static_assert(!hpx::basic_execution::is_receiver<mylib::receiver_2>::value,
        "mylib::receiver_2 should not be a receiver of std::exception_ptr");
    static_assert(
        hpx::basic_execution::is_receiver<mylib::receiver_2, int>::value,
        "mylib::receiver_2 should be a receiver");
    static_assert(
        !hpx::basic_execution::is_receiver_of<mylib::receiver_2, int>::value,
        "mylib::receiver_2 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_nothrow_receiver_of<mylib::receiver_2,
            int>::value,
        "mylib::receiver_2 should not be a nothrow receiver of int");

    static_assert(hpx::basic_execution::is_receiver<mylib::receiver_1>::value,
        "mylib::receiver_1 should be a receiver");
    static_assert(
        hpx::basic_execution::is_receiver_of<mylib::receiver_3, int>::value,
        "mylib::receiver_3 should be a receiver of an int");
    static_assert(
        hpx::basic_execution::is_nothrow_receiver_of<mylib::receiver_3,
            int>::value,
        "mylib::receiver_3 should be a nothrow receiver of an int");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::receiver_3,
                      std::string>::value,
        "mylib::receiver_3 should not be a receiver of a std::string");

    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_1>::value,
        "mylib::non_receiver_1 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_1,
                      int>::value,
        "mylib::non_receiver_1 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_2>::value,
        "mylib::non_receiver_2 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_2,
                      int>::value,
        "mylib::non_receiver_2 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_3>::value,
        "mylib::non_receiver_3 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_3,
                      int>::value,
        "mylib::non_receiver_3 should not be a receiver of int");
    static_assert(
        hpx::basic_execution::is_receiver<mylib::non_receiver_4>::value,
        "mylib::non_receiver_4 should be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_4,
                      int>::value,
        "mylib::non_receiver_4 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_5>::value,
        "mylib::non_receiver_5 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_5,
                      int>::value,
        "mylib::non_receiver_5 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_6>::value,
        "mylib::non_receiver_6 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_6,
                      int>::value,
        "mylib::non_receiver_6 should not be a receiver of int");
    static_assert(
        !hpx::basic_execution::is_receiver<mylib::non_receiver_7>::value,
        "mylib::non_receiver_7 should not be a receiver");
    static_assert(!hpx::basic_execution::is_receiver_of<mylib::non_receiver_7,
                      int>::value,
        "mylib::non_receiver_7 should not be a receiver of int");

    {
        mylib::receiver_1 rcv;
        hpx::basic_execution::set_done(std::move(rcv));
        HPX_TEST(done_called);
        done_called = false;
        hpx::basic_execution::set_error(std::move(rcv), std::exception_ptr{});
        HPX_TEST(error_called);
        error_called = false;
        hpx::basic_execution::set_value(std::move(rcv), 1);
        HPX_TEST(value_called);
        value_called = false;
    }
    {
        mylib::receiver_2 rcv;
        hpx::basic_execution::set_done(std::move(rcv));
        HPX_TEST(done_called);
        done_called = false;
        hpx::basic_execution::set_error(std::move(rcv), 1);
        HPX_TEST(error_called);
        error_called = false;
    }

    return hpx::util::report_errors();
}
