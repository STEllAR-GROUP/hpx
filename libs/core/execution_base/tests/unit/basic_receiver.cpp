//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

bool done_called = false;
bool error_called = false;
bool value_called = false;

namespace mylib {
    struct receiver_1
    {
        using is_receiver = void;
        friend void tag_invoke(ex::set_stopped_t, receiver_1&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, receiver_1&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, int) noexcept
        {
            value_called = true;
        }
    };

    struct receiver_2
    {
        using is_receiver = void;
        friend void tag_invoke(ex::set_stopped_t, receiver_2&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, receiver_2&&, int) noexcept
        {
            error_called = true;
        }
    };

    struct receiver_3
    {
        using is_receiver = void;
        friend void tag_invoke(ex::set_stopped_t, receiver_3&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, receiver_3&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_3, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_1
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_1&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_1&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_1, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_2
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_2&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_2&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_2, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_3
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_3&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_3&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_3, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_4
    {
        using is_receiver = void;
        friend void tag_invoke(ex::set_stopped_t, non_receiver_4&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_4&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_4&, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_5
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_5&&)
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_5&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }
    };

    struct non_receiver_6
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_6&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_6&&, std::exception_ptr)
        {
            error_called = true;
        }
    };

    struct non_receiver_7
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_7&&)
        {
            done_called = true;
        }

        friend void tag_invoke(
            ex::set_error_t, non_receiver_7&&, std::exception_ptr)
        {
            error_called = true;
        }
    };
}    // namespace mylib

int main()
{
    using ex::is_receiver;
    using ex::is_receiver_of;

    static_assert(is_receiver<mylib::receiver_1>::value,
        "mylib::receiver_1 should be a receiver");
    static_assert(is_receiver_of<mylib::receiver_1,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::receiver_1 should be a receiver of an int");

    static_assert(!is_receiver_of<mylib::receiver_1,
                      ex::completion_signatures<ex::set_value_t(std::string),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::receiver_1 should not be a receiver of a std::string");

    static_assert(!is_receiver_of<mylib::receiver_2,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::receiver_2 should not be a receiver of int");

    static_assert(is_receiver<mylib::receiver_1>::value,
        "mylib::receiver_1 should be a receiver");
    static_assert(is_receiver_of<mylib::receiver_3,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::receiver_3 should be a receiver of an int");

    static_assert(!is_receiver_of<mylib::receiver_3,
                      ex::completion_signatures<ex::set_value_t(std::string),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::receiver_3 should not be a receiver of a std::string");

    static_assert(!is_receiver<mylib::non_receiver_1>::value,
        "mylib::non_receiver_1 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_1,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_1 should not be a receiver of int");
    static_assert(!is_receiver<mylib::non_receiver_2>::value,
        "mylib::non_receiver_2 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_2,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_2 should not be a receiver of int");
    static_assert(!is_receiver<mylib::non_receiver_3>::value,
        "mylib::non_receiver_3 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_3,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_3 should not be a receiver of int");
    static_assert(is_receiver<mylib::non_receiver_4>::value,
        "mylib::non_receiver_4 should be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_4,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_4 should not be a receiver of int");
    static_assert(!is_receiver<mylib::non_receiver_5>::value,
        "mylib::non_receiver_5 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_5,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_5 should not be a receiver of int");
    static_assert(!is_receiver<mylib::non_receiver_6>::value,
        "mylib::non_receiver_6 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_6,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_6 should not be a receiver of int");
    static_assert(!is_receiver<mylib::non_receiver_7>::value,
        "mylib::non_receiver_7 should not be a receiver");
    static_assert(!is_receiver_of<mylib::non_receiver_7,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr),
                          ex::set_stopped_t()>>::value,
        "mylib::non_receiver_7 should not be a receiver of int");

    {
        mylib::receiver_1 rcv;
        ex::set_stopped(std::move(rcv));
        HPX_TEST(done_called);
        done_called = false;
    }
    {
        mylib::receiver_1 rcv;
        ex::set_error(std::move(rcv), std::exception_ptr{});
        HPX_TEST(error_called);
        error_called = false;
    }
    {
        mylib::receiver_1 rcv;
        ex::set_value(std::move(rcv), 1);
        HPX_TEST(value_called);
        value_called = false;
    }
    {
        mylib::receiver_2 rcv;
        ex::set_stopped(std::move(rcv));
        HPX_TEST(done_called);
        done_called = false;
    }
    {
        mylib::receiver_2 rcv;
        ex::set_error(std::move(rcv), 1);
        HPX_TEST(error_called);
        error_called = false;
    }

    return hpx::util::report_errors();
}
