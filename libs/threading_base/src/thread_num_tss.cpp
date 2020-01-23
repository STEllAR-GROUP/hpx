//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/detail/thread_num_tss.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace hpx { namespace threads { namespace detail {
    namespace {
        std::size_t& thread_num_tss()
        {
            static thread_local std::size_t thread_num_tss_ = std::size_t(-1);
            return thread_num_tss_;
        }

        thread_pool& thread_pool_tss()
        {
            static thread_local thread_pool thread_pool_tss_ = {
                std::uint16_t(-1), std::uint16_t(-1)};
            return thread_pool_tss_;
        }
    }    // namespace

    // use this to store the global thread number/id in thread local storage
    std::size_t set_thread_num_tss(std::size_t num)
    {
        std::swap(thread_num_tss(), num);
        return num;
    }

    // this returns the globl thread number from thread local storage
    std::size_t get_thread_num_tss()
    {
        return thread_num_tss();
    }

    // set the local thread number and pool index associated with this
    // system thread into thread local storage
    void set_thread_pool_tss(const thread_pool& tup)
    {
        thread_pool_tss() = tup;
    }

    // this returns a struct of the local thread number and the pool
    // Id or index that this thread is assigned to
    thread_pool get_thread_pool_tss()
    {
        return thread_pool_tss();
    }

}}}    // namespace hpx::threads::detail
