//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace hpx { namespace threads { namespace detail {
    namespace {
        thread_nums& thread_nums_tss()
        {
            static thread_local thread_nums thread_nums_tss_ = {
                std::size_t(-1), std::size_t(-1), std::size_t(-1)};
            return thread_nums_tss_;
        }
    }    // namespace

    std::size_t set_global_thread_num_tss(std::size_t num)
    {
        std::swap(thread_nums_tss().global_thread_num, num);
        return num;
    }

    std::size_t get_global_thread_num_tss()
    {
        return thread_nums_tss().global_thread_num;
    }

    std::size_t set_local_thread_num_tss(std::size_t num)
    {
        std::swap(thread_nums_tss().local_thread_num, num);
        return num;
    }

    std::size_t get_local_thread_num_tss()
    {
        return thread_nums_tss().local_thread_num;
    }

    std::size_t set_thread_pool_num_tss(std::size_t num)
    {
        std::swap(thread_nums_tss().thread_pool_num, num);
        return num;
    }

    std::size_t get_thread_pool_num_tss()
    {
        return thread_nums_tss().thread_pool_num;
    }

    void set_thread_nums_tss(const thread_nums& t)
    {
        thread_nums_tss() = t;
    }

    thread_nums get_thread_nums_tss()
    {
        return thread_nums_tss();
    }

}}}    // namespace hpx::threads::detail

namespace hpx {
    std::size_t get_worker_thread_num(error_code& /* ec */)
    {
        return threads::detail::thread_nums_tss().global_thread_num;
    }

    std::size_t get_worker_thread_num()
    {
        return get_worker_thread_num(throws);
    }

    std::size_t get_local_worker_thread_num(error_code& /* ec */)
    {
        return threads::detail::thread_nums_tss().local_thread_num;
    }

    std::size_t get_local_worker_thread_num()
    {
        return get_local_worker_thread_num(throws);
    }

    std::size_t get_thread_pool_num(error_code& /* ec */)
    {
        return threads::detail::thread_nums_tss().thread_pool_num;
    }

    std::size_t get_thread_pool_num()
    {
        return get_thread_pool_num(throws);
    }
}    // namespace hpx
