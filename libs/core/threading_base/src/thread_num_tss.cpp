//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#include <cstddef>
#include <tuple>
#include <utility>

namespace hpx::threads::detail {

    namespace {

        HPX_FORCEINLINE std::size_t& global_thread_num() noexcept
        {
            thread_local std::size_t global_thread_num_ =
                static_cast<std::size_t>(-1);
            return global_thread_num_;
        }

        HPX_FORCEINLINE std::size_t& local_thread_num() noexcept
        {
            thread_local std::size_t local_thread_num_ =
                static_cast<std::size_t>(-1);
            return local_thread_num_;
        }

        HPX_FORCEINLINE std::size_t& thread_pool_num() noexcept
        {
            thread_local std::size_t thread_pool_num_ =
                static_cast<std::size_t>(-1);
            return thread_pool_num_;
        }
    }    // namespace

    std::size_t set_global_thread_num_tss(std::size_t num) noexcept
    {
        std::swap(global_thread_num(), num);
        return num;
    }

    std::size_t get_global_thread_num_tss() noexcept
    {
        return global_thread_num();
    }

    std::size_t set_local_thread_num_tss(std::size_t num) noexcept
    {
        std::swap(local_thread_num(), num);
        return num;
    }

    std::size_t get_local_thread_num_tss() noexcept
    {
        return local_thread_num();
    }

    std::size_t set_thread_pool_num_tss(std::size_t num) noexcept
    {
        std::swap(thread_pool_num(), num);
        return num;
    }

    std::size_t get_thread_pool_num_tss() noexcept
    {
        return thread_pool_num();
    }

    void set_thread_nums_tss(thread_nums const& t) noexcept
    {
        global_thread_num() = t.global_thread_num;
        local_thread_num() = t.local_thread_num;
        thread_pool_num() = t.thread_pool_num;
    }

    thread_nums get_thread_nums_tss() noexcept
    {
        return {global_thread_num(), local_thread_num(), thread_pool_num()};
    }
}    // namespace hpx::threads::detail

namespace hpx {

    std::size_t get_worker_thread_num(error_code& /* ec */) noexcept
    {
        return threads::detail::global_thread_num();
    }

    std::size_t get_worker_thread_num() noexcept
    {
        return threads::detail::global_thread_num();
    }

    std::size_t get_local_worker_thread_num(error_code& /* ec */) noexcept
    {
        return threads::detail::local_thread_num();
    }

    std::size_t get_local_worker_thread_num() noexcept
    {
        return threads::detail::local_thread_num();
    }

    std::size_t get_thread_pool_num(error_code& /* ec */) noexcept
    {
        return threads::detail::thread_pool_num();
    }

    std::size_t get_thread_pool_num() noexcept
    {
        return threads::detail::thread_pool_num();
    }
}    // namespace hpx
