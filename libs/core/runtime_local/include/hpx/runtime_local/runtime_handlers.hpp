//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/local/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <asio/io_context.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace detail {
    HPX_NORETURN HPX_LOCAL_EXPORT void assertion_handler(
        hpx::assertion::source_location const& loc, const char* expr,
        std::string const& msg);
#if defined(HPX_HAVE_APEX)
    HPX_LOCAL_EXPORT bool enable_parent_task_handler();
#endif
    HPX_LOCAL_EXPORT void test_failure_handler();
#if defined(HPX_HAVE_VERIFY_LOCKS)
    HPX_LOCAL_EXPORT void registered_locks_error_handler();
    HPX_LOCAL_EXPORT bool register_locks_predicate();
#endif
    HPX_LOCAL_EXPORT threads::thread_pool_base* get_default_pool();
    HPX_LOCAL_EXPORT threads::mask_cref_type get_pu_mask(
        threads::topology& topo, std::size_t thread_num);
    HPX_LOCAL_EXPORT asio::io_context* get_default_timer_service();
}}    // namespace hpx::detail
