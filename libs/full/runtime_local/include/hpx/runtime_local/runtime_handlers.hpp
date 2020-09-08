//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <boost/asio/io_service.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace detail {
    HPX_NORETURN void assertion_handler(
        hpx::assertion::source_location const& loc, const char* expr,
        std::string const& msg);
#if defined(HPX_HAVE_APEX)
    bool enable_parent_task_handler();
#endif
    void test_failure_handler();
#if defined(HPX_HAVE_VERIFY_LOCKS)
    void registered_locks_error_handler();
    bool register_locks_predicate();
#endif
    threads::thread_pool_base* get_default_pool();
    threads::mask_cref_type get_pu_mask(
        threads::topology& topo, std::size_t thread_num);
    boost::asio::io_service* get_default_timer_service();
}}    // namespace hpx::detail
