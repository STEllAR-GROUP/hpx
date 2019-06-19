//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>

#include <string>

namespace hpx { namespace detail {
    HPX_NORETURN void assertion_handler(
        hpx::assertion::source_location const& loc, const char* expr,
        std::string const& msg);
    bool enable_parent_task_handler();
    void test_failure_handler();
#if defined(HPX_HAVE_VERIFY_LOCKS)
    void registered_locks_error_handler();
    bool register_locks_predicate();
#endif
}}    // namespace hpx::detail
