//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/iostreams for documentation.
//
// File:        hpx/iostream/detail/execute.hpp
// Date:        Thu Dec 06 13:21:54 MST 2007
// Copyright:   2007-2008 CodeRage, LLC
// Author:      Jonathan Turkanis
// Contact:     turkanis at coderage dot com

// Defines the overloaded function template
// hpx::iostream::detail::execute_all() and the function template
// hpx::iostream::detail::execute_foreach().
//
// execute_all() invokes a primary operation and performs a sequence of cleanup
// operations, returning the result of the primary operation if no exceptions
// are thrown. If one of the operations throws an exception, performs the
// remaining operations and rethrows the initial exception.
//
// execute_foreach() is a variant of std::foreach which invokes a function
// object for each item in a sequence, catching all exceptions and rethrowing
// the first caught exception after the function object has been invoked on each
// item.

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/functional.hpp>

#include <type_traits>

namespace hpx::iostream::detail {

    ///////////////////////////////////////////////////////////////////////////
    // Implementation with one or more cleanup operations
    HPX_CXX_CORE_EXPORT void constexpr execute_all_helper(bool&) noexcept {}

    HPX_CXX_CORE_EXPORT template <typename Op, typename... Ops>
    decltype(auto) execute_all_helper(
        bool& exception_thrown, Op&& op, Ops&&... ops)
    {
        try
        {
            auto _ = hpx::experimental::scope_success([&]() {
                execute_all_helper(exception_thrown, HPX_FORWARD(Ops, ops)...);
            });
            return HPX_FORWARD(Op, op)();
        }
        catch (...)
        {
            if (!exception_thrown)
            {
                try
                {
                    execute_all_helper(
                        exception_thrown, HPX_FORWARD(Ops, ops)...);
                }
                // NOLINTNEXTLINE(bugprone-empty-catch)
                catch (...)
                {
                }
                exception_thrown = true;
            }
            throw;
        }
    }

    HPX_CXX_CORE_EXPORT template <typename... Ops>
    decltype(auto) execute_all(Ops&&... ops)
    {
        bool exception_thrown = false;
        return execute_all_helper(exception_thrown, HPX_FORWARD(Ops, ops)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename InIt, typename Op>
    Op execute_foreach(InIt first, InIt last, Op&& op)
    {
        if (first == last)
            return HPX_FORWARD(Op, op);

        try
        {
            op(*first);
        }
        catch (...)
        {
            try
            {
                ++first;
                execute_foreach(first, last, HPX_FORWARD(Op, op));
            }
            // NOLINTNEXTLINE(bugprone-empty-catch)
            catch (...)
            {
            }
            throw;
        }
        ++first;
        return execute_foreach(first, last, HPX_FORWARD(Op, op));
    }
}    // namespace hpx::iostream::detail
