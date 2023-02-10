//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <exception>
#include <utility>

namespace hpx::detail {

    /// Helper function for a try-catch block where what would normally go in
    /// the catch block should be called after the catch block. This is useful
    /// for situations where the catch-block may yield, since the catch block
    /// should be started and ended on the same worker thread (with yielding and
    /// stealing, the catch block may end on a different worker thread than
    /// where it was started). Because of this, the helper's catch block only
    /// stores the exception pointer, and forwards it outside the catch block.
    ///
    /// Do not replace uses of try_catch_exception_ptr with a plain try-catch
    /// without ensuring that the catch-block can never yield.
    ///
    /// Note: Windows does not seem to have problems resuming a catch block on a
    /// different worker thread, but we use this nonetheless on Windows since it
    /// doesn't hurt.
    template <typename TryCallable, typename CatchCallable>
    HPX_FORCEINLINE decltype(auto) try_catch_exception_ptr(
        TryCallable&& t, CatchCallable&& c)
    {
        std::exception_ptr ep;
        try
        {
            return t();
        }
        catch (...)
        {
            ep = std::current_exception();
        }
        return c(HPX_MOVE(ep));
    }
}    // namespace hpx::detail
// namespace hpx::detail
