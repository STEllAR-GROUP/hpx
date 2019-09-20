//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ASSERT_EVALUATE_ASSERT_HPP
#define HPX_ASSERT_EVALUATE_ASSERT_HPP

#include <hpx/config.hpp>

#include <hpx/assertion/source_location.hpp>

#include <string>
#include <utility>

namespace hpx { namespace assertion { namespace detail {
    /// \cond NOINTERNAL
    HPX_EXPORT void handle_assert(
        source_location const& loc, const char* expr, std::string const& msg);

    template <typename Expr, typename Msg>
    HPX_FORCEINLINE void evaluate_assert(Expr const& expr,
        source_location const& loc, const char* expr_string, Msg const& msg)
    {
        if (!expr())
        {
            handle_assert(loc, expr_string, msg());
        }
    }
    /// \endcond
}}}    // namespace hpx::assertion::detail

#endif
