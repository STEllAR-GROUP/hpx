//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/unused.hpp>

namespace hpx { namespace traits {

    /// \cond NOINTERNAL
    namespace detail {
        template <typename Result, typename Enable = void>
        struct action_remote_result_customization_point
        {
            using type = Result;
        };

        // If an action returns void, we need to do special things
        template <>
        struct action_remote_result_customization_point<void>
        {
            using type = util::unused_type;
        };
    }    // namespace detail
    /// \endcond

    template <typename Result>
    struct action_remote_result
      : detail::action_remote_result_customization_point<Result>
    {
    };

    template <typename Result>
    using action_remote_result_t = typename action_remote_result<Result>::type;
}}    // namespace hpx::traits
