//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <utility>

namespace hpx { namespace traits {
    template <typename Result, typename RemoteResult, typename Enable = void>
    struct get_remote_result
    {
        static Result call(RemoteResult const& rhs)
        {
            return Result(rhs);
        }

        static Result call(RemoteResult&& rhs)    //-V659
        {
            return Result(std::move(rhs));
        }
    };

    template <typename Result>
    struct get_remote_result<Result, Result>
    {
        static Result const& call(Result const& rhs)
        {
            return rhs;
        }

        static Result&& call(Result&& rhs)
        {
            return std::move(rhs);
        }
    };
}}    // namespace hpx::traits
