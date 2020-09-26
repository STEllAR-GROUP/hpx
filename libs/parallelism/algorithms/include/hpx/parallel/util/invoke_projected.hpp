//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Pred, typename Proj>
    struct invoke_projected
    {
        typedef typename std::decay<Pred>::type pred_type;
        typedef typename std::decay<Proj>::type proj_type;

        pred_type pred_;
        proj_type proj_;

        template <typename Pred_, typename Proj_>
        invoke_projected(Pred_&& pred, Proj_&& proj)
          : pred_(std::forward<Pred_>(pred))
          , proj_(std::forward<Proj_>(proj))
        {
        }

        template <typename T>
        auto operator()(T&& t) -> decltype(
            HPX_INVOKE(pred_, HPX_INVOKE(proj_, std::forward<T>(t))))
        {
            return HPX_INVOKE(pred_, HPX_INVOKE(proj_, std::forward<T>(t)));
        }
    };
}}}    // namespace hpx::parallel::util
