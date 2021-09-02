//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Pred, typename Proj>
    struct invoke_projected
    {
        using pred_type = typename std::decay<Pred>::type;
        using proj_type = typename std::decay<Proj>::type;

        pred_type pred_;
        proj_type proj_;

        template <typename Pred_, typename Proj_>
        invoke_projected(Pred_&& pred, Proj_&& proj)
          : pred_(std::forward<Pred_>(pred))
          , proj_(std::forward<Proj_>(proj))
        {
        }

        template <typename T>
        decltype(auto) operator()(T&& t)
        {
            return HPX_INVOKE(pred_, HPX_INVOKE(proj_, std::forward<T>(t)));
        }

        template <typename T>
        decltype(auto) operator()(T&& t, T&& u)
        {
            return HPX_INVOKE(pred_, HPX_INVOKE(proj_, std::forward<T>(t)),
                HPX_INVOKE(proj_, std::forward<T>(u)));
        }
    };

    template <typename Pred>
    struct invoke_projected<Pred, projection_identity>
    {
        using pred_type = typename std::decay<Pred>::type;

        pred_type pred_;

        template <typename Pred_>
        invoke_projected(Pred_&& pred, projection_identity)
          : pred_(std::forward<Pred_>(pred))
        {
        }

        template <typename T>
        decltype(auto) operator()(T&& t)
        {
            return HPX_INVOKE(pred_, std::forward<T>(t));
        }

        template <typename T>
        bool operator()(T&& t, T&& u)
        {
            return HPX_INVOKE(pred_, std::forward<T>(t), std::forward<T>(u));
        }
    };
}}}    // namespace hpx::parallel::util
