//  Copyright (c) 2016-2020 Hartmut Kaiser
//  Copyright (c) 2018 Christopher Ogle
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>

#include <hpx/parallel/util/projection_identity.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Compare, typename... Proj>
    struct compare_projected;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Compare, typename Proj>
    struct compare_projected<Compare, Proj>
    {
        template <typename Compare_, typename Proj_>
        constexpr compare_projected(Compare_&& comp, Proj_&& proj)
          : comp_(HPX_FORWARD(Compare_, comp))
          , proj_(HPX_FORWARD(Proj_, proj))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_INVOKE(proj_, HPX_FORWARD(T1, t1)),
                HPX_INVOKE(proj_, HPX_FORWARD(T2, t2)));
        }

        Compare comp_;
        Proj proj_;
    };

    template <typename Compare>
    struct compare_projected<Compare, util::projection_identity>
    {
        template <typename Compare_>
        constexpr compare_projected(Compare_&& comp, util::projection_identity)
          : comp_(HPX_FORWARD(Compare_, comp))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_FORWARD(T1, t1), HPX_FORWARD(T2, t2));
        }

        Compare comp_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Compare, typename Proj1, typename Proj2>
    struct compare_projected<Compare, Proj1, Proj2>
    {
        template <typename Compare_, typename Proj1_, typename Proj2_>
        constexpr compare_projected(
            Compare_&& comp, Proj1_&& proj1, Proj2_&& proj2)
          : comp_(HPX_FORWARD(Compare_, comp))
          , proj1_(HPX_FORWARD(Proj1_, proj1))
          , proj2_(HPX_FORWARD(Proj2_, proj2))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_INVOKE(proj1_, HPX_FORWARD(T1, t1)),
                HPX_INVOKE(proj2_, HPX_FORWARD(T2, t2)));
        }

        Compare comp_;
        Proj1 proj1_;
        Proj2 proj2_;
    };

    template <typename Compare, typename Proj2>
    struct compare_projected<Compare, util::projection_identity, Proj2>
    {
        template <typename Compare_, typename Proj2_>
        constexpr compare_projected(
            Compare_&& comp, util::projection_identity, Proj2_&& proj2)
          : comp_(HPX_FORWARD(Compare_, comp))
          , proj2_(HPX_FORWARD(Proj2_, proj2))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_FORWARD(T1, t1),
                HPX_INVOKE(proj2_, HPX_FORWARD(T2, t2)));
        }

        Compare comp_;
        Proj2 proj2_;
    };

    template <typename Compare, typename Proj1>
    struct compare_projected<Compare, Proj1, util::projection_identity>
    {
        template <typename Compare_, typename Proj1_>
        constexpr compare_projected(
            Compare_&& comp, Proj1_&& proj1, util::projection_identity)
          : comp_(HPX_FORWARD(Compare_, comp))
          , proj1_(HPX_FORWARD(Proj1_, proj1))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_INVOKE(proj1_, HPX_FORWARD(T1, t1)),
                HPX_FORWARD(T2, t2));
        }

        Compare comp_;
        Proj1 proj1_;
    };

    template <typename Compare>
    struct compare_projected<Compare, util::projection_identity,
        util::projection_identity>
    {
        template <typename Compare_>
        constexpr compare_projected(Compare_&& comp, util::projection_identity,
            util::projection_identity)
          : comp_(HPX_FORWARD(Compare_, comp))
        {
        }

        template <typename T1, typename T2>
        inline constexpr bool operator()(T1&& t1, T2&& t2) const
        {
            return HPX_INVOKE(comp_, HPX_FORWARD(T1, t1), HPX_FORWARD(T2, t2));
        }

        Compare comp_;
    };
}}}    // namespace hpx::parallel::util
