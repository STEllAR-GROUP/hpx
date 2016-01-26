//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_COMPARE_PROJECTED_JAN_25_2016_1144AM)
#define HPX_PARALLEL_UTIL_COMPARE_PROJECTED_JAN_25_2016_1144AM

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Compare, typename Proj>
    struct compare_projected
    {
        template <typename Compare_, typename Proj_>
        compare_projected(Compare_ && comp, Proj_ && proj)
            : comp_(std::forward<Compare_>(comp)),
            proj_(std::forward<Proj_>(proj))
        {}

        template <typename T1, typename T2>
        inline bool operator()(T1 && t1, T2 && t2)
        {
            return hpx::util::invoke(comp_,
                hpx::util::invoke(proj_, t1),
                hpx::util::invoke(proj_, t2));
        }

        Compare comp_;
        Proj proj_;
    };
}}}

#endif
