//  Copyright (c) 2016-2017 Hartmut Kaiser
//  Copyright (c) 2018 Christopher Ogle
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
    template <typename Compare, typename ... Proj>
    struct compare_projected;

    template <typename Compare, typename Proj>
    struct compare_projected<Compare, Proj>
    {
        template <typename Compare_, typename Proj_>
        compare_projected(Compare_ && comp, Proj_ && proj)
            : comp_(std::forward<Compare_>(comp)),
            proj_(std::forward<Proj_>(proj))
        {}

        template <typename T1, typename T2>
        inline bool operator()(T1 && t1, T2 && t2) const
        {
            return hpx::util::invoke(comp_,
                hpx::util::invoke(proj_, std::forward<T1>(t1)),
                hpx::util::invoke(proj_, std::forward<T2>(t2)));
        }

        Compare comp_;
        Proj proj_;
    };

    template <typename Compare, typename Proj1, typename Proj2>
    struct compare_projected<Compare, Proj1, Proj2>
    {
        template <typename Compare_, typename Proj1_, typename Proj2_>
        compare_projected(Compare_ && comp, Proj1_ && proj1, Proj2_ && proj2)
            : comp_(std::forward<Compare_>(comp)),
            proj1_(std::forward<Proj1_>(proj1)),
            proj2_(std::forward<Proj2_>(proj2))
        {}

        template <typename T1, typename T2>
        inline bool operator()(T1 && t1, T2 && t2) const
        {
            return hpx::util::invoke(comp_,
                hpx::util::invoke(proj1_, std::forward<T1>(t1)),
                hpx::util::invoke(proj1_, std::forward<T2>(t2)));
        }

        Compare comp_;
        Proj1 proj1_;
        Proj2 proj2_;
    };
}}}

#endif
