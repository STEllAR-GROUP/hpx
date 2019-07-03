//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_INVOKE_PROJECTED_SEP_27_2016_0912AM)
#define HPX_PARALLEL_UTIL_INVOKE_PROJECTED_SEP_27_2016_0912AM

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Pred, typename Proj>
    struct invoke_projected
    {
        typedef typename hpx::util::decay<Pred>::type pred_type;
        typedef typename hpx::util::decay<Proj>::type proj_type;

        template <typename Pred_, typename Proj_>
        invoke_projected(Pred_ && pred, Proj_ && proj)
          : pred_(std::forward<Pred_>(pred)),
            proj_(std::forward<Proj_>(proj))
        {}

        template <typename T>
        auto operator()(T && t)
        ->  decltype(
                hpx::util::invoke(std::declval<pred_type>(),
                    hpx::util::invoke(std::declval<proj_type>(),
                        std::forward<T>(t)))
            )
        {
            return hpx::util::invoke(pred_,
                hpx::util::invoke(proj_, std::forward<T>(t)));
        }

        pred_type pred_;
        proj_type proj_;
    };
}}}

#endif
