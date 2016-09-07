//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_PARTITIONER_ITERATION)
#define HPX_PARALLEL_UTIL_DETAIL_PARTITIONER_ITERATION

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        // Hand-crafted function object allowing to replace a more complex
        // bind(functional::invoke_fused(), f1, _1)
        template <typename Result, typename F>
        struct partitioner_iteration
        {
            typename hpx::util::decay<F>::type f_;

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            Result operator()(T && t)
            {
                return hpx::util::invoke_fused(f_, std::forward<T>(t));
            }
        };
    }
}}}

#endif
