//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_REVERSE_BASE_JUL_29_2014_0718PM)
#define HPX_PARALLEL_REVERSE_BASE_JUL_29_2014_0718PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>

#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    template <typename ReverseIter>
    typename ReverseIter::iterator_type
    reverse_base(ReverseIter && it)
    {
        return it.base();
    }

    template <typename ReverseIter>
    hpx::future<typename ReverseIter::iterator_type>
    reverse_base(hpx::future<ReverseIter> && it)
    {
        return
            it.then(
                [](hpx::future<ReverseIter> && it)
                {
                    return it.get().base();
                });
    }
}}}}

#endif
