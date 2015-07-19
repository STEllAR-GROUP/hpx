//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_PREDICATES_JUL_13_2014_0824PM)
#define HPX_PARALLEL_DETAIL_PREDICATES_JUL_13_2014_0824PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ForwardIt>
    ForwardIt next(ForwardIt it,
        typename std::iterator_traits<ForwardIt>::difference_type n)
    {
        std::advance(it, n);
        return it;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct equal_to
    {
        template <typename T1, typename T2>
        bool operator()(T1 const& t1, T2 const& t2) const
        {
            return t1 == t2;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct less
    {
        template <typename T1, typename T2>
        bool operator()(T1 const& t1, T2 const& t2) const
        {
            return t1 < t2;
        }
    };
}}}}

#endif
