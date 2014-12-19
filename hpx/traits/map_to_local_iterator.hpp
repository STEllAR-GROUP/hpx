//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MAP_TO_LOCAL_ITERATOR_NOV_30_2014_1131AM)
#define HPX_MAP_TO_LOCAL_ITERATOR_NOV_30_2014_1131AM

#include <hpx/traits.hpp>

namespace hpx { namespace traits
{
    // Some 'remote' iterators need to be mapped before being applied to the
    // local algorithms.
    template <typename T, typename Enable>
    struct map_to_local_iterator
    {
        static T const& call(T const& t) { return t; }
    };
}}

#endif

