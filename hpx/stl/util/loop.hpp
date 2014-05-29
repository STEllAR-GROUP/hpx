//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_UTIL_LOOP_MAY_27_2014_1040PM)
#define HPX_STL_UTIL_LOOP_MAY_27_2014_1040PM

#include <hpx/hpx_fwd.hpp>

#include <iterator>
#include <algorithm>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // Helper class to repeatedly call a function starting from a given
    // iterator position.
    template <typename Iter,
        typename IterCat = typename std::iterator_traits<Iter>::iterator_category>
    struct loop
    {
        template <typename Diff, typename F>
        static Iter call(Iter it, Diff count, F && func)
        {
            for (/**/; count != 0; --count, ++it)
                func(*it);

            return it;
        }
    };

    // specialization for random access iterators
    template <typename Iter>
    struct loop<Iter, std::random_access_iterator_tag>
    {
        template <typename Diff, typename F>
        static Iter call(Iter it, Diff count, F && func)
        {
            for (Diff i = 0; i != count; ++i)
                func(it[i]);

            std::advance(it, count);
            return it;
        }
    };
}}}

#endif
