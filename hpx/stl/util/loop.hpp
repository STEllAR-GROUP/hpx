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
    // Helper class to repeatedly call a function starting from a given
    // iterator position.
    template <typename IterCat>
    struct loop
    {
        template <typename Iter, typename F>
        static Iter call(Iter it, Iter end, F && func)
        {
            for (/**/; it != end; ++it)
                func(*it);

            return it;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Helper class to repeatedly call a function a given number of times
    // starting from a given iterator position.
    template <typename IterCat>
    struct loop_n
    {
        template <typename Iter, typename F>
        static Iter call(Iter it, std::size_t count, F && func)
        {
            for (/**/; count != 0; --count, ++it)
                func(*it);

            return it;
        }
    };

    // specialization for random access iterators
    template <>
    struct loop_n<std::random_access_iterator_tag>
    {
        template <typename Iter, typename F>
        static Iter call(Iter it, std::size_t count, F && func)
        {
            for (std::size_t i = 0; i != count; ++i)
                func(it[i]);

            std::advance(it, count);
            return it;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Helper class to repeatedly call a function a given number of times
    // starting from a given iterator position.
    template <typename IterCat>
    struct accumulate_n
    {
        template <typename Iter, typename T, typename Pred>
        static T call(Iter it, std::size_t count, T init, Pred && func)
        {
            for (/**/; count != 0; --count, ++it)
                init = func(init, *it);
            return init;
        }
    };

    // specialization for random access iterators
    template <>
    struct accumulate_n<std::random_access_iterator_tag>
    {
        template <typename Iter, typename T, typename Pred>
        static T call(Iter it, std::size_t count, T init, Pred && func)
        {
            for (std::size_t i = 0; i != count; ++i)
                init = func(init, it[i]);
            return init;
        }
    };
}}}

#endif
