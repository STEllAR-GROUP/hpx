//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_SYNCHRONIZE_JUN_17_1138AM)
#define HPX_STL_DETAIL_SYNCHRONIZE_JUN_17_1138AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/is_future.hpp>

#include <iterator>

namespace hpx { namespace parallel { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    static void synchronize(Iter begin, Iter end, boost::mpl::false_)
    {
    }

    template <typename Iter>
    static void synchronize(Iter begin, Iter end, boost::mpl::true_)
    {
        typedef typename std::iterator_traits<Iter>::value_type type;
        std::for_each(begin, end, [](type& fut) { fut.wait(); });
    }

    template <typename Iter>
    void synchronize(Iter begin, Iter end)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        typedef typename hpx::traits::is_future<
            typename std::iterator_traits<Iter>::value_type
        >::type pred;

        detail::synchronize(begin, end, pred());
    }
}}}

#endif
