//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_SYNCHRONIZE_JUN_17_1138AM)
#define HPX_PARALLEL_DETAIL_SYNCHRONIZE_JUN_17_1138AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/is_future.hpp>

#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
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
        typedef typename hpx::traits::is_future<
            typename std::iterator_traits<Iter>::value_type
        >::type pred;

        detail::synchronize(begin, end, pred());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    static void synchronize_binary(Iter1 begin1, Iter1 end1, Iter2 begin2,
        boost::mpl::false_)
    {
    }

    template <typename Iter1, typename Iter2, typename F>
    void for_each2(Iter1 begin1, Iter1 end1, Iter2 begin2, F && f)
    {
        for (/**/; begin1 != end1; ++begin1, ++begin2)
            f(*begin1, *begin2);
    }

    template <typename Future>
    typename boost::disable_if<hpx::traits::is_future<Future> >::type
    future_wait(Future& f)
    {
    }

    template <typename Future>
    typename boost::enable_if<hpx::traits::is_future<Future> >::type
    future_wait(Future& f)
    {
        f.wait();
    }

    template <typename Iter1, typename Iter2>
    static void synchronize_binary(Iter1 begin1, Iter1 end1, Iter2 begin2,
        boost::mpl::true_)
    {
        typedef typename std::iterator_traits<Iter1>::value_type type1;
        typedef typename std::iterator_traits<Iter2>::value_type type2;

        for_each2(begin1, end1, begin2,
            [](type1& v1, type2& v2)
            {
                future_wait(v1);
                future_wait(v2);
            });
    }

    template <typename Iter1, typename Iter2>
    void synchronize_binary(Iter1 begin1, Iter1 end1, Iter2 begin2)
    {
        typedef typename boost::mpl::or_<
            hpx::traits::is_future<
                typename std::iterator_traits<Iter1>::value_type>,
            hpx::traits::is_future<
                typename std::iterator_traits<Iter2>::value_type>
        >::type pred;

        detail::synchronize_binary(begin1, end1, begin2, pred());
    }
}}}}

#endif
