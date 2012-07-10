//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/move.hpp>

#include <boost/range.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace hpx {

    template <typename Range, typename F>
    std::vector<
        lcos::future<
            typename boost::result_of<
                F(typename boost::range_value<Range>::type)
            >::type
        >
    >
    for_each(Range const & range, F f)
    {
        typedef typename boost::range_value<Range>::type value_type;
        typedef typename boost::result_of<F(value_type)>::type result_type;
        typedef lcos::future<result_type> future_type;
        typedef std::vector<future_type> futures_type;
        typedef typename boost::range_iterator<Range const>::type iterator_type;

        futures_type futures(boost::size(range));

        if(futures.size() < 2)
        {
            std::size_t i = 0;
            BOOST_FOREACH(value_type const & v, range)
            {
                futures[i] = hpx::async(HPX_STD_BIND(hpx::util::protect(f), v));
            }

            return futures;
        }

        iterator_type begin = boost::begin(range);
        iterator_type mid = begin + boost::size(range)/2;
        iterator_type end = boost::end(range);

        futures_type (*for_each_impl)(boost::iterator_range<iterator_type> const&, F) = for_each;

        lcos::future<futures_type>
            left_future(
                hpx::async(
                    HPX_STD_BIND(
                        for_each_impl
                      , boost::make_iterator_range(begin, mid)
                      , f
                    )
                )
            );

        lcos::future<futures_type>
            right_future(
                hpx::async(
                    HPX_STD_BIND(
                        for_each_impl
                      , boost::make_iterator_range(mid, end)
                      , f
                    )
                )
            );

        hpx::wait_all(left_future, right_future);

        typedef typename boost::range_iterator<futures_type>::type futures_iterator_type;

        futures_iterator_type futures_begin = boost::begin(futures);
        futures_iterator_type futures_mid = futures_begin + boost::size(futures)/2;
        futures_iterator_type futures_end = boost::end(futures);

        boost::copy(left_future.get(), futures_begin);
        boost::copy(right_future.get(), futures_mid);

        return futures;
    }
}
