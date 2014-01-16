//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/move.hpp>

#include <boost/range.hpp>

namespace hpx {

    namespace detail
    {
        template <typename Output>
        struct for_each_move_future
        {
            typedef void result_type;

            Output & out_;

            for_each_move_future(Output & out)
              : out_(out)
            {}

            template <typename R>
            result_type operator()(lcos::unique_future<R> f)
            {
                R futures = f.get();

                std::move(futures.begin(), futures.end(), out_);
            }
        };
    }

    template <typename Range, typename F>
    inline std::vector<
        lcos::unique_future<
            typename util::result_of<
                F(typename boost::range_value<Range>::type)
            >::type
        >
    >
    for_each(Range const & range, F f)
    {
        typedef typename boost::range_value<Range>::type value_type;
        typedef typename util::result_of<F(value_type)>::type result_type;
        typedef lcos::unique_future<result_type> future_type;
        typedef std::vector<future_type> futures_type;
        typedef typename boost::range_iterator<Range const>::type iterator_type;

        futures_type futures;
        futures.reserve(boost::size(range));

        std::size_t granularity = hpx::get_os_thread_count() == 1 ? 2 : hpx::get_os_thread_count();

        if(futures.size() < granularity)
        {
            BOOST_FOREACH(value_type const & v, range)
            {
                futures.push_back(hpx::async(HPX_STD_PROTECT(f), v));
            }

            return futures;
        }

        iterator_type begin = boost::begin(range);
        iterator_type mid = begin + boost::size(range)/2;
        iterator_type end = boost::end(range);

        futures_type (*for_each_impl)(boost::iterator_range<iterator_type> const&, F) = for_each;

        lcos::unique_future<futures_type> left_future =
            hpx::async(
                  for_each_impl
                , boost::make_iterator_range(begin, mid)
                , HPX_STD_PROTECT(f)
            );

        lcos::unique_future<futures_type> right_future =
            hpx::async(
                  for_each_impl
                , boost::make_iterator_range(mid, end)
                , HPX_STD_PROTECT(f)
            );

        typedef typename futures_type::iterator futures_iterator_type;
        typedef detail::for_each_move_future<futures_iterator_type> move_futures_type;

        futures_iterator_type futures_begin = futures.begin();
        futures_iterator_type futures_mid = futures.begin() + futures.size()/2;

        hpx::wait_all(
            left_future.then(move_futures_type(futures_begin))
          , right_future.then(move_futures_type(futures_mid))
        );

        return futures;
    }
}
