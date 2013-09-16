//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/async.hpp>
#include <hpx/util/move.hpp>

#include <boost/range.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace hpx {

    namespace detail
    {
        template <typename Output>
        struct for_each_copy_future
        {
            typedef void result_type;

            Output & out_;

            for_each_copy_future(Output & out)
                : out_(out)
            {}

            template <typename Future>
            result_type operator()(Future const & f)
            {
                boost::copy(f.get(), out_);
            }
        };
    }

    template <typename Range, typename F>
    inline std::vector<
        lcos::future<
#if defined(HPX_HAVE_CXX11_DECLTYPE)
            decltype(boost::declval<F>()(boost::declval<typename boost::range_value<Range>::type>()))
#else
            typename boost::result_of<
                F(typename boost::range_value<Range>::type)
            >::type
#endif
        >
    >
    for_each(Range const & range, F f)
    {
        typedef typename boost::range_value<Range>::type value_type;
#if defined(HPX_HAVE_CXX11_DECLTYPE)
        typedef decltype(boost::declval<F>()(boost::declval<value_type>())) result_type;
#else
        typedef typename boost::result_of<F(value_type)>::type result_type;
#endif
        typedef lcos::future<result_type> future_type;
        typedef std::vector<future_type> futures_type;
        typedef typename boost::range_iterator<Range const>::type iterator_type;

        futures_type futures(boost::size(range));

        std::size_t granularity = hpx::get_os_thread_count() == 1 ? 2 : hpx::get_os_thread_count();

        if(futures.size() < granularity)
        {
            std::size_t i = 0;
            BOOST_FOREACH(value_type const & v, range)
            {
                futures[i] = hpx::async(HPX_STD_BIND(hpx::util::protect(f), v));
                ++i;
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

        typedef typename futures_type::iterator futures_iterator_type;

        futures_iterator_type futures_begin = futures.begin();
        futures_iterator_type futures_mid = futures.begin() + futures.size()/2;
        
        std::vector<hpx::lcos::future<void> > v;
        v.reserve(2);
        v.push_back(left_future.then(detail::for_each_copy_future<futures_iterator_type>(futures_begin)));
        v.push_back(right_future.then(detail::for_each_copy_future<futures_iterator_type>(futures_mid)));
        hpx::lcos::wait(v);

        return futures;
    }
}
