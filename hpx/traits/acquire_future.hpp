//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACQUIRE_FUTURE_DEC_23_2014_0911AM)
#define HPX_TRAITS_ACQUIRE_FUTURE_DEC_23_2014_0911AM

#include <hpx/config.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/range/functions.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <vector>
#include <iterator>

namespace hpx { namespace traits
{
    template <typename T, typename Enable = void>
    struct acquire_future_impl;

    template <typename T, typename Enable>
    struct acquire_future
      : acquire_future_impl<typename util::decay<T>::type>
    {};

    struct acquire_future_disp
    {
        template <typename T>
        BOOST_FORCEINLINE typename acquire_future<T>::type
        operator()(T && t) const
        {
            return acquire_future<T>()(std::forward<T>(t));
        }
    };

    template <typename T>
    struct acquire_future_impl<T,
        typename boost::disable_if_c<is_future_or_future_range<T>::value>::type>
    {
        typedef T type;

        template <typename T_>
        BOOST_FORCEINLINE
        T operator()(T_ && value) const
        {
            return std::forward<T_>(value);
        }
    };

    template <typename R>
    struct acquire_future_impl<hpx::lcos::future<R> >
    {
        typedef hpx::lcos::future<R> type;

        BOOST_FORCEINLINE hpx::lcos::future<R>
        operator()(hpx::lcos::future<R>& future) const
        {
            return std::move(future);
        }

        BOOST_FORCEINLINE hpx::lcos::future<R>
        operator()(hpx::lcos::future<R>&& future) const
        {
            return std::move(future);
        }
    };

    template <typename R>
    struct acquire_future_impl<hpx::lcos::shared_future<R> >
    {
        typedef hpx::lcos::shared_future<R> type;

        BOOST_FORCEINLINE hpx::lcos::shared_future<R>
        operator()(hpx::lcos::shared_future<R> future) const
        {
            return future;
        }
    };

    namespace detail
    {
        // Reserve sufficient space in the given vector if the underlying
        // iterator type of the given range allow calculating the size on O(1).
        template <typename Future, typename Range>
        BOOST_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>&, Range const&,
            boost::mpl::false_)
        {
        }

        template <typename Future, typename Range>
        BOOST_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>& v, Range const& r,
            boost::mpl::true_)
        {
            v.reserve(boost::size(r));
        }

        template <typename Range1, typename Range2>
        BOOST_FORCEINLINE
        void reserve_if_random_access(Range1&, Range2 const&)
        {
            // do nothing if it's not a vector
        }

        template <typename Future, typename Range>
        BOOST_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>& v, Range const& r)
        {
            typedef typename std::iterator_traits<
                    typename Range::iterator
                >::iterator_category iterator_category;

            typedef typename boost::is_same<
                    iterator_category, std::random_access_iterator_tag
                >::type is_random_access;

            reserve_if_random_access(v, r, is_random_access());
        }
    }

    template <typename Range>
    struct acquire_future_impl<Range,
        typename boost::enable_if_c<traits::is_future_range<Range>::value>::type>
    {
        typedef typename traits::future_range_traits<Range>::future_type
            future_type;

        typedef Range type;

        BOOST_FORCEINLINE Range
        operator()(Range&& futures) const
        {
            return std::move(futures);
        }

        template <typename Range_>
        BOOST_FORCEINLINE Range
        operator()(Range_&& futures) const
        {
            Range values;
            detail::reserve_if_random_access(values, futures);

            std::transform(boost::begin(futures), boost::end(futures),
                std::back_inserter(values), acquire_future_disp());

            return values;
        }
    };
}}

#endif
