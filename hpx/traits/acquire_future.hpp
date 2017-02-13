//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACQUIRE_FUTURE_DEC_23_2014_0911AM)
#define HPX_TRAITS_ACQUIRE_FUTURE_DEC_23_2014_0911AM

#include <hpx/config.hpp>
#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/decay.hpp>

#include <boost/range/functions.hpp>
#include <boost/range/iterator_range.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct acquire_future_impl;
    }

    template <typename T, typename Enable = void>
    struct acquire_future
      : detail::acquire_future_impl<typename util::decay<T>::type>
    {};

    struct acquire_future_disp
    {
        template <typename T>
        HPX_FORCEINLINE typename acquire_future<T>::type
        operator()(T && t) const
        {
            return acquire_future<T>()(std::forward<T>(t));
        }
    };

    namespace detail
    {
        template <typename T>
        struct acquire_future_impl<T,
            typename std::enable_if<
                !is_future_or_future_range<T>::value
            >::type>
        {
            typedef T type;

            template <typename T_>
            HPX_FORCEINLINE
            T operator()(T_ && value) const
            {
                return std::forward<T_>(value);
            }
        };

        template <typename R>
        struct acquire_future_impl<hpx::lcos::future<R> >
        {
            typedef hpx::lcos::future<R> type;

            HPX_FORCEINLINE hpx::lcos::future<R>
            operator()(hpx::lcos::future<R>& future) const
            {
                return std::move(future);
            }

            HPX_FORCEINLINE hpx::lcos::future<R>
            operator()(hpx::lcos::future<R>&& future) const
            {
                return std::move(future);
            }
        };

        template <typename R>
        struct acquire_future_impl<hpx::lcos::shared_future<R> >
        {
            typedef hpx::lcos::shared_future<R> type;

            HPX_FORCEINLINE hpx::lcos::shared_future<R>
            operator()(hpx::lcos::shared_future<R> future) const
            {
                return future;
            }
        };

        // Reserve sufficient space in the given vector if the underlying
        // iterator type of the given range allow calculating the size on O(1).
        template <typename Future, typename Range>
        HPX_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>&, Range const&,
            std::false_type)
        {
        }

        template <typename Future, typename Range>
        HPX_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>& v, Range const& r,
            std::true_type)
        {
            v.reserve(boost::size(r));
        }

        template <typename Range1, typename Range2>
        HPX_FORCEINLINE
        void reserve_if_random_access(Range1&, Range2 const&)
        {
            // do nothing if it's not a vector
        }

        template <typename Future, typename Range>
        HPX_FORCEINLINE
        void reserve_if_random_access(std::vector<Future>& v, Range const& r)
        {
            typedef typename std::iterator_traits<
                    typename Range::iterator
                >::iterator_category iterator_category;

            typedef std::is_same<
                    iterator_category, std::random_access_iterator_tag
                > is_random_access;

            reserve_if_random_access(v, r, is_random_access());
        }

        template <typename Container>
        HPX_FORCEINLINE
        void reserve_if_vector(Container&, std::size_t)
        {
        }

        template <typename Future>
        HPX_FORCEINLINE
        void reserve_if_vector(std::vector<Future>& v, std::size_t n)
        {
            v.reserve(n);
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(push_back);

        ///////////////////////////////////////////////////////////////////////
        template <typename Range>
        struct acquire_future_impl<Range,
            typename std::enable_if<
                traits::is_future_range<Range>::value
            >::type>
        {
            typedef typename traits::future_range_traits<Range>::future_type
                future_type;

            typedef Range type;

            HPX_FORCEINLINE Range
            operator()(Range&& futures) const
            {
                return std::move(futures);
            }

            template <typename Range_>
            typename std::enable_if<
                has_push_back<typename std::decay<Range_>::type>::value
            >::type
            transform_future_disp(Range_ && futures, Range& values) const
            {
                detail::reserve_if_random_access(values, futures);
                std::transform(boost::begin(futures), boost::end(futures),
                    std::back_inserter(values), acquire_future_disp());
            }

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
            template <typename Range_>
            typename std::enable_if<
                !has_push_back<typename std::decay<Range_>::type>::value
            >::type
            transform_future_disp(Range_ && futures, Range& values) const
            {
                detail::reserve_if_random_access(values, futures);
                std::transform(boost::begin(futures), boost::end(futures),
                    boost::begin(values), acquire_future_disp());
            }
#endif

            template <typename Range_>
            HPX_FORCEINLINE Range
            operator()(Range_ && futures) const
            {
                Range values;
                transform_future_disp(std::forward<Range_>(futures), values);
                return values;
            }
        };
    }
}}

#endif
