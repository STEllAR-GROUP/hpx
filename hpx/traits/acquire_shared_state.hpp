//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACQUIRE_SHARED_STATE_JUN_24_2015_0923AM)
#define HPX_TRAITS_ACQUIRE_SHARED_STATE_JUN_24_2015_0923AM

#include <hpx/config.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/decay.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/range/functions.hpp>

#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct acquire_shared_state_impl;
    }

    template <typename T, typename Enable = void>
    struct acquire_shared_state
      : detail::acquire_shared_state_impl<typename util::decay<T>::type>
    {};

    struct acquire_shared_state_disp
    {
        template <typename T>
        HPX_FORCEINLINE typename acquire_shared_state<T>::type
        operator()(T && t) const
        {
            return acquire_shared_state<T>()(std::forward<T>(t));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        HPX_FORCEINLINE typename acquire_shared_state<T>::type
        get_shared_state(T && t)
        {
            return acquire_shared_state<T>()(std::forward<T>(t));
        }

        template <typename R>
        HPX_FORCEINLINE
        boost::intrusive_ptr<lcos::detail::future_data<R> > const&
        get_shared_state(
            boost::intrusive_ptr<lcos::detail::future_data<R> > const& t)
        {
            return t;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        struct wait_get_shared_state
        {
            HPX_FORCEINLINE
            typename traits::detail::shared_state_ptr_for<Future>::type const&
            operator()(Future const& f) const
            {
                return traits::detail::get_shared_state(f);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        struct acquire_shared_state_impl<T,
            typename std::enable_if<!is_future_or_future_range<T>::value>::type>
        {
            typedef T type;

            template <typename T_>
            HPX_FORCEINLINE
            T operator()(T_ && value) const
            {
                return value;
            }
        };

        template <typename T>
        struct acquire_shared_state_impl<T,
            typename std::enable_if<is_future<T>::value>::type>
        {
            typedef typename traits::detail::shared_state_ptr<
                typename traits::future_traits<T>::type
            >::type const& type;

            HPX_FORCEINLINE type
            operator()(T const& f) const
            {
                return traits::future_access<T>::get_shared_state(f);
            }
        };

        template <typename Range>
        struct acquire_shared_state_impl<Range,
            typename std::enable_if<traits::is_future_range<Range>::value>::type>
        {
            typedef typename traits::future_range_traits<Range>::future_type
                future_type;

            typedef typename traits::detail::shared_state_ptr_for<future_type>::type
                shared_state_ptr;
            typedef std::vector<shared_state_ptr> type;

            template <typename Range_>
            HPX_FORCEINLINE type
            operator()(Range_&& futures) const
            {
                std::vector<shared_state_ptr> values;
                detail::reserve_if_random_access(values, futures);

                std::transform(boost::begin(futures), boost::end(futures),
                    std::back_inserter(values), acquire_shared_state_disp());

                return values;
            }
        };
    }
}}

#endif
