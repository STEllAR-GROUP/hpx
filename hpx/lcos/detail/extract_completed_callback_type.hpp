//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DETAIL_EXTRACT_COMPLETED_CALLBACK_TYPE_HPP
#define HPX_LCOS_LOCAL_DETAIL_EXTRACT_COMPLETED_CALLBACK_TYPE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace hpx { namespace lcos { namespace local { namespace detail
{
    template <
        typename Future
      , typename IsFutureRange = typename traits::is_future_range<Future>::type
    >
    struct extract_completed_callback_type;

    template <typename Future>
    struct extract_completed_callback_type<Future, boost::mpl::true_>
    {
        typedef
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Future
                >::type
            >::type::value_type::future_data_type
            future_data_type;

        typedef
            typename future_data_type::completed_callback_type
            type;
    };

    template <typename Future>
    struct extract_completed_callback_type<Future, boost::mpl::false_>
    {
        typedef
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Future
                >::type
            >::type::future_data_type
            future_data_type;

        typedef
            typename future_data_type::completed_callback_type
            type;
    };

    template <typename F1, typename F2>
    struct compose_cb_impl
    {
        typename util::detail::remove_reference<F1>::type f1_;
        typename util::detail::remove_reference<F2>::type f2_;

        template <typename A1, typename A2>
        compose_cb_impl(BOOST_FWD_REF(A1) f1, BOOST_FWD_REF(A2) f2)
          : f1_(boost::forward<A1>(f1))
          , f2_(boost::forward<A2>(f2))
        {}

        typedef void result_type;

        template <typename Future>
        void operator()(Future & f)
        {
            f1_(f);
            f2_(f);
        }
    };

    template <typename F1, typename F2>
    compose_cb_impl<F1, F2>
    compose_cb(BOOST_FWD_REF(F1) f1, BOOST_FWD_REF(F2) f2)
    {
        return
            boost::move(
                compose_cb_impl<F1, F2>(
                    boost::forward<F1>(f1)
                  , boost::forward<F2>(f2)
                )
            );
    }
}}}}

#endif
