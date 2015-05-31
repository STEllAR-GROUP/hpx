//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INVOKE_FUSED_HPP
#define HPX_UTIL_INVOKE_FUSED_HPP

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/type_traits/add_const.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Indices, typename F>
        struct invoke_fused_result_of_impl;

        template <std::size_t ...Is, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            detail::pack_c<std::size_t, Is...>, F(Tuple)
        > : result_of<
                F(typename detail::qualify_as<typename util::tuple_element<
                        Is, typename util::decay<Tuple>::type
                    >::type, Tuple>::type...)
            >
        {};
    }

    template <typename F>
    struct invoke_fused_result_of;

    template <typename F, typename Tuple>
    struct invoke_fused_result_of<F(Tuple)>
      : detail::invoke_fused_result_of_impl<
            typename detail::make_index_pack<
                util::tuple_size<typename util::decay<Tuple>::type>::value
            >::type
          , F(Tuple)
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R, std::size_t ...Is, typename F, typename Tuple>
        BOOST_FORCEINLINE
        R
        invoke_fused_impl(detail::pack_c<std::size_t, Is...>,
            F && f, Tuple const& args)
        {
            return invoke_r<R>(
                std::forward<F>(f), util::get<Is>(args)...);
        }

        template <typename R, std::size_t ...Is, typename F, typename Tuple>
        BOOST_FORCEINLINE
        R
        invoke_fused_impl(detail::pack_c<std::size_t, Is...>, //-V659
            F && f, Tuple&& args)
        {
            return invoke_r<R>(
                std::forward<F>(f), util::get<Is>(std::move(args))...);
        }
    }

    template <typename R, typename F, typename ...Ts>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(F && f, util::tuple<Ts...> const& args)
    {
        return detail::invoke_fused_impl<R>(
            typename detail::make_index_pack<sizeof...(Ts)>::type(),
            std::forward<F>(f), args);
    }

    template <typename R, typename F, typename ...Ts>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(F && f, util::tuple<Ts...>&& args) //-V659
    {
        return detail::invoke_fused_impl<R>(
            typename detail::make_index_pack<sizeof...(Ts)>::type(),
            std::forward<F>(f), std::move(args));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    BOOST_FORCEINLINE
    typename result_of<F(typename boost::add_const<Ts>::type...)>::type
    invoke_fused(F && f, util::tuple<Ts...> const& args)
    {
        typedef typename result_of<
            F(typename boost::add_const<Ts>::type...)
        >::type result_type;

        return detail::invoke_fused_impl<result_type>(
            typename detail::make_index_pack<sizeof...(Ts)>::type(),
            std::forward<F>(f), args);
    }

    template <typename F, typename ...Ts>
    BOOST_FORCEINLINE
    typename result_of<F(Ts...)>::type
    invoke_fused(F && f, util::tuple<Ts...>&& args) //-V659
    {
        typedef typename result_of<F(Ts...)>::type result_type;

        return detail::invoke_fused_impl<result_type>(
            typename detail::make_index_pack<sizeof...(Ts)>::type(),
            std::forward<F>(f), std::move(args));
    }

    namespace functional
    {
        struct invoke_fused
        {
            template <typename F, typename Tuple>
            typename invoke_fused_result_of<F(Tuple)>::type
            operator()(F && f, Tuple && args)
            {
                return util::invoke_fused(
                    std::forward<F>(f),
                    std::forward<Tuple>(args));
            }
        };

        template <typename R>
        struct invoke_fused_r
        {
            template <typename F, typename Tuple>
            R operator()(F && f, Tuple && args)
            {
                return util::invoke_fused_r<R>(
                    std::forward<F>(f),
                    std::forward<Tuple>(args));
            }
        };
    }
}}

#endif
