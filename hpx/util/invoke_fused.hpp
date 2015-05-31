//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INVOKE_FUSED_HPP
#define HPX_UTIL_INVOKE_FUSED_HPP

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Tuple>
        struct fused_index_pack
          : make_index_pack<
                util::tuple_size<typename std::decay<Tuple>::type>::value
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Tuple, typename Is>
        struct fused_result_of_impl;

        template <typename F, typename Tuple, std::size_t ...Is>
        struct fused_result_of_impl<F, Tuple&, pack_c<std::size_t, Is...> >
          : util::result_of<
                F(typename util::tuple_element<Is, Tuple>::type&...)
            >
        {};

        template <typename F, typename Tuple, std::size_t ...Is>
        struct fused_result_of_impl<F, Tuple&&, pack_c<std::size_t, Is...> >
          : util::result_of<
                F(typename util::tuple_element<Is, Tuple>::type&&...)
            >
        {};

        template <typename T>
        struct fused_result_of;

        template <typename F, typename Tuple>
        struct fused_result_of<F(Tuple)>
          : fused_result_of_impl<F, Tuple&&, typename fused_index_pack<Tuple>::type>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Tuple, std::size_t ...Is>
        inline typename fused_result_of<F&&(Tuple&&)>::type
        invoke_fused_impl(F&&f, Tuple&& t, pack_c<std::size_t, Is...>)
        {
            using util::get;
            return util::invoke(std::forward<F>(f),
                get<Is>(std::forward<Tuple>(t))...);
        }
    }

    template <typename F, typename Tuple>
    inline typename detail::fused_result_of<F&&(Tuple&&)>::type
    invoke_fused(F&& f, Tuple&& t)
    {
        return detail::invoke_fused_impl(
            std::forward<F>(f), std::forward<Tuple>(t),
            typename detail::fused_index_pack<Tuple>::type());
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R>
        struct invoke_fused_guard
        {
            template <typename F, typename Tuple>
            inline R operator()(F&& f, Tuple&& t)
            {
                return detail::invoke_fused_impl(
                    std::forward<F>(f), std::forward<Tuple>(t),
                    typename detail::fused_index_pack<Tuple>::type());
            }
        };

        template <>
        struct invoke_fused_guard<void>
        {
            template <typename F, typename Tuple>
            inline void operator()(F&& f, Tuple&& t)
            {
                detail::invoke_fused_impl(
                    std::forward<F>(f), std::forward<Tuple>(t),
                    typename detail::fused_index_pack<Tuple>::type());
            }
        };
    }

    template <typename R, typename F, typename Tuple>
    inline R invoke_fused_r(F&& f, Tuple&& t)
    {
        return detail::invoke_fused_guard<R>()(
            std::forward<F>(f), std::forward<Tuple>(t));
    }

    namespace functional
    {
        struct invoke_fused
        {
            template <typename F, typename Tuple>
            typename util::detail::fused_result_of<F&&(Tuple&&)>::type
            operator()(F&& f, Tuple&& args)
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
            R operator()(F&& f, Tuple&& args)
            {
                return util::invoke_fused_r<R>(
                    std::forward<F>(f),
                    std::forward<Tuple>(args));
            }
        };
    }
}}

#endif
