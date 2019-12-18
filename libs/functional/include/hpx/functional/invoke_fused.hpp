//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:util::result_of

#ifndef HPX_UTIL_INVOKE_FUSED_HPP
#define HPX_UTIL_INVOKE_FUSED_HPP

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/result_of.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tuple>
        struct fused_index_pack
          : make_index_pack<
                util::tuple_size<typename std::decay<Tuple>::type>::value>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Tuple, typename Is>
        struct fused_result_of_impl;

        template <typename F, typename Tuple, std::size_t... Is>
        struct fused_result_of_impl<F, Tuple&, index_pack<Is...>>
          : util::result_of<F(
                typename util::tuple_element<Is, Tuple>::type&...)>
        {
        };

        template <typename F, typename Tuple, std::size_t... Is>
        struct fused_result_of_impl<F, Tuple&&, index_pack<Is...>>
          : util::result_of<F(
                typename util::tuple_element<Is, Tuple>::type&&...)>
        {
        };

        template <typename T>
        struct fused_result_of;

        template <typename F, typename Tuple>
        struct fused_result_of<F(Tuple)>
          : fused_result_of_impl<F, Tuple&&,
                typename fused_index_pack<Tuple>::type>
        {
        };

        template <typename F, typename Tuple>
        struct invoke_fused_result
          : fused_result_of_impl<F, Tuple&&,
                typename fused_index_pack<Tuple>::type>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t... Is, typename F, typename Tuple>
        constexpr HPX_HOST_DEVICE typename invoke_fused_result<F, Tuple>::type
        invoke_fused_impl(index_pack<Is...>, F&& f, Tuple&& t)
        {
            return HPX_INVOKE(
                std::forward<F>(f), util::get<Is>(std::forward<Tuple>(t))...);
        }
    }    // namespace detail

    /// Invokes the given callable object f with the content of
    /// the sequenced type t (tuples, pairs)
    ///
    /// \param f Must be a callable object. If f is a member function pointer,
    ///          the first argument in the sequenced type will be treated as
    ///          the callee (this object).
    ///
    /// \param t A type which is content accessible through a call
    ///          to hpx#util#get.
    ///
    /// \returns The result of the callable object when it's called with
    ///          the content of the given sequenced type.
    ///
    /// \throws std::exception like objects thrown by call to object f
    ///         with the arguments contained in the sequenceable type t.
    ///
    /// \note This function is similar to `std::apply` (C++17)
    template <typename F, typename Tuple>
    constexpr HPX_HOST_DEVICE
        typename detail::invoke_fused_result<F, Tuple>::type
        invoke_fused(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return detail::invoke_fused_impl(
            index_pack{}, std::forward<F>(f), std::forward<Tuple>(t));
    }

    /// \copydoc invoke_fused
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given sequenced type.
    template <typename R, typename F, typename Tuple>
    constexpr HPX_HOST_DEVICE R invoke_fused_r(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return util::void_guard<R>(),
               detail::invoke_fused_impl(
                   index_pack{}, std::forward<F>(f), std::forward<Tuple>(t));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace functional {
        struct invoke_fused
        {
            template <typename F, typename Tuple>
            constexpr HPX_HOST_DEVICE
                typename util::detail::invoke_fused_result<F, Tuple>::type
                operator()(F&& f, Tuple&& t) const
            {
                using index_pack =
                    typename util::detail::fused_index_pack<Tuple>::type;
                return util::detail::invoke_fused_impl(
                    index_pack{}, std::forward<F>(f), std::forward<Tuple>(t));
            }
        };

        template <typename R>
        struct invoke_fused_r
        {
            template <typename F, typename Tuple>
            constexpr HPX_HOST_DEVICE R operator()(F&& f, Tuple&& t) const
            {
                using index_pack =
                    typename util::detail::fused_index_pack<Tuple>::type;
                return util::void_guard<R>(),
                       util::detail::invoke_fused_impl(index_pack{},
                           std::forward<F>(f), std::forward<Tuple>(t));
            }
        };
    }    // namespace functional
    /// \endcond
}}    // namespace hpx::util

#endif
