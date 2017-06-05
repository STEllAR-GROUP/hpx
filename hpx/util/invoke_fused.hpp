//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c) 2016 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INVOKE_FUSED_HPP
#define HPX_UTIL_INVOKE_FUSED_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/void_guard.hpp>

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
        template <typename R, typename F, typename Tuple, std::size_t ...Is>
        HPX_HOST_DEVICE
        inline R
        invoke_fused_impl(F&&f, Tuple&& t, pack_c<std::size_t, Is...>)
        {
            using util::get;
            return util::invoke_r<R>(std::forward<F>(f),
                get<Is>(std::forward<Tuple>(t))...);
        }
    }

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
    HPX_HOST_DEVICE HPX_FORCEINLINE
    typename detail::fused_result_of<F&&(Tuple&&)>::type
    invoke_fused(F&& f, Tuple&& t)
    {
        typedef typename detail::fused_result_of<F&&(Tuple&&)>::type R;

        return detail::invoke_fused_impl<R>(
            std::forward<F>(f), std::forward<Tuple>(t),
            typename detail::fused_index_pack<Tuple>::type());
    }

    /// \copydoc invoke_fused
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given sequenced type.
    template <typename R, typename F, typename Tuple>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    R invoke_fused_r(F&& f, Tuple&& t)
    {
        return detail::invoke_fused_impl<R>(
            std::forward<F>(f), std::forward<Tuple>(t),
            typename detail::fused_index_pack<Tuple>::type());
    }
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace functional
    {
        struct invoke_fused
        {
            template <typename F, typename Tuple>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::detail::fused_result_of<F&&(Tuple&&)>::type
            operator()(F&& f, Tuple&& args)
            {
                typedef typename util::detail::fused_result_of<F&&(Tuple&&)>::type R;

                return hpx::util::void_guard<R>(), util::invoke_fused(
                    std::forward<F>(f),
                    std::forward<Tuple>(args));
            }
        };

        template <typename R>
        struct invoke_fused_r
        {
            template <typename F, typename Tuple>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F&& f, Tuple&& args)
            {
                return hpx::util::void_guard<R>(), util::invoke_fused_r<R>(
                    std::forward<F>(f),
                    std::forward<Tuple>(args));
            }
        };
    }
    /// \endcond
}}

#endif
