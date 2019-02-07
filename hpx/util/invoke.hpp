//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#ifndef HPX_UTIL_INVOKE_HPP
#define HPX_UTIL_INVOKE_HPP

#include <hpx/config.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/void_guard.hpp>

#include <boost/ref.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename C>
        struct invoke_mem_obj
        {
            T C::*f;

            HPX_CONSTEXPR HPX_HOST_DEVICE
            invoke_mem_obj(T C::*mem_obj) noexcept
              : f(mem_obj)
            {}

            // t0.*f
            template <typename T0>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                util::invoke_result<T C::*, T0>
            >::type::type
            operator()(T0&& v0) const noexcept
            {
                return std::forward<T0>(v0).*f;
            }

            // (*t0).*f
            template <typename T0>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                util::invoke_result<T C::*, T0>
            >::type::type
            operator()(T0&& v0) const noexcept(noexcept(*std::forward<T0>(v0)))
            {
                return (*std::forward<T0>(v0)).*f;
            }
        };

        template <typename T, typename C>
        struct invoke_mem_fun
        {
            T C::*f;

            HPX_CONSTEXPR HPX_HOST_DEVICE
            invoke_mem_fun(T C::*mem_fun) noexcept
              : f(mem_fun)
            {}

            // (t0.*f)(t1, ..., tN)
            template <typename T0, typename ...Ts>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                util::invoke_result<T C::*, T0, Ts...>
            >::type::type
            operator()(T0&& v0, Ts&&... vs) const
            {
                return (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
            }

            // ((*t0).*f)(t1, ..., tN)
            template <typename T0, typename ...Ts>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                util::invoke_result<T C::*, T0, Ts...>
            >::type::type
            operator()(T0&& v0, Ts&&... vs) const
            {
                return ((*std::forward<T0>(v0)).*f)(std::forward<Ts>(vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename FD = typename std::decay<F>::type>
        struct dispatch_invoke
        {
            using type = F&&;
        };

        template <typename F, typename T, typename C>
        struct dispatch_invoke<F, T C::*>
        {
            using type = typename std::conditional<
                    std::is_function<T>::value,
                    invoke_mem_fun<T, C>,
                    invoke_mem_obj<T, C>
                >::type;
        };

        // support boost::[c]ref, which is not callable as std::[c]ref
        template <typename F, typename X>
        struct dispatch_invoke<F, ::boost::reference_wrapper<X> >
        {
            using type = X&;
        };
    }

    /// Invokes the given callable object f with the content of
    /// the argument pack vs
    ///
    /// \param f Requires to be a callable object.
    ///          If f is a member function pointer, the first argument in
    ///          the pack will be treated as the callee (this object).
    ///
    /// \param vs An arbitrary pack of arguments
    ///
    /// \returns The result of the callable object when it's called with
    ///          the given argument types.
    ///
    /// \throws std::exception like objects thrown by call to object f
    ///         with the argument types vs.
    ///
    /// \note This function is similar to `std::invoke` (C++17)
    template <typename F, typename ...Ts>
    HPX_CONSTEXPR HPX_HOST_DEVICE
    typename util::invoke_result<F, Ts...>::type
    invoke(F&& f, Ts&&... vs)
    {
        using invoke_impl = typename detail::dispatch_invoke<F>::type;
        return invoke_impl(std::forward<F>(f))(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \copydoc invoke
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given argument types vs.
    template <typename R, typename F, typename ...Ts>
    HPX_CONSTEXPR HPX_HOST_DEVICE
    R invoke_r(F&& f, Ts&&... vs)
    {
        using invoke_impl = typename detail::dispatch_invoke<F>::type;
        return util::void_guard<R>(), invoke_impl(std::forward<F>(f))(
                std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace functional
    {
        struct invoke
        {
            template <typename F, typename... Ts>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename util::invoke_result<F, Ts...>::type
            operator()(F && f, Ts &&... vs)
            {
                return util::invoke(std::forward<F>(f),
                    std::forward<Ts>(vs)...);
            }
        };

        template <typename R>
        struct invoke_r
        {
            template <typename F, typename... Ts>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            R operator()(F && f, Ts &&... vs)
            {
                return util::invoke_r<R>(std::forward<F>(f),
                    std::forward<Ts>(vs)...);
            }
        };
    }
}}

#endif
