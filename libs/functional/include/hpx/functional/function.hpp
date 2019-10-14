//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FUNCTION_HPP
#define HPX_UTIL_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/functional/detail/basic_function.hpp>
#include <hpx/functional/detail/function_registration.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_callable.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Serializable = true>
    class function;

    template <typename R, typename... Ts, bool Serializable>
    class function<R(Ts...), Serializable>
      : public detail::basic_function<R(Ts...), true, Serializable>
    {
        using base_type = detail::basic_function<R(Ts...), true, Serializable>;

    public:
        typedef R result_type;

        HPX_CONSTEXPR function(std::nullptr_t = nullptr) noexcept {}

        function(function const&) = default;
        function(function&&) noexcept = default;
        function& operator=(function const&) = default;
        function& operator=(function&&) noexcept = default;

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable1 = typename std::enable_if<
                !std::is_same<FD, function>::value>::type,
            typename Enable2 = typename std::enable_if<
                traits::is_invocable_r<R, FD&, Ts...>::value>::type>
        function(F&& f)
        {
            assign(std::forward<F>(f));
        }

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable1 = typename std::enable_if<
                !std::is_same<FD, function>::value>::type,
            typename Enable2 = typename std::enable_if<
                traits::is_invocable_r<R, FD&, Ts...>::value>::type>
        function& operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::empty;
        using base_type::reset;
        using base_type::target;
    };

    template <typename Sig>
    using function_nonser = function<Sig, false>;
}}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    template <typename Sig, bool Serializable>
    struct get_function_address<util::function<Sig, Serializable>>
    {
        static std::size_t call(
            util::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig, bool Serializable>
    struct get_function_annotation<util::function<Sig, Serializable>>
    {
        static char const* call(
            util::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig, bool Serializable>
    struct get_function_annotation_itt<util::function<Sig, Serializable>>
    {
        static util::itt::string_handle call(
            util::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}}    // namespace hpx::traits
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, F, Name)                   \
    HPX_DECLARE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)               \
    /**/

#define HPX_UTIL_REGISTER_FUNCTION(Sig, F, Name)                               \
    HPX_DEFINE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)                \
    /**/

#endif
