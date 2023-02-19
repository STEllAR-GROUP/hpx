//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2023 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file function.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/basic_function.hpp>
#include <hpx/functional/detail/function_registration.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    /// Class template hpx::function is a general-purpose polymorphic function
    /// wrapper. Instances of hpx::function can store, copy, and invoke any
    /// CopyConstructible Callable target -- functions, lambda expressions, bind
    /// expressions, or other function objects, as well as pointers to member
    /// functions and pointers to data members. The stored callable object is
    /// called the target of hpx::function. If an hpx::function contains no
    /// target, it is called empty. Invoking the target of an empty
    /// hpx::function results in \a hpx#error#bad_function_call exception being
    /// thrown. hpx::function satisfies the requirements of CopyConstructible
    /// and CopyAssignable.
    template <typename Sig, bool Serializable = false>
    class function;

    template <typename R, typename... Ts, bool Serializable>
    class function<R(Ts...), Serializable>
      : public util::detail::basic_function<R(Ts...), true, Serializable>
    {
        using base_type =
            util::detail::basic_function<R(Ts...), true, Serializable>;

    public:
        using result_type = R;

        constexpr function(std::nullptr_t = nullptr) noexcept {}    //-V832

        function(function const&) = default;
        function(function&&) noexcept = default;
        function& operator=(function const&) = default;
        function& operator=(function&&) noexcept = default;

        ~function() = default;

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 = std::enable_if_t<!std::is_same_v<FD, function>>,
            typename Enable2 =
                std::enable_if_t<is_invocable_r_v<R, FD&, Ts...>>>
        function(F&& f)
        {
            assign(HPX_FORWARD(F, f));
        }

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 = std::enable_if_t<!std::is_same_v<FD, function>>,
            typename Enable2 =
                std::enable_if_t<is_invocable_r_v<R, FD&, Ts...>>>
        function& operator=(F&& f)
        {
            assign(HPX_FORWARD(F, f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::empty;
        using base_type::reset;
        using base_type::target;
    };

    namespace distributed {

        // serializable function is equivalent to hpx::distributed::function
        template <typename Sig>
        using function = hpx::function<Sig, true>;
    }    // namespace distributed
}    // namespace hpx

namespace hpx::util {

    template <typename Sig, bool Serializable = true>
    using function HPX_DEPRECATED_V(1, 8,
        "hpx::util::function is deprecated. Please use hpx::function "
        "instead.") = hpx::function<Sig, Serializable>;

    template <typename Sig>
    using function_nonser HPX_DEPRECATED_V(1, 8,
        "hpx::util::function_nonser is deprecated. Please use hpx::function "
        "instead.") = hpx::function<Sig>;
}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx::traits {

    template <typename Sig, bool Serializable>
    struct get_function_address<hpx::function<Sig, Serializable>>
    {
        [[nodiscard]] static constexpr std::size_t call(
            hpx::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig, bool Serializable>
    struct get_function_annotation<hpx::function<Sig, Serializable>>
    {
        [[nodiscard]] static constexpr char const* call(
            hpx::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig, bool Serializable>
    struct get_function_annotation_itt<hpx::function<Sig, Serializable>>
    {
        [[nodiscard]] static util::itt::string_handle call(
            hpx::function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}    // namespace hpx::traits
#endif

////////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, F, Name)                   \
    HPX_DECLARE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)               \
    /**/

#define HPX_UTIL_REGISTER_FUNCTION(Sig, F, Name)                               \
    HPX_DEFINE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)                \
    /**/
