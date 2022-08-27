//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2022 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
    template <typename Sig, bool Serializable = false>
    class move_only_function;

    template <typename R, typename... Ts, bool Serializable>
    class move_only_function<R(Ts...), Serializable>
      : public util::detail::basic_function<R(Ts...), false, Serializable>
    {
        using base_type =
            util::detail::basic_function<R(Ts...), false, Serializable>;

    public:
        using result_type = R;

        constexpr move_only_function(std::nullptr_t = nullptr) noexcept {}

        move_only_function(move_only_function const&) = delete;
        move_only_function(move_only_function&&) noexcept = default;
        move_only_function& operator=(move_only_function const&) = delete;
        move_only_function& operator=(move_only_function&&) noexcept = default;

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 =
                std::enable_if_t<!std::is_same_v<FD, move_only_function>>,
            typename Enable2 =
                std::enable_if_t<is_invocable_r_v<R, FD&, Ts...>>>
        move_only_function(F&& f)
        {
            assign(HPX_FORWARD(F, f));
        }

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 =
                std::enable_if_t<!std::is_same_v<FD, move_only_function>>,
            typename Enable2 =
                std::enable_if_t<is_invocable_r_v<R, FD&, Ts...>>>
        move_only_function& operator=(F&& f)
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

        // serializable move_only_function is equivalent to
        // hpx::distributed::move_only_function
        template <typename Sig>
        using move_only_function = hpx::move_only_function<Sig, true>;
    }    // namespace distributed
}    // namespace hpx

namespace hpx::util {

    template <typename Sig, bool Serializable = true>
    using unique_function HPX_DEPRECATED_V(1, 8,
        "hpx::util::unique_function is deprecated. Please use "
        "hpx::move_only_function instead.") =
        hpx::move_only_function<Sig, Serializable>;

    template <typename Sig>
    using unique_function_nonser HPX_DEPRECATED_V(1, 8,
        "hpx::util::unique_function_nonser is deprecated. Please use "
        "hpx::move_only_function instead.") = hpx::move_only_function<Sig>;
}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    template <typename Sig, bool Serializable>
    struct get_function_address<hpx::move_only_function<Sig, Serializable>>
    {
        static constexpr std::size_t call(
            hpx::move_only_function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig, bool Serializable>
    struct get_function_annotation<hpx::move_only_function<Sig, Serializable>>
    {
        static constexpr char const* call(
            hpx::move_only_function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig, bool Serializable>
    struct get_function_annotation_itt<
        hpx::move_only_function<Sig, Serializable>>
    {
        static util::itt::string_handle call(
            hpx::move_only_function<Sig, Serializable> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}}    // namespace hpx::traits
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(Sig, F, Name)            \
    HPX_DECLARE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, F, Name)        \
    /**/

#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION(Sig, F, Name)                        \
    HPX_DEFINE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, F, Name)         \
    /**/
