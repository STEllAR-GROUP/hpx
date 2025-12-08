//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    // Helper to detect if a type is tuple-like
    template <typename T, typename = void>
    struct is_tuple_like : std::false_type
    {
    };

    template <typename T>
    struct is_tuple_like<T,
        std::void_t<decltype(hpx::tuple_size<std::decay_t<T>>::value)>>
      : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

    // Hand-crafted function object allowing to replace a more complex
    // bind(hpx::functional::invoke_fused(), f1, _1)
    HPX_CXX_EXPORT template <typename Result, typename F>
    struct partitioner_iteration
    {
        std::decay_t<F> f_;

        // Overload for tuple-like types
        template <typename T, typename = std::enable_if_t<is_tuple_like_v<T>>>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Result operator()(T&& t)
        {
            return hpx::invoke_fused_r<Result>(f_, HPX_FORWARD(T, t));
        }

        // Overload for non-tuple types (std::size_t from stdexec bulk)
        template <typename T, typename = std::enable_if_t<!is_tuple_like_v<T>>,
            typename = void>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Result operator()(T&& t)
        {
            return HPX_INVOKE_R(Result, f_, HPX_FORWARD(T, t));
        }

        template <std::size_t... Is, typename... Ts>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Result operator()(
            hpx::util::index_pack<Is...>, hpx::tuple<Ts...>& t)
        {
            return HPX_INVOKE(f_, hpx::get<Is>(t)...);
        }

        template <std::size_t... Is, typename... Ts>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Result operator()(
            hpx::util::index_pack<Is...>, hpx::tuple<Ts...>&& t)
        {
            // NOLINTBEGIN(bugprone-use-after-move)
            return HPX_INVOKE(f_, hpx::get<Is>(HPX_MOVE(t))...);
            // NOLINTEND(bugprone-use-after-move)
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & f_;
        }
    };
}    // namespace hpx::parallel::util::detail

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/modules/functional.hpp>

namespace hpx::traits {

    HPX_CXX_EXPORT template <typename Result, typename F>
    struct get_function_address<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        [[nodiscard]] static constexpr std::size_t call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_address<std::decay_t<F>>::call(f.f_);
        }
    };

    HPX_CXX_EXPORT template <typename Result, typename F>
    struct get_function_annotation<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        [[nodiscard]] static constexpr char const* call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_annotation<std::decay_t<F>>::call(f.f_);
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    HPX_CXX_EXPORT template <typename Result, typename F>
    struct get_function_annotation_itt<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        [[nodiscard]] static util::itt::string_handle call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_annotation_itt<std::decay_t<F>>::call(f.f_);
        }
    };
#endif
}    // namespace hpx::traits

#endif
