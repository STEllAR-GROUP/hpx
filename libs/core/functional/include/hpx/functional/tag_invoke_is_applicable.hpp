//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>

#if defined(DOXYGEN)
namespace hpx { namespace functional {

    /// The `hpx::functional::is_tag_invoke_applicable` trait is used to impose
    /// general constraints onto the object that is derived from
    /// `hpx::functional::tag`, `hpx::functional::tag_fallback`, or
    /// `hpx::functional::tag_priority_fallback`, or their noexcept variations
    /// (in the following referred to as 'the CPO'). The value of this trait
    /// is either `std::true_type` (if the CPO does not expose any additional
    /// constraints), or evaluates to the value that is exposed by the CPO using
    /// a constant bool variable named `is_applicable_v`. For instance:
    ///
    /// struct my_cpo_t : hpx::functional::tag<my_cpo_t>
    /// {
    ///     template <typename... Args>
    ///     static constexpr bool is_applicable_v = ...;
    /// } my_cpo{};
    ///
    /// where `Args...` are the types of the arguments the CPO was invoked with.
    /// The tag_invoke machinery will static_assert if this trait evaluates to
    /// `std::false_type`.

}}    // namespace hpx::functional

#else

namespace hpx { namespace functional {

    namespace detail {

        template <typename Tag, typename Pack, typename Enable = void>
        struct is_tag_invoke_applicable : std::true_type
        {
        };

        template <typename Tag, typename... Args>
        struct is_tag_invoke_applicable<Tag, util::pack<Args...>,
            util::always_void_t<decltype(
                Tag::template is_applicable_v<Args...>)>>
          : std::integral_constant<bool, Tag::template is_applicable_v<Args...>>
        {
        };
    }    // namespace detail

    template <typename Tag, typename... Args>
    struct is_tag_invoke_applicable
      : detail::is_tag_invoke_applicable<Tag, util::pack<std::decay_t<Args>...>>
    {
    };

    template <typename Tag, typename... Args>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_tag_invoke_applicable_v =
        is_tag_invoke_applicable<Tag, Args...>::value;

}}    // namespace hpx::functional

#endif
