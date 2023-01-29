//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Property, typename Enable = void>
    struct is_scheduling_property : std::false_type
    {
    };

    template <typename Property>
    inline constexpr bool is_scheduling_property_v =
        is_scheduling_property<Property>::value;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // The given property (Tag) is not supported on the given type (first
        // type in Args). Ensure that you are including the correct headers if
        // the property is supported. Alternatively, implement support for the
        // property by overloading tag_invoke for the given property and type.
        // If the property is not required, you can use prefer to fall back to
        // the identity transformation when a property is not supported.
        template <typename Tag, typename... Ts>
        struct property_not_supported;

        template <typename Tag>
        struct property_base : hpx::functional::detail::tag_fallback<Tag>
        {
        private:
            // attempt to improve error messages if property is not supported
            template <typename... Ts>
            friend constexpr auto tag_fallback_invoke(Tag, Ts&&...) noexcept
                -> decltype(property_not_supported<Tag, Ts...>());
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Property>
    struct is_scheduling_property<Property,
        std::enable_if_t<
            std::is_base_of_v<detail::property_base<Property>, Property>>>
      : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct with_priority_t final
      : detail::property_base<with_priority_t>
    {
    } with_priority{};

    inline constexpr struct get_priority_t final
      : hpx::functional::detail::tag_fallback<get_priority_t>
    {
    private:
        // simply return default_ if get_priority is not supported
        template <typename Target>
        friend HPX_FORCEINLINE constexpr hpx::threads::thread_priority
        tag_fallback_invoke(get_priority_t, Target&&) noexcept
        {
            return hpx::threads::thread_priority::default_;
        }
    } get_priority{};

    template <>
    struct is_scheduling_property<get_priority_t> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct with_stacksize_t final
      : detail::property_base<with_stacksize_t>
    {
    } with_stacksize{};

    inline constexpr struct get_stacksize_t final
      : hpx::functional::detail::tag_fallback<get_stacksize_t>
    {
    private:
        // simply return default_ if get_stacksize is not supported
        template <typename Target>
        friend HPX_FORCEINLINE constexpr hpx::threads::thread_stacksize
        tag_fallback_invoke(get_stacksize_t, Target&&) noexcept
        {
            return hpx::threads::thread_stacksize::default_;
        }
    } get_stacksize{};

    template <>
    struct is_scheduling_property<get_stacksize_t> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct with_hint_t final
      : detail::property_base<with_hint_t>
    {
    } with_hint{};

    inline constexpr struct get_hint_t final
      : hpx::functional::detail::tag_fallback<get_hint_t>
    {
    private:
        // simply return default constructed hint if get_hint is not supported
        template <typename Target>
        friend HPX_FORCEINLINE constexpr hpx::threads::thread_schedule_hint
        tag_fallback_invoke(get_hint_t, Target&&) noexcept
        {
            return {};
        }
    } get_hint{};

    template <>
    struct is_scheduling_property<get_hint_t> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct with_annotation_t final
      : detail::property_base<with_annotation_t>
    {
    } with_annotation{};

    inline constexpr struct get_annotation_t final
      : hpx::functional::detail::tag_fallback<get_annotation_t>
    {
    private:
        // simply return nullptr if get_annotation is not supported
        template <typename Target>
        friend HPX_FORCEINLINE constexpr char const* tag_fallback_invoke(
            get_annotation_t, Target&&) noexcept
        {
            return nullptr;
        }
    } get_annotation{};

    template <>
    struct is_scheduling_property<get_annotation_t> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct with_first_core_t final
      : detail::property_base<with_first_core_t>
    {
    } with_first_core{};

    inline constexpr struct get_first_core_t final
      : hpx::functional::detail::tag_fallback<get_first_core_t>
    {
    private:
        // simply return nullptr if get_annotation is not supported
        template <typename Target>
        friend HPX_FORCEINLINE constexpr std::size_t tag_fallback_invoke(
            get_first_core_t, Target&&) noexcept
        {
            return 0;
        }
    } get_first_core{};

    template <>
    struct is_scheduling_property<get_first_core_t> : std::true_type
    {
    };
}    // namespace hpx::execution::experimental
