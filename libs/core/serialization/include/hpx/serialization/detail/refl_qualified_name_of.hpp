//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/config/defines.hpp>

#if defined(HPX_SERIALIZATION_HAVE_ALLOW_AUTO_GENERATE)

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <experimental/meta>
#include <ranges>

namespace hpx::serialization::detail {
    // A simple wrapper around a char array that can be constructed
    // and concatenated at compile time.
    template <std::size_t N>
    struct fixed_string
    {
        char data[N + 1]{};    // +1 for null terminator
        static constexpr std::size_t size = N;

        consteval fixed_string(std::string_view sv) noexcept
        {
            for (std::size_t i = 0; i < sv.size(); ++i)
                data[i] = sv[i];

            data[sv.size()] = '\0';
        }

        consteval fixed_string(char const (&s)[N + 1]) noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                data[i] = s[i];

            data[N] = '\0';
        }

        template <std::size_t M>
        consteval auto operator+(fixed_string<M> const& other) const noexcept
        {
            fixed_string<N + M> res("");
            for (std::size_t i = 0; i < N; ++i)
                res.data[i] = data[i];
            for (std::size_t i = 0; i < M; ++i)
                res.data[N + i] = other.data[i];

            res.data[N + M] = '\0';
            return res;
        }
    };

    // Deduction guide for fixed_string
    template <std::size_t N>
    fixed_string(char const (&)[N]) -> fixed_string<N - 1>;

    // Helper to recursively build the enclosing namespaces of a type
    template <std::meta::info Scope>
    struct scope_builder
    {
        static consteval auto get_value() noexcept
        {
            if constexpr (Scope == ^^::)
            {
                return fixed_string("");
            }
            else if constexpr (!std::meta::has_identifier(Scope))
            {
                // For types without identifiers (primitives,
                // pointers, etc.), use display_string_of as the
                // fallback,  this should guarantee uniqueness of
                // the string made for a type even though
                // display_string_of is implementationd defined
                constexpr auto name_view = std::meta::display_string_of(Scope);
                return fixed_string<name_view.size()>(name_view);
            }
            else
            {
                constexpr auto parent = std::meta::parent_of(Scope);
                constexpr auto prefix = scope_builder<parent>::value;
                constexpr auto id = std::meta::identifier_of(Scope);
                constexpr auto name = fixed_string<id.size()>(id);

                if constexpr (prefix.size > 0)
                {
                    return prefix + fixed_string("::") + name;
                }
                else
                {
                    return name;
                }
            }
        }
        static constexpr auto value = get_value();
    };

    template <auto StringGetter>
    static consteval auto make_fixed() noexcept
    {
        constexpr std::string_view sv = StringGetter();
        return fixed_string<sv.size()>(sv);
    }

    // Primary template for non-template types
    HPX_CXX_CORE_EXPORT template <typename T>
    struct qualified_name_of
    {
    private:
        static constexpr auto dT = dealias(^^T);
        static constexpr auto storage = scope_builder<dT>::value;

    public:
        [[nodiscard]] static constexpr char const* get() noexcept
        {
            return storage.data;
        }
    };

    // Partial specialization for template types
    HPX_CXX_CORE_EXPORT template <template <typename...> typename T,
        typename... Args>
    struct qualified_name_of<T<Args...>>
    {
    private:
        static constexpr auto dT = dealias(^^T);
        static constexpr auto scoped_name = scope_builder<dT>::value;

        template <std::size_t I, typename Arg>
        static consteval auto fragment() noexcept
        {
            if constexpr (I == 0)
                return make_fixed<qualified_name_of<Arg>::get>();
            else
                return fixed_string(",") +
                    make_fixed<qualified_name_of<Arg>::get>();
        }

        static consteval auto make_args_list() noexcept
        {
            return ([]<std::size_t... Is>(std::index_sequence<Is...>) {
                return (fragment<Is, Args>() + ...);
            })(std::make_index_sequence<sizeof...(Args)>{});
        }

        // Recursively call qualified_name_of for each arg
        static consteval auto get_args_name() noexcept
        {
            if constexpr (sizeof...(Args) == 0)
            {
                return fixed_string("");
            }
            else
            {
                // The lambda templ param trick allows this computation
                // to be done at compile time.
                return make_args_list();
            }
        }

        // Final string
        static constexpr auto storage = scoped_name + fixed_string("<") +
            get_args_name() + fixed_string(">");

    public:
        [[nodiscard]] static constexpr char const* get() noexcept
        {
            return storage.data;
        }
    };
}    // namespace hpx::serialization::detail

#endif
