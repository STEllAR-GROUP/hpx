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
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <meta>
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

    template <typename T>
    using base_type_t = std::remove_cv_t<
        std::remove_pointer_t<std::remove_reference_t<std::remove_cv_t<T>>>>;

    // Helper to recursively build the enclosing namespaces of a type
    template <std::meta::info Scope>
    struct scope_builder
    {
        static consteval auto fundamental_type_name() noexcept
        {
            using type = base_type_t<typename[:Scope:]>;
            if constexpr (std::is_same_v<type, signed char>)
            {
                return fixed_string("signed char");
            }
            else if constexpr (std::is_same_v<type, unsigned char>)
            {
                return fixed_string("unsigned char");
            }
            else if constexpr (std::is_same_v<type, short int>)
            {
                return fixed_string("short int");
            }
            else if constexpr (std::is_same_v<type, unsigned short int>)
            {
                return fixed_string("unsigned short int");
            }
            else if constexpr (std::is_same_v<type, int>)
            {
                return fixed_string("int");
            }
            else if constexpr (std::is_same_v<type, unsigned int>)
            {
                return fixed_string("unsigned int");
            }
            else if constexpr (std::is_same_v<type, long int>)
            {
                return fixed_string("long int");
            }
            else if constexpr (std::is_same_v<type, unsigned long int>)
            {
                return fixed_string("unsigned long int");
            }
            else if constexpr (std::is_same_v<type, long long int>)
            {
                return fixed_string("long long int");
            }
            else if constexpr (std::is_same_v<type, unsigned long long int>)
            {
                return fixed_string("unsigned long long int");
            }
            else if constexpr (std::is_same_v<type, char8_t>)
            {
                return fixed_string("char8_t");
            }
            else if constexpr (std::is_same_v<type, char16_t>)
            {
                return fixed_string("char16_t");
            }
            else if constexpr (std::is_same_v<type, char32_t>)
            {
                return fixed_string("char32_t");
            }
            else if constexpr (std::is_same_v<type, wchar_t>)
            {
                return fixed_string("wchar_t");
            }
            else if constexpr (std::is_same_v<type, char>)
            {
                return fixed_string("char");
            }
            else if constexpr (std::is_same_v<type, bool>)
            {
                return fixed_string("bool");
            }
            else if constexpr (std::is_same_v<type, float>)
            {
                return fixed_string("float");
            }
            else if constexpr (std::is_same_v<type, double>)
            {
                return fixed_string("double");
            }
            else if constexpr (std::is_same_v<type, long double>)
            {
                return fixed_string("long double");
            }
            else if constexpr (std::is_same_v<type, void>)
            {
                return fixed_string("void");
            }
            else if constexpr (std::is_same_v<type, std::nullptr_t>)
            {
                return fixed_string("std::nullptr_t");
            }
        }

        static consteval auto get_value() noexcept
        {
            if constexpr (Scope == ^^::)
            {
                return fixed_string("");
            }
            else if constexpr (!std::meta::has_identifier(Scope))
            {
                if constexpr (std::meta::is_type(Scope) &&
                    std::is_fundamental_v<base_type_t<typename[:Scope:]>>)
                {
                    using raw_type = typename[:Scope:];
                    using base_type = std::remove_pointer_t<
                        std::remove_reference_t<std::remove_cv_t<raw_type>>>;

                    constexpr auto base = [] {
                        if constexpr (std::is_const_v<base_type> &&
                            std::is_volatile_v<base_type>)
                            return fixed_string("const volatile ") +
                                fundamental_type_name();
                        else if constexpr (std::is_const_v<base_type>)
                            return fixed_string("const ") +
                                fundamental_type_name();
                        else if constexpr (std::is_volatile_v<base_type>)
                            return fixed_string("volatile ") +
                                fundamental_type_name();
                        else
                            return fundamental_type_name();
                    }();

                    if constexpr (std::is_pointer_v<raw_type>)
                    {
                        if constexpr (std::is_const_v<raw_type> &&
                            std::is_volatile_v<raw_type>)
                        {
                            return base + fixed_string("* const volatile");
                        }
                        else if constexpr (std::is_const_v<raw_type>)
                        {
                            return base + fixed_string("* const");
                        }
                        else if constexpr (std::is_volatile_v<raw_type>)
                        {
                            return base + fixed_string("* volatile");
                        }
                        else
                        {
                            return base + fixed_string("*");
                        }
                    }
                    else if constexpr (std::is_lvalue_reference_v<raw_type>)
                    {
                        return base + fixed_string("&");
                    }
                    else if constexpr (std::is_rvalue_reference_v<raw_type>)
                    {
                        return base + fixed_string("&&");
                    }
                    else
                    {
                        return base;
                    }
                }
                else
                {
                    // Fallback for missed types such as functions, lambdas, etc.
                    constexpr auto name_view =
                        std::meta::display_string_of(Scope);
                    return fixed_string<name_view.size()>(name_view);
                }
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
