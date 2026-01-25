//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <hpx/serialization/base_object.hpp>

// This file is only ever included by access.hpp
// but we will still guard against direct inclusion
#if defined(HPX_SERIALIZATION_HAVE_ALLOW_AUTO_GENERATE)

#include <algorithm>
#include <string_view>
#include <vector>

#include <experimental/meta>
#include <ranges>

namespace hpx::serialization::detail {
    template <std::size_t N>
    struct fixed_string
    {
        char data[N]{};

        consteval fixed_string() = default;

        consteval fixed_string(std::string_view str)
        {
            std::copy_n(str.data(), std::min(N - 1, str.size()), data);
            data[N - 1] = '\0';    // Ensure null termination
        }

        [[nodiscard]] constexpr char const* c_str() const noexcept
        {
            return data;
        }
    };

    HPX_CXX_EXPORT template <typename T>
    struct qualified_name_of
    {
        static char const* get() noexcept
        {
            return NULL;
        }
    }

    HPX_CXX_EXPORT template <template <typename...> typename T,
        typename... Args>
    struct qualified_name_of<T<Args...>>
    {
        [[nodiscard]] char const* operator()() const noexcept
        {
            constexpr auto has_parent = [](std::meta::info info_T) {
                return std::meta::parent_of(info_T) !=
                    std::meta::parent_of(std::meta::parent_of(info_T));
            };
            .

                static constexpr auto qualified_name = [&has_parent]() {
                constexpr auto dT = dealias(^^T);

                // Needs to construct the fully qualified name and
                // find each enclosing namespace
                std::vector<std::meta::info> scopes;
                for (auto scope = dT; has_parent(scope);
                     scope = std::meta::parent_of(scope))
                {
                    scopes.push_back(scope);
                }

                // Add all enclosing namespaces in reverse
                // Todo: Should I use ranges? Should we filter global namespace?
                auto const scoped_name = scopes | std::views::reverse |
                    std::views::transform(std::meta::identifier_of) |
                    std::views::join_with(std::string_view("::")) |
                    std::ranges::to<std::string>();

                // Put all the template arguments into a string view
                auto const template_args =
                    std::define_static_array(
                        std::meta::template_arguments_of(dT)) |
                    std::views::transform(std::meta::identifier_of) |
                    std::views::join_with(',') | std::ranges::to<std::string>();

                return scoped_name + "<" + template_args + ">";
            };

            return qualified_name().c_str();
        }
    };
}    // namespace hpx::serialization::detail

#endif
