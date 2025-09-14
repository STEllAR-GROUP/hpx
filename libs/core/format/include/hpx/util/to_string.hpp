//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Make HPX inspect tool happy: hpxinspect:noinclude:to_string

#pragma once

#include <hpx/config.hpp>
#include <hpx/format/api.hpp>
#include <hpx/util/bad_lexical_cast.hpp>

#include <string>
#include <type_traits>

namespace hpx::util {

    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T,
            typename Enable = void>
        struct to_string
        {
            [[nodiscard]] static std::string call(T const& value)
            {
                return util::format("{}", value);
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct to_string<T,
            std::enable_if_t<std::is_integral_v<T> ||
                std::is_floating_point_v<T>>>
        {
            [[nodiscard]] static std::string call(T const& value)
            {
                return std::to_string(value);
            }
        };
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    [[nodiscard]] std::string to_string(T const& v)
    {
        try
        {
            return detail::to_string<T>::call(v);
        }
        catch (...)
        {
            return detail::throw_bad_lexical_cast<T, std::string>();
        }
    }
}    // namespace hpx::util
