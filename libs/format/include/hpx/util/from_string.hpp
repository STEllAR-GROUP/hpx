//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/util/bad_lexical_cast.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {

    namespace detail {
        template <typename T, typename Enable = void>
        struct from_string
        {
            static void call(std::string const& value, T& target)
            {
                std::istringstream stream(value);
                stream.exceptions(std::ios_base::failbit);
                stream >> target;
            }
        };

        template <typename T, typename U>
        T check_out_of_range(U const& value)
        {
            U const min = (std::numeric_limits<T>::min)();
            U const max = (std::numeric_limits<T>::max)();
            if (value < min || value > max)
                throw std::out_of_range("from_string: out of range");
            return static_cast<T>(value);
        }

        template <typename T>
        struct from_string<T,
            typename std::enable_if<std::is_integral<T>::value>::type>
        {
            static void call(std::string const& value, int& target)
            {
                target = std::stoi(value);
            }
            static void call(std::string const& value, long& target)
            {
                target = std::stol(value);
            }
            static void call(std::string const& value, long long& target)
            {
                target = std::stoll(value);
            }

            static void call(std::string const& value, unsigned int& target)
            {
                // there is no std::stoui
                unsigned long target_long;
                call(value, target_long);
                target = check_out_of_range<T>(target_long);
            }
            static void call(std::string const& value, unsigned long& target)
            {
                target = std::stoul(value);
            }
            static void call(
                std::string const& value, unsigned long long& target)
            {
                target = std::stoull(value);
            }

            template <typename U>
            static void call(std::string const& value, U& target)
            {
                using promoted_t = decltype(+std::declval<U>());
                static_assert(!std::is_same<promoted_t, U>::value, "");

                promoted_t promoted;
                call(value, promoted);
                target = check_out_of_range<U>(promoted);
            }
        };

        template <typename T>
        struct from_string<T,
            typename std::enable_if<std::is_floating_point<T>::value>::type>
        {
            static void call(std::string const& value, float& target)
            {
                target = std::stof(value);
            }
            static void call(std::string const& value, double& target)
            {
                target = std::stod(value);
            }
            static void call(std::string const& value, long double& target)
            {
                target = std::stold(value);
            }
        };
    }    // namespace detail

    template <typename T>
    T from_string(std::string const& v)
    {
        T target;
        try
        {
            detail::from_string<T>::call(v, target);
        }
        catch (...)
        {
            return detail::throw_bad_lexical_cast<std::string, T>();
        }
        return target;
    }

    template <typename T, typename U>
    T from_string(std::string const& v, U&& default_value)
    {
        T target;
        try
        {
            detail::from_string<T>::call(v, target);
            return target;
        }
        catch (...)
        {
            return std::forward<U>(default_value);
        }
    }

}}    // namespace hpx::util
