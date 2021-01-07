//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/util/bad_lexical_cast.hpp>

#include <algorithm>
#include <cstddef>
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
            template <typename Char>
            static void call(std::basic_string<Char> const& value, T& target)
            {
                std::basic_istringstream<Char> stream(value);
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

        template <typename Char>
        void check_only_whitespace(
            std::basic_string<Char> const& s, std::size_t pos)
        {
            auto i = s.begin();
            std::advance(i, pos);
            i = std::find_if(
                i, s.end(), [](int c) { return !std::isspace(c); });

            if (i != s.end())
            {
                throw std::invalid_argument(
                    "from_string: found non-whitespace after token");
            }
        }

        template <typename T>
        struct from_string<T,
            typename std::enable_if<std::is_integral<T>::value>::type>
        {
            template <typename Char>
            static void call(std::basic_string<Char> const& value, int& target)
            {
                std::size_t pos = 0;
                target = std::stoi(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(std::basic_string<Char> const& value, long& target)
            {
                std::size_t pos = 0;
                target = std::stol(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, long long& target)
            {
                std::size_t pos = 0;
                target = std::stoll(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, unsigned int& target)
            {
                // there is no std::stoui
                unsigned long target_long;
                call(value, target_long);
                target = check_out_of_range<T>(target_long);
            }

            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, unsigned long& target)
            {
                std::size_t pos = 0;
                target = std::stoul(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(std::basic_string<Char> const& value,
                unsigned long long& target)
            {
                std::size_t pos = 0;
                target = std::stoull(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char, typename U>
            static void call(std::basic_string<Char> const& value, U& target)
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
            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, float& target)
            {
                std::size_t pos = 0;
                target = std::stof(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, double& target)
            {
                std::size_t pos = 0;
                target = std::stod(value, &pos);
                check_only_whitespace(value, pos);
            }

            template <typename Char>
            static void call(
                std::basic_string<Char> const& value, long double& target)
            {
                std::size_t pos = 0;
                target = std::stold(value, &pos);
                check_only_whitespace(value, pos);
            }
        };
    }    // namespace detail

    template <typename T, typename Char>
    T from_string(std::basic_string<Char> const& v)
    {
        T target;
        try
        {
            detail::from_string<T>::call(v, target);
        }
        catch (...)
        {
            return detail::throw_bad_lexical_cast<std::basic_string<Char>, T>();
        }
        return target;
    }

    template <typename T, typename U, typename Char>
    T from_string(std::basic_string<Char> const& v, U&& default_value)
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
