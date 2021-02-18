//  Copyright (c) 2017-2018 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <boost/utility/string_ref.hpp>

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct type_specifier
        {
            static char const* value() noexcept;
        };

#define DECL_TYPE_SPECIFIER(Type, Spec)                                        \
    template <>                                                                \
    struct type_specifier<Type>                                                \
    {                                                                          \
        static char const* value() noexcept                                    \
        {                                                                      \
            return #Spec;                                                      \
        }                                                                      \
    }

        DECL_TYPE_SPECIFIER(char, c);
        DECL_TYPE_SPECIFIER(wchar_t, lc);

        DECL_TYPE_SPECIFIER(signed char, hhd);
        DECL_TYPE_SPECIFIER(short, hd);
        DECL_TYPE_SPECIFIER(int, d);
        DECL_TYPE_SPECIFIER(long, ld);
        DECL_TYPE_SPECIFIER(long long, lld);

        DECL_TYPE_SPECIFIER(unsigned char, hhu);
        DECL_TYPE_SPECIFIER(unsigned short, hu);
        DECL_TYPE_SPECIFIER(unsigned int, u);
        DECL_TYPE_SPECIFIER(unsigned long, lu);
        DECL_TYPE_SPECIFIER(unsigned long long, llu);

        DECL_TYPE_SPECIFIER(float, f);
        DECL_TYPE_SPECIFIER(double, lf);
        DECL_TYPE_SPECIFIER(long double, Lf);

        // the following type-specifiers are used elsewhere, we add them for
        // completeness
        DECL_TYPE_SPECIFIER(char const*, s);
        DECL_TYPE_SPECIFIER(wchar_t const*, ls);

#undef DECL_TYPE_SPECIFIER

        ///////////////////////////////////////////////////////////////////////
        template <typename T,
            bool IsFundamental = std::is_fundamental<T>::value>
        struct formatter
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* ptr)
            {
                // conversion specifier
                char const* conv_spec = "";
                if (spec.empty() || !std::isalpha(spec.back()))
                    conv_spec = type_specifier<T>::value();

                // copy spec to a null terminated buffer
                char format[16];
                std::sprintf(format, "%%%.*s%s", (int) spec.size(), spec.data(),
                    conv_spec);

                T const& value = *static_cast<T const*>(ptr);
                std::size_t length = std::snprintf(nullptr, 0, format, value);
                std::vector<char> buffer(length + 1);
                length =
                    std::snprintf(buffer.data(), length + 1, format, value);

                os.write(buffer.data(), length);
            }
        };

        template <>
        struct formatter<bool> : formatter<int>
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* ptr)
            {
                int const value = *static_cast<bool const*>(ptr) ? 1 : 0;
                return formatter<int>::call(os, spec, &value);
            }
        };

        template <>
        struct formatter<void const*, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, boost::string_ref /*spec*/, void const* ptr)
            {
                os << ptr;
            }
        };

        template <typename T>
        struct formatter<T const*, /*IsFundamental=*/false>
          : formatter<void const*>
        {
        };

        template <>
        struct formatter<char const*, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* ptr)
            {
                char const* value = static_cast<char const*>(ptr);

                // conversion specifier
                if (spec.empty() || spec == "s")
                {
                    os << value;
                }
                else
                {
                    // copy spec to a null terminated buffer
                    char format[16];
                    std::sprintf(
                        format, "%%%.*ss", (int) spec.size(), spec.data());

                    std::size_t length =
                        std::snprintf(nullptr, 0, format, value);
                    std::vector<char> buffer(length + 1);
                    length =
                        std::snprintf(buffer.data(), length + 1, format, value);

                    os.write(buffer.data(), length);
                }
            }
        };

        template <>
        struct formatter<std::string, /*IsFundamental=*/false>
          : formatter<char const*>
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* ptr)
            {
                std::string const& value =
                    *static_cast<std::string const*>(ptr);

                if (spec.empty() || spec == "s")
                    os.write(value.data(), value.size());
                else
                    formatter<char const*>::call(os, spec, value.c_str());
            }
        };

        template <>
        struct formatter<std::tm, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* ptr)
            {
                std::tm const& value = *static_cast<std::tm const*>(ptr);

                // conversion specifier
                if (spec.empty())
                    spec = "%c";    // standard date and time string

                // copy spec to a null terminated buffer
                std::string format(spec.to_string());

                std::size_t length = 0;
                std::vector<char> buffer(1);
                buffer.resize(buffer.capacity());
                do
                {
                    length = std::strftime(
                        buffer.data(), buffer.size(), format.c_str(), &value);
                    if (length == 0)
                        buffer.resize(buffer.capacity() * 2);
                } while (length == 0);

                os.write(buffer.data(), length);
            }
        };

        template <typename T>
        void format_value(
            std::ostream& os, boost::string_ref spec, T const& value)
        {
            if (!spec.empty())
                throw std::runtime_error("Not a valid format specifier");

            os << value;
        }

        template <typename T>
        struct formatter<T, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, boost::string_ref spec, void const* value)
            {
                // ADL customization point
                format_value(os, spec, *static_cast<T const*>(value));
            }
        };

        struct format_arg
        {
            template <typename T>
            format_arg(T const& arg)
              : _data(&arg)
              , _formatter(&detail::formatter<T>::call)
            {
            }

            template <typename T>
            format_arg(T* arg)
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {
            }
            template <typename T>
            format_arg(T const* arg)
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {
            }

            void operator()(std::ostream& os, boost::string_ref spec) const
            {
                _formatter(os, spec, _data);
            }

            void const* _data;
            void (*_formatter)(
                std::ostream&, boost::string_ref spec, void const*);
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CORE_EXPORT void format_to(std::ostream& os,
            boost::string_ref format_str, format_arg const* args,
            std::size_t count);

        HPX_CORE_EXPORT std::string format(boost::string_ref format_str,
            format_arg const* args, std::size_t count);
    }    // namespace detail

    template <typename... Args>
    std::string format(boost::string_ref format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = {args..., 0};
        return detail::format(format_str, format_args, sizeof...(Args));
    }

    template <typename... Args>
    std::ostream& format_to(
        std::ostream& os, boost::string_ref format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = {args..., 0};
        detail::format_to(os, format_str, format_args, sizeof...(Args));
        return os;
    }
}}    // namespace hpx::util
