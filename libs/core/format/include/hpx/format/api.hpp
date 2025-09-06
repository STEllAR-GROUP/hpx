//  Copyright (c) 2017-2021 Agustin Berge
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <ios>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace hpx::util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct type_specifier
        {
            static char const* value() noexcept = delete;
        };

#define DECL_TYPE_SPECIFIER(Type, Spec)                                        \
    HPX_CORE_MODULE_EXPORT_EXTERN template <>                                  \
    struct type_specifier<Type>                                                \
    {                                                                          \
        static constexpr char const* value() noexcept                          \
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
        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T,
            bool IsFundamental = std::is_fundamental_v<T>>
        struct formatter
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* ptr)
            {
                // conversion specifier
                char const* conv_spec = "";
                if (spec.empty() || !std::isalpha(spec.back()))
                    conv_spec = type_specifier<T>::value();

                // copy spec to a null terminated buffer
                char format[16];
                int const len =
                    std::snprintf(format, sizeof(format), "%%%.*s%s",
                        static_cast<int>(spec.size()), spec.data(), conv_spec);
                if (len < 0 || len >= static_cast<int>(sizeof(format)))
                {
                    throw std::runtime_error("Not a valid format specifier");
                }

                T const& value = *static_cast<T const*>(ptr);    //-V206
                std::size_t length = std::snprintf(nullptr, 0, format, value);
                std::vector<char> buffer(length + 1);
                length =
                    std::snprintf(buffer.data(), length + 1, format, value);

                os.write(buffer.data(), static_cast<std::streamsize>(length));
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <>
        struct formatter<bool> : formatter<int>
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* ptr)
            {
                int const value = *static_cast<bool const*>(ptr) ? 1 : 0;
                return formatter<int>::call(os, spec, &value);
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <>
        struct formatter<void const*, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, std::string_view /*spec*/, void const* ptr)
            {
                os << ptr;
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct formatter<T const*, /*IsFundamental=*/false>
          : formatter<void const*>
        {
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <>
        struct formatter<char const*, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* ptr)
            {
                auto const value = static_cast<char const*>(ptr);

                // conversion specifier
                if (spec.empty() || spec == "s")
                {
                    os << value;
                }
                else
                {
                    // copy spec to a null terminated buffer
                    char format[16];
                    int const len = std::snprintf(format, sizeof(format),
                        "%%%.*ss", static_cast<int>(spec.size()), spec.data());
                    if (len <= 0 || len >= static_cast<int>(sizeof(format)))
                    {
                        throw std::runtime_error(
                            "Not a valid format specifier");
                    }

                    std::size_t length =
                        std::snprintf(nullptr, 0, format, value);
                    std::vector<char> buffer(length + 1);
                    length =
                        std::snprintf(buffer.data(), length + 1, format, value);

                    os.write(
                        buffer.data(), static_cast<std::streamsize>(length));
                }
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <>
        struct formatter<std::string, /*IsFundamental=*/false>
          : formatter<char const*>
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* ptr)
            {
                std::string const& value =
                    *static_cast<std::string const*>(ptr);

                if (spec.empty() || spec == "s")
                {
                    os.write(value.data(),
                        static_cast<std::streamsize>(value.size()));
                }
                else
                {
                    formatter<char const*>::call(os, spec, value.c_str());
                }
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <>
        struct formatter<std::tm, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* ptr)
            {
                std::tm const& value = *static_cast<std::tm const*>(ptr);

                // conversion specifier
                if (spec.empty())
                    spec = "%c";    // standard date and time string

                // copy spec to a null terminated buffer
                std::string const format(spec);

                std::size_t length;
                std::vector<char> buffer(1);
                buffer.resize(buffer.capacity());
                do
                {
                    length = std::strftime(
                        buffer.data(), buffer.size(), format.c_str(), &value);
                    if (length == 0)
                        buffer.resize(buffer.capacity() * 2);
                } while (length == 0);

                os.write(buffer.data(), static_cast<std::streamsize>(length));
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        void format_value(
            std::ostream& os, std::string_view spec, T const& value)
        {
            if (!spec.empty())
                throw std::runtime_error("Not a valid format specifier");

            os << value;
        }

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        struct formatter<T, /*IsFundamental=*/false>
        {
            static void call(
                std::ostream& os, std::string_view spec, void const* value)
            {
                // ADL customization point
                format_value(os, spec, *static_cast<T const*>(value));
            }
        };

        HPX_CORE_MODULE_EXPORT_EXTERN struct format_arg
        {
            format_arg() = default;

            template <typename T>
            explicit constexpr format_arg(T const& arg) noexcept
              : _data(&arg)
              , _formatter(&detail::formatter<T>::call)
            {
            }

            template <typename T>
            explicit constexpr format_arg(T* arg) noexcept
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {
            }
            template <typename T>
            explicit constexpr format_arg(T const* arg) noexcept
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {
            }

            void operator()(std::ostream& os, std::string_view spec) const
            {
                _formatter(os, spec, _data);
            }

            void const* _data = nullptr;
            void (*_formatter)(
                std::ostream&, std::string_view spec, void const*) = nullptr;
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CORE_MODULE_EXPORT void format_to(std::ostream& os,
            std::string_view format_str, format_arg const* args,
            std::size_t count);

        HPX_CORE_MODULE_EXPORT std::string format(std::string_view format_str,
            format_arg const* args, std::size_t count);
    }    // namespace detail

    // enable using format in variadic contexts
    HPX_CORE_MODULE_EXPORT std::string const& format();

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Args>
    std::string format(std::string_view format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = {
            detail::format_arg(args)..., detail::format_arg()};
        return detail::format(format_str, format_args, sizeof...(Args));
    }

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename... Args>
    std::ostream& format_to(
        std::ostream& os, std::string_view format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = {
            detail::format_arg(args)..., detail::format_arg()};
        detail::format_to(os, format_str, format_args, sizeof...(Args));
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename Range>
        struct format_join
        {
            Range const& rng;
            std::string_view delim;

            friend void format_value(std::ostream& os, std::string_view spec,
                format_join const& value)
            {
                bool first = true;
                for (auto const& elem : value.rng)
                {
                    if (!first)
                        os << value.delim;
                    first = false;

#if defined(__NVCC__)
                    using value_type = typename Range::value_type;
#else
                    using value_type = std::decay_t<decltype(elem)>;
#endif
                    detail::formatter<value_type>::call(os, spec, &elem);
                }
            }
        };
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Range>
    detail::format_join<Range> format_join(
        Range const& range, std::string_view delimiter) noexcept
    {
        return {range, delimiter};
    }
}    // namespace hpx::util
