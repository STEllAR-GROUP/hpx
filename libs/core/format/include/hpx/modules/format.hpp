//  Copyright (c) 2017-2021 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <boost/utility/string_ref.hpp>

#include <cstddef>
#include <ctime>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {

    // format_parse_context
    using format_parse_context = boost::string_ref;

    // format_error
    class HPX_CORE_EXPORT format_error : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
        ~format_error();
    };

    // formatter
    namespace detail {
        struct ostream_formatter
        {
            static void parse(format_parse_context spec)
            {
                if (!spec.empty())
                    throw format_error("Not a valid format specifier");
            }

            template <typename T>
            static void format(std::ostream& os, T const& value)
            {
                os << value;
            }
        };
    }    // namespace detail

    template <typename T>
    struct formatter : detail::ostream_formatter
    {
        static_assert(
            std::is_same<T, std::decay_t<T>>::value, "Bad formatter type");
    };

    namespace detail {
        class HPX_CORE_EXPORT sprintf_formatter
        {
        protected:
            char _format_str[16];
            mutable std::vector<char> _buffer;

            void _format(std::ostream* os, ...) const;

        public:
            void parse(format_parse_context spec, char const* type_specifier);

            template <typename T>
            HPX_FORCEINLINE void format(std::ostream& os, T value) const
            {
                return _format(&os, value);
            }
        };
    }    // namespace detail

#define HPX_SPRINTF_FORMATTER(Type, Spec)                                      \
    template <>                                                                \
    struct formatter<Type> : detail::sprintf_formatter                         \
    {                                                                          \
        HPX_FORCEINLINE void parse(format_parse_context spec)                  \
        {                                                                      \
            return sprintf_formatter::parse(spec, #Spec);                      \
        }                                                                      \
    };

    // character types
    HPX_SPRINTF_FORMATTER(char, c);
    HPX_SPRINTF_FORMATTER(signed char, hhd);
    HPX_SPRINTF_FORMATTER(unsigned char, hhu);

    // integer types
    HPX_SPRINTF_FORMATTER(short, hd);
    HPX_SPRINTF_FORMATTER(int, d);
    HPX_SPRINTF_FORMATTER(long, ld);
    HPX_SPRINTF_FORMATTER(long long, lld);

    HPX_SPRINTF_FORMATTER(unsigned short, hu);
    HPX_SPRINTF_FORMATTER(unsigned int, u);
    HPX_SPRINTF_FORMATTER(unsigned long, lu);
    HPX_SPRINTF_FORMATTER(unsigned long long, llu);

    // floating point types
    HPX_SPRINTF_FORMATTER(float, f);
    HPX_SPRINTF_FORMATTER(double, lf);
    HPX_SPRINTF_FORMATTER(long double, Lf);

    // pointers
    template <typename T>
    struct formatter<T*> : formatter<T const*>
    {
    };

    HPX_SPRINTF_FORMATTER(void const*, p);

    template <>
    struct formatter<std::nullptr_t> : formatter<void const*>
    {
    };

    template <typename T>
    struct formatter<T const*> : formatter<void const*>
    {
    };

    // strings
    namespace detail {
        class HPX_CORE_EXPORT string_formatter : public sprintf_formatter
        {
        public:
            void format(std::ostream& os, char const* value,
                std::size_t len = -1) const;
        };
    }    // namespace detail

    template <>
    struct HPX_CORE_EXPORT formatter<char const*> : detail::string_formatter
    {
        HPX_FORCEINLINE void parse(format_parse_context spec)
        {
            return sprintf_formatter::parse(spec, "s");
        }
    };

    template <>
    struct HPX_CORE_EXPORT formatter<std::string> : detail::string_formatter
    {
        HPX_FORCEINLINE void parse(format_parse_context spec)
        {
            return sprintf_formatter::parse(spec, "s");
        }
        HPX_FORCEINLINE void format(
            std::ostream& os, std::string const& value) const
        {
            return string_formatter::format(os, value.c_str(), value.size());
        }
    };

    // bool
    template <>
    struct formatter<bool> : formatter<char const*>
    {
        HPX_FORCEINLINE void format(std::ostream& os, bool value) const
        {
            return formatter<char const*>::format(os, value ? "true" : "false");
        }
    };

#undef HPX_SPRINTF_FORMATTER

    namespace detail {
        class HPX_CORE_EXPORT strftime_formatter
        {
            std::string _format;
            mutable std::vector<char> _buffer;

        public:
            void parse(format_parse_context spec);
            void format(std::ostream& os, std::tm const& value) const;
        };
    }    // namespace detail

    template <>
    struct formatter<std::tm> : detail::strftime_formatter
    {
    };

    namespace detail {
        template <typename T>
        static void erased_formatter(
            std::ostream& os, format_parse_context spec, void const* data)
        {
            T const& value = *static_cast<T const*>(data);

            formatter<T> fmt;
            fmt.parse(spec);
            fmt.format(os, value);
        }
        template <typename T>
        static void erased_formatter_ptr(
            std::ostream& os, format_parse_context spec, void const* data)
        {
            T const* value = static_cast<T const*>(data);

            formatter<T const*> fmt;
            fmt.parse(spec);
            fmt.format(os, value);
        }

        struct format_arg
        {
            template <typename T>
            format_arg(T const& arg) noexcept
              : formatter(&detail::erased_formatter<T>)
              , data(&arg)
            {
            }
            template <typename T>
            format_arg(T* arg) noexcept
              : formatter(&detail::erased_formatter_ptr<T>)
              , data(arg)
            {
            }
            template <typename T>
            format_arg(T const* arg) noexcept
              : formatter(&detail::erased_formatter_ptr<T>)
              , data(arg)
            {
            }

            void (*formatter)(
                std::ostream&, format_parse_context spec, void const*);
            void const* data;
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

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Range>
        struct format_join
        {
            Range const& rng;
            boost::string_ref delim;
        };

        template <typename Range,
            typename Value =
                std::decay_t<decltype(*std::declval<Range const&>().begin())>>
        struct join_formatter : formatter<Value>
        {
            void format(std::ostream& os, format_join<Range> const& value) const
            {
                bool first = true;
                for (auto const& elem : value.rng)
                {
                    if (!first)
                        os << value.delim;
                    first = false;

                    formatter<Value>::format(os, elem);
                }
            }
        };
    }    // namespace detail

    template <typename Range>
    struct formatter<detail::format_join<Range>> : detail::join_formatter<Range>
    {
    };

    template <typename Range>
    detail::format_join<Range> format_join(
        Range const& range, boost::string_ref delimiter) noexcept
    {
        return {range, delimiter};
    }
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
