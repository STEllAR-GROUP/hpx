//  Copyright (c) 2017-2018 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FORMAT_HPP
#define HPX_UTIL_FORMAT_HPP

#include <hpx/config.hpp>

#include <boost/utility/string_ref.hpp>

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace iostreams {
    template <typename Char, typename Sink>
    struct ostream;
}}

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct type_specifier
        {
            static char const* value() noexcept;
        };

#       define DECL_TYPE_SPECIFIER(Type, Spec)                                \
        template <> struct type_specifier<Type>                               \
        { static char const* value() noexcept { return #Spec; } }             \
        /**/

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

#       undef DECL_TYPE_SPECIFIER

        ///////////////////////////////////////////////////////////////////////
        template <typename T, bool IsFundamental = std::is_fundamental<T>::value>
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
                std::sprintf(format, "%%%.*s%s",
                    (int)spec.size(), spec.data(), conv_spec);

                T const& value = *static_cast<T const*>(ptr);
                std::size_t length = std::snprintf(nullptr, 0, format, value);
                std::vector<char> buffer(length + 1);
                length = std::snprintf(buffer.data(), length + 1, format, value);

                os.write(buffer.data(), length);
            }
        };

        template <>
        struct formatter<bool>
          : formatter<int>
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
        {};

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
                } else {
                    // copy spec to a null terminated buffer
                    char format[16];
                    std::sprintf(format, "%%%.*ss",
                        (int)spec.size(), spec.data());

                    std::size_t length = std::snprintf(nullptr, 0, format, value);
                    std::vector<char> buffer(length + 1);
                    length = std::snprintf(buffer.data(), length + 1, format, value);

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
                std::string const& value = *static_cast<std::string const*>(ptr);

                if (spec.empty() || spec == "s")
                    os.write(value.data(), value.size());
                else
                    formatter<char const*>::call(os, spec, value.c_str());
            }
        };

        template <typename T>
        void format_value(std::ostream& os, boost::string_ref spec, T const& value)
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
            {}

            template <typename T>
            format_arg(T* arg)
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {}
            template <typename T>
            format_arg(T const* arg)
              : _data(arg)
              , _formatter(&detail::formatter<T const*>::call)
            {}

            void operator()(std::ostream& os, boost::string_ref spec) const
            {
                _formatter(os, spec, _data);
            }

            void const* _data;
            void (*_formatter)(std::ostream&, boost::string_ref spec, void const*);
        };

        ///////////////////////////////////////////////////////////////////////
        /*HPX_EXPORT*/ void format_to(
            std::ostream& os,
            boost::string_ref format_str,
            format_arg const* args, std::size_t count);

        /*HPX_EXPORT*/ std::string format(
            boost::string_ref format_str,
            format_arg const* args, std::size_t count);
    }

    template <typename ...Args>
    std::string format(
        boost::string_ref format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = { args..., 0 };
        return detail::format(format_str, format_args, sizeof...(Args));
    }

    template <typename ...Args>
    std::ostream& format_to(
        std::ostream& os,
        boost::string_ref format_str, Args const&... args)
    {
        detail::format_arg const format_args[] = { args..., 0 };
        detail::format_to(os, format_str, format_args, sizeof...(Args));
        return os;
    }

    template <typename Char, typename Sink, typename ...Args>
    hpx::iostreams::ostream<Char, Sink>& format_to(
        hpx::iostreams::ostream<Char, Sink>& os,
        std::string const& format_str, Args const&... args)
    {
        return os << format(format_str, args...);
    }
}}

///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/util/format.hpp>

#include <boost/utility/string_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::size_t format_atoi(
        boost::string_ref str, std::size_t* pos = nullptr) noexcept
    {
        // copy input to a null terminated buffer
        static constexpr std::size_t digits10 =
            std::numeric_limits<std::size_t>::digits10 + 1;
        char buffer[digits10 + 1] = {};
        std::memcpy(buffer, str.data(), (std::min)(str.size(), digits10));

        char const* first = buffer;
        char* last = buffer;
        std::size_t r = std::strtoull(first, &last, 10);
        if (pos != nullptr)
            *pos = last - first;
        return r;
    }

    inline boost::string_ref format_substr(
        boost::string_ref str,
        std::size_t start, std::size_t end = boost::string_ref::npos) noexcept
    {
        return start < str.size()
            ? str.substr(start, end - start) : boost::string_ref{};
    }

    ///////////////////////////////////////////////////////////////////////////
    // replacement-field ::= '{' [arg-id] [':' format-spec] '}'
    struct format_field
    {
        std::size_t arg_id;
        boost::string_ref spec;
    };

    inline format_field parse_field(boost::string_ref field) noexcept
    {
        std::size_t const sep = field.find(':');
        if (sep != field.npos)
        {
            boost::string_ref const arg_id = format_substr(field, 0, sep);
            boost::string_ref const spec = format_substr(field, sep + 1);

            std::size_t const id = format_atoi(arg_id);
            return format_field{id, spec};
        } else {
            std::size_t const id = format_atoi(field);
            return format_field{id, ""};
        }
    }

    inline void format_to(
        std::ostream& os,
        boost::string_ref format_str,
        format_arg const* args, std::size_t /*count*/)
    {
        std::size_t index = 0;
        while (!format_str.empty())
        {
            if (format_str[0] == '{' || format_str[0] == '}')
            {
                HPX_ASSERT(!format_str.empty());
                if (format_str[1] == format_str[0])
                {
                    os.write(format_str.data(), 1); // '{' or '}'
                } else {
                    HPX_ASSERT(format_str[0] != '}');
                    std::size_t const end = format_str.find('}');
                    boost::string_ref field_str = format_substr(format_str, 1, end);
                    format_field const field = parse_field(field_str);
                    format_str.remove_prefix(end - 1);

                    std::size_t const id =
                        field.arg_id ? field.arg_id - 1 : index;
                    args[id](os, field.spec);
                    ++index;
                }
                format_str.remove_prefix(2);
            } else {
                std::size_t const next = format_str.find_first_of("{}");
                std::size_t const count =
                    next != format_str.npos ? next : format_str.size();

                os.write(format_str.data(), count);
                format_str.remove_prefix(count);
            }
        }
    }

    inline std::string format(
        boost::string_ref format_str,
        format_arg const* args, std::size_t count)
    {
        std::ostringstream os;
        detail::format_to(os, format_str, args, count);
        return os.str();
    }
}}}

#endif /*HPX_UTIL_FORMAT_HPP*/
