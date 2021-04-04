///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/format.hpp>

#include <hpx/assert.hpp>

#include <boost/utility/string_ref.hpp>

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace util {
    format_error::~format_error() = default;
}}    // namespace hpx::util

namespace hpx { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    void sprintf_formatter::parse(
        format_parse_context spec, char const* type_specifier)
    {
        // conversion specifier
        char const* conv_spec = "";
        if (spec.empty() || !std::isalpha(spec.back()))
            conv_spec = type_specifier;

        // copy spec to a null terminated buffer
        std::snprintf(_format_str, sizeof(_format_str), "%%%.*s%s",
            (int) spec.size(), spec.data(), conv_spec);
    }

    void sprintf_formatter::_format(std::ostream* os, ...) const
    {
        std::va_list arg1;
        va_start(arg1, os);
        std::va_list arg2;
        va_copy(arg2, arg1);

        std::size_t length = 0;
        if (_buffer.empty())
            _buffer.resize(32);

        _buffer.resize(_buffer.capacity());
        length =
            std::vsnprintf(_buffer.data(), _buffer.size(), _format_str, arg1);
        va_end(arg1);
        if (length > _buffer.size())
        {
            _buffer.resize(length);
            length = std::vsnprintf(
                _buffer.data(), _buffer.size(), _format_str, arg2);
        }
        va_end(arg2);

        os->write(_buffer.data(), length);
    }

    ///////////////////////////////////////////////////////////////////////////
    void string_formatter::format(
        std::ostream& os, char const* value, std::size_t len) const
    {
        if (std::strcmp(_format_str, "%s") == 0)
        {
            os.write(value, len == std::size_t(-1) ? std::strlen(value) : len);
            return;
        }

        return sprintf_formatter::_format(&os, value);
    }

    ///////////////////////////////////////////////////////////////////////////
    void strftime_formatter::parse(format_parse_context spec)
    {
        // conversion specifier
        if (spec.empty())
            spec = "%c";    // standard date and time string

        // copy spec to a null terminated buffer
        _format = spec.to_string();
    }

    void strftime_formatter::format(
        std::ostream& os, std::tm const& value) const
    {
        std::size_t length = 0;
        if (_buffer.empty())
            _buffer.resize(32);
        do
        {
            _buffer.resize(_buffer.capacity());
            length = std::strftime(
                _buffer.data(), _buffer.size(), _format.c_str(), &value);
            if (length == 0)
                _buffer.resize(_buffer.capacity() * 2);
        } while (length == 0);

        os.write(_buffer.data(), length);
    }

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

    inline boost::string_ref format_substr(boost::string_ref str,
        std::size_t start, std::size_t end = boost::string_ref::npos) noexcept
    {
        return start < str.size() ? str.substr(start, end - start) :
                                    boost::string_ref{};
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
        }
        else
        {
            std::size_t const id = format_atoi(field);
            return format_field{id, ""};
        }
    }

    void format_to(std::ostream& os, boost::string_ref format_str,
        format_arg const* args, std::size_t count)
    {
        std::size_t index = 0;
        while (!format_str.empty())
        {
            if (format_str[0] == '{' || format_str[0] == '}')
            {
                HPX_ASSERT(!format_str.empty());
                if (format_str[1] == format_str[0])
                {
                    os.write(format_str.data(), 1);    // '{' or '}'
                }
                else
                {
                    HPX_ASSERT(format_str[0] != '}');
                    std::size_t const end = format_str.find('}');
                    boost::string_ref field_str =
                        format_substr(format_str, 1, end);
                    format_field const field = parse_field(field_str);
                    format_str.remove_prefix(end - 1);

                    std::size_t const id =
                        field.arg_id ? field.arg_id - 1 : index;
                    HPX_ASSERT(id < count);
                    (void)count;
                    args[id].formatter(os, field.spec, args[id].data);
                    ++index;
                }
                format_str.remove_prefix(2);
            }
            else
            {
                std::size_t const next = format_str.find_first_of("{}");
                std::size_t const count =
                    next != format_str.npos ? next : format_str.size();

                os.write(format_str.data(), count);
                format_str.remove_prefix(count);
            }
        }
    }

    std::string format(
        boost::string_ref format_str, format_arg const* args, std::size_t count)
    {
        std::ostringstream os;
        detail::format_to(os, format_str, args, count);
        return std::move(os).str();
    }
}}}    // namespace hpx::util::detail
