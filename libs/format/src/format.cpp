///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/modules/format.hpp>

#include <boost/utility/string_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

namespace hpx { namespace util { namespace detail {
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
                    args[id](os, field.spec);
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
        return os.str();
    }
}}}    // namespace hpx::util::detail
