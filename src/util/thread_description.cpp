//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <iostream>
#include <sstream>

namespace hpx { namespace util
{
    std::ostream& operator<<(std::ostream& os, thread_description const& d)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        if (d.kind() == thread_description::data_type_description)
        {
            os << d.get_description();
        }
        else
        {
            HPX_ASSERT(d.kind() == thread_description::data_type_address);
            os << d.get_address();
        }
#else
        os << "<unknown>";
#endif
        return os;
    }

    std::string as_string(thread_description const& desc)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        if (desc.kind() == util::thread_description::data_type_description)
            return desc ? desc.get_description() : "<unknown>";

        std::stringstream strm;
        strm << "address: 0x" << std::hex
             << util::safe_lexical_cast<std::string>(desc.get_address());
        return strm.str();
#else
        return "<unknown>";
#endif
    }

    void thread_description::init_from_alternative_name(char const* altname)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION) && !defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        type_ = data_type_description;
        data_.desc_ = altname;
        if (altname == 0)
        {
            hpx::threads::thread_id_type id = hpx::threads::get_self_id();
            if (id) *this = hpx::threads::get_thread_description(id);
        }
#endif
    }
}}
