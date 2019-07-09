//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/thread_description.hpp>

#include <iostream>
#include <sstream>
#include <string>

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
            os << d.get_address(); //-V128
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

    /* The priority of description is id::name, altname, id::address */
    void thread_description::init_from_alternative_name(char const* altname)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION) && !defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        hpx::threads::thread_id_type id = hpx::threads::get_self_id();
        if (id)
        {
            // get the current task description
            thread_description desc = hpx::threads::get_thread_description(id);
            type_ = desc.kind();
            // if the current task has a description, use it.
            if (type_ == data_type_description)
            {
                data_.desc_ = desc.get_description();
            }
            else
            {
                // if there is an alternate name passed in, use it.
                if (altname != nullptr) {
                    type_ = data_type_description;
                    data_.desc_ = altname;
                } else {
                    // otherwise, use the address of the task.
                    HPX_ASSERT(type_ == data_type_address);
                    data_.addr_ = desc.get_address();
                }
            }
        } else {
            type_ = data_type_description;
            data_.desc_ = altname;
        }
#endif
    }
}}
