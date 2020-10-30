//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/to_string.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace hpx { namespace util {
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
            os << d.get_address();    //-V128
        }
#else
        HPX_UNUSED(d);
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
             << util::to_string(desc.get_address());
        return strm.str();
#else
        HPX_UNUSED(desc);
        return "<unknown>";
#endif
    }

    /* The priority of description is altname, id::name, id::address */
    void thread_description::init_from_alternative_name(char const* altname)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION) &&                                    \
    !defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        if (altname != nullptr)
        {
            type_ = data_type_description;
            data_.desc_ = altname;
            return;
        }
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
                // otherwise, use the address of the task.
                HPX_ASSERT(type_ == data_type_address);
                data_.addr_ = desc.get_address();
            }
        }
        else
        {
            type_ = data_type_description;
            data_.desc_ = "<unknown>";
        }
#else
        HPX_UNUSED(altname);
#endif
    }
}}    // namespace hpx::util

namespace hpx { namespace threads {
    util::thread_description get_thread_description(
        thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_description() :
                    util::thread_description("<unknown>");
    }

    util::thread_description set_thread_description(thread_id_type const& id,
        util::thread_description const& desc, error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_description",
                "null thread id encountered");
            return util::thread_description();
        }
        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->set_description(desc);
    }

    ///////////////////////////////////////////////////////////////////////////
    util::thread_description get_thread_lco_description(
        thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::get_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->get_lco_description();
    }

    util::thread_description set_thread_lco_description(
        thread_id_type const& id, util::thread_description const& desc,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, null_thread_id,
                "hpx::threads::set_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->set_lco_description(desc);
    }
}}    // namespace hpx::threads
