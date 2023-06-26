//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/util/to_string.hpp>

#include <ostream>
#include <string>

namespace hpx::threads {

    std::ostream& operator<<(
        std::ostream& os, [[maybe_unused]] thread_description const& d)
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
        os << "<unknown>";
#endif
        return os;
    }

    std::string as_string([[maybe_unused]] thread_description const& desc)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        if (desc.kind() == threads::thread_description::data_type_description)
            return desc ? desc.get_description() : "<unknown>";

        return hpx::util::format("address: {:#x}", desc.get_address());
#else
        return "<unknown>";
#endif
    }

    /* The priority of description is altname, id::name, id::address */
    void thread_description::init_from_alternative_name(
        [[maybe_unused]] char const* altname)
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION) &&                                    \
    !defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        if (altname != nullptr)
        {
            data_.type_ = data_type_description;
            data_.desc_ = altname;
            return;
        }

        hpx::threads::thread_id_type const id = hpx::threads::get_self_id();
        if (id)
        {
            // get the current task description
            thread_description const desc =
                hpx::threads::get_thread_description(id);
            data_.type_ = desc.kind();

            // if the current task has a description, use it.
            if (data_.type_ == data_type_description)
            {
                data_.desc_ = desc.get_description();
            }
            else
            {
                // otherwise, use the address of the task.
                HPX_ASSERT(data_.type_ == data_type_address);
                data_.addr_ = desc.get_address();
            }
        }
        else
        {
            data_.type_ = data_type_description;
            data_.desc_ = "<unknown>";
        }
#endif
    }

    threads::thread_description get_thread_description(
        thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_description() :
                    threads::thread_description("<unknown>");
    }

    threads::thread_description set_thread_description(thread_id_type const& id,
        threads::thread_description const& desc, error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                "hpx::threads::set_thread_description",
                "null thread id encountered");
            return {};
        }
        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->set_description(desc);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_description get_thread_lco_description(
        thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                "hpx::threads::get_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->get_lco_description();
    }

    threads::thread_description set_thread_lco_description(
        thread_id_type const& id, threads::thread_description const& desc,
        error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                "hpx::threads::set_thread_lco_description",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_id_data(id)->set_lco_description(desc);
    }
}    // namespace hpx::threads
