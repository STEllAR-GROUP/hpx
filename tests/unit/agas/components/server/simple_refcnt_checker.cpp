////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>

#include <sstream>

#include <tests/unit/agas/components/server/simple_refcnt_checker.hpp>

namespace hpx { namespace test { namespace server
{

simple_refcnt_checker::~simple_refcnt_checker()
{
    const boost::uint32_t prefix_ = get_locality_id();
    const naming::gid_type this_ = get_base_gid();

    std::ostringstream strm;

    if (!references_.empty())
    {
        strm << ( boost::format("[%1%/%2%]: held references\n")
                % prefix_ % this_);

        for (naming::id_type const& ref : references_)
        {
            strm << "  " << ref << " "
                 << naming::get_management_type_name(ref.get_management_type())
                 << "\n";
        }

        // Flush garbage collection.
        references_.clear();
        agas::garbage_collect();
    }

    if (naming::invalid_id != target_)
    {
        strm << ( boost::format("[%1%/%2%]: destroying object\n")
                % prefix_ % this_);

        strm << ( boost::format("[%1%/%2%]: triggering flag %3%\n")
                % prefix_ % this_ % target_);

        hpx::trigger_lco_event(target_);
    }

    std::string const str = strm.str();

    if (!str.empty())
        cout << str << flush;
}

}}}

