//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <boost/exception_ptr.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    continuation::continuation()
    {}

    continuation::continuation(naming::id_type const& gid)
      : gid_(gid)
    {
        // Try to resolve the address locally ...
        if(gid && !agas::is_local_address_cached(gid_, addr_))
        {
            addr_ = naming::address();
        }
    }

    continuation::continuation(naming::id_type&& gid)
      : gid_(std::move(gid))
    {
        // Try to resolve the address locally ...
        if(gid && !agas::is_local_address_cached(gid_, addr_))
        {
            addr_ = naming::address();
        }
    }
    continuation::continuation(naming::id_type const& gid, naming::address && addr)
      : gid_(gid)
      , addr_(std::move(addr))
    {
    }

    continuation::continuation(naming::id_type&& gid, naming::address && addr)
      : gid_(std::move(gid))
      , addr_(std::move(addr))
    {
    }

    continuation::continuation(continuation&& o)
      : gid_(std::move(o.gid_))
      , addr_(std::move(o.addr_))
    {}

    continuation& continuation::operator=(continuation&& o)
    {
        gid_ = std::move(o.gid_);
        addr_ = std::move(o.addr_);
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(boost::exception_ptr const& e)
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, this->get_addr(), e);
    }

    void continuation::trigger_error(boost::exception_ptr && e) //-V659
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, this->get_addr(), std::move(e));
    }

    void continuation::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar & gid_ & addr_;
    }

    void continuation::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar & gid_ & addr_;
    }
}}

