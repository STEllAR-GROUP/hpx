//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void destroy_component(naming::gid_type const& gid,
        naming::address const& addr, error_code& ec)
    {
        using components::stubs::runtime_support;
        agas::gva g (addr.locality_, addr.type_, 1, addr.address_);
        runtime_support::free_component_sync(g, gid, 1);
    }
}}}

