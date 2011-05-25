//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

namespace hpx { namespace components
{

bool bind_gid (naming::gid_type const& gid_, naming::address const& addr)
{
    return hpx::applier::get_applier().get_agas_client().bind(gid_, addr);
}

void unbind_gid (naming::gid_type const& gid_)
{
    if (gid_) 
        hpx::applier::get_applier().get_agas_client().unbind(gid_);
}

}}

