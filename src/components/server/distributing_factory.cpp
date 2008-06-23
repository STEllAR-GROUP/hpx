//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/components/server/runtime_support.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <hpx/components/server/distributing_factory.hpp>
#include <hpx/components/server/manage_component.hpp>
#include <hpx/components/continuation_impl.hpp>

namespace hpx { namespace components { namespace server { namespace detail
{
    // create a new instance of a component
    threadmanager::thread_state distributing_factory::create(
        threadmanager::px_thread_self& self, applier::applier& appl,
        naming::id_type* gid, components::component_type type,
        std::size_t count)
    {
    // set result if requested
        if (0 != gid)
            *gid = naming::invalid_id;
        return hpx::threadmanager::terminated;
    }

}}}}

