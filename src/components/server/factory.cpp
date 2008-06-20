//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/server/factory.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <hpx/components/server/manage_component.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>

namespace hpx { namespace components { namespace server
{
    // create a new instance of a component
    threadmanager::thread_state factory::create(
        threadmanager::px_thread_self& self, applier::applier& appl,
        naming::id_type* gid, components::component_type type,
        std::size_t count)
    {
    // create new component instance
        naming::id_type id = naming::invalid_id;
        switch (type) {
        case accumulator::value:
            id = server::create<server::accumulator>(appl, count);
            break;

        default:
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    get_component_type_name(type)));
            break;
        }

    // set result if requested
        if (0 != gid)
            *gid = id;
        return hpx::threadmanager::terminated;
    }

    // delete an existing instance of a component
    threadmanager::thread_state factory::free(
        threadmanager::px_thread_self& self, applier::applier& appl,
        components::component_type type, naming::id_type const& gid,
        std::size_t count)
    {
        switch (type) {
        case accumulator::value:
            server::destroy<server::accumulator>(appl, gid, count);
            break;

        default:
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    get_component_type_name(type)));
            break;
        }
        return hpx::threadmanager::terminated;
    }

}}}

