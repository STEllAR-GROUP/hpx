//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/exception.hpp>
#include <hpx/components/server/factory.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <hpx/components/server/manage_component.hpp>

namespace hpx { namespace components { namespace server
{
    bool factory::create(hpx::threadmanager::px_thread_self& self,
            components::component_type type, naming::id_type gid)
    {
        switch (type) {
        case accumulator::value:
            server::create<accumulator>(dgas_, gid);
            break;
            
        default:
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    get_component_type_name(type)));
            break;
        }
        return true;
    }
    
}}}

///////////////////////////////////////////////////////////////////////////////
// enable serialization support (these need to be in the global namespace)
BOOST_CLASS_EXPORT(hpx::components::server::factory::create_component_action);

