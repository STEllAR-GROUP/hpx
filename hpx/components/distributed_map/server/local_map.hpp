//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_LOCAL_MAP_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_SERVER_LOCAL_MAP_AUG_13_2009_1054PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The local_map is an HPX component.
    ///
    template <typename List>
    class HPX_COMPONENT_EXPORT local_map
      : public simple_component_base<local_map<List> >
    {
    private:
        typedef simple_component_base<local_map> base_type;
        
    public:
        local_map();
        
        //typedef local_map::server::local_map wrapping_type;
        typedef hpx::components::server::local_map<List> wrapping_type;
        
        enum actions
        {
            local_map_append = 0,
            local_map_get = 1,
            local_map_value = 2
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef List list_type;

        int append(List);

        List get(void);

        naming::id_type value(naming::id_type, components::component_type value_type);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            local_map, int, local_map_append, List, &local_map::append
        > append_action;

        typedef hpx::actions::result_action0<
            local_map, List, local_map_get, &local_map::get
        > get_action;

        typedef hpx::actions::result_action2<
            local_map, naming::id_type, local_map_value, naming::id_type, components::component_type, &local_map::value
        > value_action;

    private:
        naming::id_type gid_;

        hpx::lcos::mutex mtx_;
        naming::id_type dist_map_;
        List local_map_;
    };

}}}

#endif
