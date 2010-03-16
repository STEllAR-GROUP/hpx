//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_DISTRIBUTED_MAP_AUG_14_2009_1129AM)
#define HPX_COMPONENTS_SERVER_DISTRIBUTED_MAP_AUG_14_2009_1129AM

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
    /// The distributed_map is an HPX component.
    ///
    template <typename List>
    class HPX_COMPONENT_EXPORT distributed_map
      : public simple_component_base<distributed_map<List> >
    {
    private:
        typedef simple_component_base<distributed_map> base_type;

        // avoid warnings about using this in member initializer list
        distributed_map* This() { return this; }

    public:
        distributed_map();

        typedef hpx::components::server::distributed_map<List> wrapping_type;

        enum actions
        {
            distributed_map_get_local = 0,
            distributed_map_locals = 1
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef List list_type;

        naming::id_type get_local(naming::id_type);

        std::vector<naming::id_type> locals(void);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            distributed_map, naming::id_type, distributed_map_get_local,
            naming::id_type,
            &distributed_map::get_local
        > get_local_action;

        typedef hpx::actions::result_action0<
            distributed_map, std::vector<naming::id_type>, distributed_map_locals,
            &distributed_map::locals
        > locals_action;

    private:
        naming::gid_type gid_;

        hpx::lcos::mutex mtx_;
        std::map<naming::id_type,naming::id_type> map_;
        std::vector<naming::id_type> locals_;
    };

}}}

#endif
