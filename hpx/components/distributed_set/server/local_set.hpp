//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_LOCAL_SET_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_SERVER_LOCAL_SET_AUG_13_2009_1054PM

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
    /// The local_set is an HPX component.
    ///
    template <typename Item>
    class HPX_COMPONENT_EXPORT local_set
      : public simple_component_base<local_set<Item> >
    {
    private:
        typedef simple_component_base<local_set> base_type;
        
    public:
        local_set();
        
        //typedef local_set::server::local_set wrapping_type;
        typedef hpx::components::server::local_set<Item> wrapping_type;
        
        enum actions
        {
            local_set_add_item = 0,
            local_set_append = 1,
            local_set_get = 2
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef std::vector<naming::id_type> set_type;

        naming::id_type add_item(naming::id_type);

        int append(set_type);

        set_type get(void);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.

        typedef hpx::actions::result_action1<
            local_set, naming::id_type, local_set_add_item, naming::id_type, &local_set::add_item
        > add_item_action;

        typedef hpx::actions::result_action1<
            local_set, int, local_set_append, set_type, &local_set::append
        > append_action;

        typedef hpx::actions::result_action0<
            local_set, set_type, local_set_get, &local_set::get
        > get_action;

    private:
        naming::id_type gid_;

        lcos::mutex local_set_mtx_;
        set_type local_set_;
    };

}}}

#endif
