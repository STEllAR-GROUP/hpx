//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_LOCAL_LIST_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_SERVER_LOCAL_LIST_AUG_13_2009_1054PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The local_list is an HPX component.
    ///
    template <typename List>
    class HPX_COMPONENT_EXPORT local_list
      : public simple_component_base<local_list<List> >
    {
    private:
        typedef simple_component_base<local_list> base_type;
        
    public:
        local_list();
        
        //typedef local_list::server::local_list wrapping_type;
        typedef hpx::components::server::local_list<List> wrapping_type;
        
        enum actions
        {
            local_list_append = 0
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef List list_type;

        int append(List);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            local_list, int, local_list_append, List, &local_list::append
        > append_action;

    private:
        List local_list_;
    };

}}}

#endif
