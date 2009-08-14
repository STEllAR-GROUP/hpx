//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_KERNEL2_AUG_13_2009_1040AM)
#define HPX_COMPONENTS_STUBS_KERNEL2_AUG_13_2009_1040AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/kernel2.hpp"

typedef hpx::components::server::kernel2::edge_list_type edge_list_type;

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#kernel2 class is the client side representation of all
    /// \a server#kernel2 components
    struct kernel2
      : components::stubs::stub_base<server::kernel2>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef hpx::components::server::distributing_factory::locality_result
            locality_result;

        static int
        large_set(naming::id_type gid,
                  naming::id_type G,
                  naming::id_type edge_list)
        {
            typedef server::kernel2::large_set_action action_type;
            return lcos::eager_future<action_type>(gid, G, edge_list).get();
        }

        static int
        large_set_local(naming::id_type gid,
                        locality_result local_list,
                        naming::id_type edge_list,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco)
        {
            typedef server::kernel2::large_set_local_action action_type;
            return lcos::eager_future<action_type>(
                       gid, local_list, edge_list, local_max_lco, global_max_lco
                   ).get();
        }
    };

}}}

#endif
