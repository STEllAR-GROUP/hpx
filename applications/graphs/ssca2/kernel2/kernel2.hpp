//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_KERNEL2_AUG_14_2009_1044AM)
#define HPX_COMPONENTS_KERNEL2_AUG_14_2009_1044AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/kernel2.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a kernel2 class is the client side representation of a
    /// specific \a server#kernel2 component
    class kernel2
      : public client_base<kernel2, stubs::kernel2>
    {
    private:
        typedef client_base<kernel2, stubs::kernel2> base_type;

    public:
        kernel2(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int
        large_set(naming::id_type G, naming::id_type edge_list)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::large_set(gid_, G, edge_list);
        }

        int
        large_set_local(locality_result local_list,
                        naming::id_type edge_list,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::large_set_local(
                gid_, local_list, edge_list, local_max_lco, global_max_lco);
        }
    };
    
}}

#endif
