//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_DISTRIBUTED_SET_AUG_14_2009_1131AM)
#define HPX_COMPONENTS_STUBS_DISTRIBUTED_SET_AUG_14_2009_1131AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/distributed_set.hpp"

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#distributed_set class is the client side representation of
    /// all \a server#distributed_set components
    template <typename List>
    struct distributed_set
      : components::stubs::stub_base<server::distributed_set<List> >
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#distributed_set instance
        /// with the given \a gid
        static naming::id_type get_local(naming::id_type gid, naming::id_type locale)
        {
            typedef typename server::distributed_set<List>::get_local_action
                action_type;
            return lcos::eager_future<action_type>(gid, locale).get();
        }

        static std::vector<naming::id_type>
        locals(naming::id_type gid)
        {
            typedef typename server::distributed_set<List>::locals_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }
    };

}}}

#endif
