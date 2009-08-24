//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_LOCAL_MAP_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_STUBS_LOCAL_MAP_AUG_13_2009_1054PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/local_map.hpp"

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#local_map class is the client side representation of all
    /// \a server#local_map components
    template <typename List>
    struct local_map
      : components::stubs::stub_base<server::local_map<List> >
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        static int append(naming::id_type gid, List list)
        {
            typedef typename server::local_map<List>::append_action action_type;
            return lcos::eager_future<action_type>(gid, list).get();
        }

        static List get(naming::id_type gid)
        {
            typedef typename server::local_map<List>::get_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }

        static naming::id_type value(naming::id_type gid, naming::id_type key, components::component_type value_type)
        {
            typedef typename server::local_map<List>::value_action action_type;
            return lcos::eager_future<action_type>(gid, key, value_type).get();
        }
    };

}}}

#endif
