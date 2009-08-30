//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_LOCAL_SET_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_STUBS_LOCAL_SET_AUG_13_2009_1054PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/local_set.hpp"

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#local_set class is the client side representation of all
    /// \a server#local_set components
    template <typename Item>
    struct local_set
      : components::stubs::stub_base<server::local_set<Item> >
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef std::vector<naming::id_type> set_type;

        static naming::id_type add_item(naming::id_type gid, naming::id_type item=naming::invalid_id)
        {
            typedef typename server::local_set<Item>::add_item_action action_type;
            return lcos::eager_future<action_type>(gid, item).get();
        }

        static int append(naming::id_type gid, set_type list)
        {
            typedef typename server::local_set<Item>::append_action action_type;
            return lcos::eager_future<action_type>(gid, list).get();
        }

        static set_type get(naming::id_type gid)
        {
            typedef typename server::local_set<Item>::get_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }
    };

}}}

#endif
