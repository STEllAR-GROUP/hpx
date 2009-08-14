//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_VERTEX_LIST_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_VERTEX_LIST_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/vertex_list.hpp"

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#vertex_list class is the client side representation of all
    /// \a server#vertex_list components
    struct vertex_list 
      : components::stubs::stub_base<server::vertex_list>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef hpx::components::distributing_factory::result_type result_type;

        /// Initialize the server#vertex_list instance 
        /// with the given \a gid
        static int init(naming::id_type gid, components::component_type item_type,
            std::size_t order) 
        {
            typedef server::vertex_list::init_action action_type;
            return lcos::eager_future<action_type>(gid, item_type, order).get();
        }

        static int size(naming::id_type gid)
        {
            typedef server::vertex_list::size_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }

        static naming::id_type at_index(naming::id_type gid, const int index)
        {
        	typedef server::vertex_list::at_index_action action_type;
        	return lcos::eager_future<action_type>(gid, index).get();
        }

        static result_type list(naming::id_type gid)
        {
            typedef server::vertex_list::list_action action_type;
            return lcos::eager_future<action_type>(gid).get();
        }
    };

}}}

#endif
