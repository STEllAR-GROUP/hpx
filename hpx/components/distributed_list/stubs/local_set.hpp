//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_LOCAL_LIST_AUG_13_2009_1054PM)
#define HPX_COMPONENTS_STUBS_LOCAL_LIST_AUG_13_2009_1054PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/local_list.hpp"

namespace hpx { namespace components { namespace stubs
{    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#local_list class is the client side representation of all
    /// \a server#local_list components
    template <typename List>
    struct local_list
      : components::stubs::stub_base<server::local_list<List> >
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        static int append(naming::id_type gid, List list)
        {
            typedef typename server::local_list<List>::append_action action_type;
            return lcos::eager_future<action_type>(gid, list).get();
        }
    };

}}}

#endif
