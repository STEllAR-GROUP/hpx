//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_PROPS_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_PROPS_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/props.hpp"

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#props class is the client side representation of all
    /// \a server#props components
    struct props : stub_base<server::props>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#props instance
        /// with the given \a gid
        static int init(naming::id_type gid, int label)
        {
            typedef server::props::init_action action_type;
            return lcos::eager_future<action_type>(gid, label).get();
        }

        static int color(naming::id_type gid, int d)
        {
            typedef server::props::color_action action_type;
            return lcos::eager_future<action_type>(gid, d).get();
        }

        static double incr(naming::id_type gid, double d)
        {
            typedef server::props::incr_action action_type;
            return lcos::eager_future<action_type>(gid, d).get();
        }

    };

}}}

#endif
