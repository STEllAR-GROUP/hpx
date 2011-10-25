//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_STUBS_REMOTE_OBJECT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/remote_object/server/remote_object.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a stubs#distributing_factory class is the client side
    // representation of a \a server#distributing_factory component
    struct remote_object : stub_base<server::remote_object>
    {
        static lcos::promise<void>
        apply_async(naming::id_type const & target_id, hpx::util::function<void(void**)> f)
        {
            typedef server::remote_object::apply_action action_type;
            return lcos::eager_future<action_type>(target_id, f, 0);
        }

        static void 
        apply(naming::id_type const & target_id, hpx::util::function<void(void**)> f)
        {
            apply_async(target_id, f).get();
        }
    };
}}}
#endif
