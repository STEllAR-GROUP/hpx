//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_MEMORY_BLOCK_JUN_22_2008_0417PM)
#define HPX_COMPONENTS_STUBS_MEMORY_BLOCK_JUN_22_2008_0417PM

#include <boost/bind.hpp>

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a runtime_support class is the client side representation of a 
    // \a server#memory_block component
    class memory_block : public stub_base<server::memory_block>
    {
    public:
        /// Create a client side representation for any existing 
        /// \a server#memory_block instance
        memory_block(applier::applier& app) 
          : stub_base<server::memory_block>(app)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        static lcos::future_value<components::memory_block_data> get_async(
            applier::applier& appl, naming::id_type const& targetgid) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::get_action action_type;
            typedef components::memory_block_data data_type;
            return lcos::eager_future<action_type, data_type>(appl, targetgid);
        }

        static components::memory_block_data get(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& targetgid) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return get_async(appl, targetgid).get(self);
        }

        /// Exposed functionality: get returns either the local memory pointers
        /// or a copy of the remote data.
        lcos::future_value<components::memory_block_data> get_async(
            naming::id_type const& gid) 
        {
            return get_async(appl_, gid);
        }

        /// 
        components::memory_block_data get(threads::thread_self& self, 
            naming::id_type const& gid) 
        {
            return get(self, appl_, gid);
        }
    };

}}}

#endif
