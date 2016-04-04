//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_MEMORY_BLOCK_JUN_22_2008_0417PM)
#define HPX_COMPONENTS_STUBS_MEMORY_BLOCK_JUN_22_2008_0417PM

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    template <typename T>
    class access_memory_block;
}}

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a runtime_support class is the client side representation of a
    // \a server#memory_block component
    struct memory_block : stub_base<server::memory_block>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        template <typename T, typename Config>
        static lcos::future<naming::id_type>
        create_async(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_memory_block_action action_type;
            return hpx::async<action_type>(targetgid, count, act);
        }

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        template <typename T, typename Config>
        static naming::id_type
        create(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return create_async(targetgid, count, act).get();
        }

        /// Exposed functionality: get returns either the local memory pointers
        /// or a copy of the remote data.

        HPX_EXPORT static
        lcos::future<components::memory_block_data> get_data_async(
            naming::id_type const& targetgid);

        static components::memory_block_data get_data(
            naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return get_data_async(targetgid).get();
        }

        HPX_EXPORT static
        lcos::future<components::memory_block_data> get_data_async(
            naming::id_type const& targetgid,
            components::memory_block_data const& cfg);

        static components::memory_block_data get_data(
            naming::id_type const& targetgid,
            components::memory_block_data const& cfg)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return get_data_async(targetgid, cfg).get();
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT static
        lcos::future<components::memory_block_data> checkout_async(
            naming::id_type const& targetgid);

        static components::memory_block_data checkout(
            naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return checkout_async(targetgid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT static lcos::future<naming::id_type>
        clone_async(naming::id_type const& targetgid);

        static naming::id_type clone(naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return clone_async(targetgid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        static lcos::future<void>
        checkin_async(naming::id_type const& targetgid,
            components::access_memory_block<T> const& data)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::checkin_action action_type;
            return hpx::async<action_type>(targetgid, data.get_memory_block());
        }

        template <typename T>
        static void checkin(naming::id_type const& targetgid,
            components::access_memory_block<T> const& data)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            checkin_async(targetgid, data).get();
        }
    };
}}}

#endif
