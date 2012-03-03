//  Copyright (c) 2007-2012 Hartmut Kaiser
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
//         template <typename T>
//         static lcos::promise<naming::id_type, naming::gid_type>
//         create_async(naming::id_type const& targetgid, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // Create an eager_future, execute the required action,
//             // we simply return the initialized promise, the caller needs
//             // to call get() on the return value to obtain the result
//             typedef server::runtime_support::create_memory_block_action action_type;
//             return lcos::eager_future<action_type, naming::id_type>(targetgid, count, act);
//         }
//
//         /// Create a new component \a type using the runtime_support with the
//         /// given \a targetgid. Block for the creation to finish.
//         template <typename T>
//         static naming::id_type
//         create(naming::id_type const& targetgid, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // The following get yields control while the action above
//             // is executed and the result is returned to the eager_future
//             return create_async(targetgid, count, act).get();
//         }

        template <typename T, typename Config>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_async(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_memory_block_action action_type;
            return lcos::eager_future<action_type, naming::id_type>(targetgid, count, act);
        }

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        template <typename T, typename Config>
        static naming::id_type
        create(naming::id_type const& targetgid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_async(targetgid, count, act).get();
        }

        /// Exposed functionality: get returns either the local memory pointers
        /// or a copy of the remote data.

        static lcos::promise<components::memory_block_data> get_async(
            naming::id_type const& targetgid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::get_action action_type;
            typedef components::memory_block_data data_type;
            return lcos::eager_future<action_type, data_type>(targetgid);
        }

        static components::memory_block_data get(
            naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return get_async(targetgid).get();
        }

        static lcos::promise<components::memory_block_data> get_async(
            naming::id_type const& targetgid,
            components::memory_block_data const& cfg)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::get_config_action action_type;
            typedef components::memory_block_data data_type;
            return lcos::eager_future<action_type, data_type>(targetgid, cfg);
        }

        static components::memory_block_data get(
            naming::id_type const& targetgid,
            components::memory_block_data const& cfg)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return get_async(targetgid, cfg).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::promise<components::memory_block_data> checkout_async(
            naming::id_type const& targetgid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::checkout_action action_type;
            typedef components::memory_block_data data_type;
            return lcos::eager_future<action_type, data_type>(targetgid);
        }

        static components::memory_block_data checkout(
            naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return checkout_async(targetgid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::promise<naming::id_type, naming::gid_type>
        clone_async(naming::id_type const& targetgid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::clone_action action_type;
            typedef naming::gid_type data_type;
            return lcos::eager_future<action_type, naming::id_type>(targetgid);
        }

        static naming::id_type clone(naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return clone_async(targetgid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        static lcos::promise<void>
        checkin_async(naming::id_type const& targetgid,
            components::access_memory_block<T> const& data)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::detail::memory_block::checkin_action action_type;
            return lcos::eager_future<action_type, void>(targetgid, data.get_memory_block());
        }

        template <typename T>
        static void checkin(naming::id_type const& targetgid,
            components::access_memory_block<T> const& data)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            checkin_async(targetgid, data).get();
        }
    };
}}}

#endif
