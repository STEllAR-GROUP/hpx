//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM

#include <boost/bind.hpp>

#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/lcos/eager_future.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a runtime_support class is the client side representation of a 
    // \a server#runtime_support component
    struct runtime_support
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief  The function \a get_factory_properties is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        static lcos::future_value<int> get_factory_properties_async(
            naming::id_type const& targetgid, components::component_type type) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef 
                server::runtime_support::factory_properties_action 
            action_type;
            return lcos::eager_future<action_type>(targetgid, type);
        }

        static int get_factory_properties(naming::id_type const& targetgid, 
            components::component_type type) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return get_factory_properties_async(targetgid, type).get();
        }

        /// Create a new component \a type using the runtime_support with the 
        /// given \a targetgid. This is a non-blocking call. The caller needs 
        /// to call \a future_value#get on the result of this function 
        /// to obtain the global id of the newly created object.
        static lcos::future_value<naming::id_type> create_component_async(
            naming::id_type const& targetgid, components::component_type type, 
            std::size_t count = 1) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_component_action action_type;
            return lcos::eager_future<action_type>(targetgid, type, count);
        }

        /// Create a new component \a type using the runtime_support with the 
        /// given \a targetgid. Block for the creation to finish.
        static naming::id_type create_component(
            naming::id_type const& targetgid, components::component_type type, 
            std::size_t count = 1) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return create_component_async(targetgid, type, count).get();
        }

        /// Destroy an existing component
        static void free_component(components::component_type type, 
            naming::id_type const& gid) 
        {
            typedef server::runtime_support::free_component_action action_type;

            // Determine whether the gid of the component to delete is local or 
            // remote
            naming::address addr;
            applier::applier& appl = hpx::applier::get_applier();
            if (appl.address_is_local(gid, addr)) {
                // apply locally
                applier::detail::apply_helper2<
                    action_type, components::component_type, naming::id_type
                >::call(appl.get_runtime_support_gid().get_lsb(), type, gid);
            }
            else {
                // apply remotely
                // zero address will be interpreted as a reference to the 
                // remote runtime support object
                addr.address_ = 0;
                hpx::applier::apply<action_type>(addr, naming::invalid_id, type, gid);
            }
        }

        static void free_component_sync(components::component_type type, 
            naming::id_type const& gid) 
        {
            typedef server::runtime_support::free_component_action action_type;

            // Determine whether the gid of the component to delete is local or remote
            naming::address addr;
            applier::applier& appl = hpx::applier::get_applier();
            if (appl.address_is_local(gid, addr)) {
                // apply locally
                applier::detail::apply_helper2<
                    action_type, components::component_type, naming::id_type
                >::call(appl.get_runtime_support_gid().get_lsb(), type, gid);
            }
            else {
                // apply remotely
                lcos::eager_future<action_type, void>(
                    naming::invalid_id, type, gid).get();
            }
        }

        /// \brief Shutdown the given runtime system
        static void 
        shutdown(naming::id_type const& targetgid)
        {
            hpx::applier::apply<server::runtime_support::shutdown_action>(targetgid);
        }

        /// \brief Shutdown the runtime systems of all localities
        static void 
        shutdown_all(naming::id_type const& targetgid)
        {
            hpx::applier::apply<server::runtime_support::shutdown_all_action>(targetgid);
        }

        static void shutdown_all()
        {
            hpx::applier::apply<server::runtime_support::shutdown_all_action>(
                hpx::applier::get_applier().get_runtime_support_gid());
        }
    };

}}}

#endif
