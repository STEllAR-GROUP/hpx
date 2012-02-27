//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM

#include <boost/bind.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/util/ini.hpp>

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
        static lcos::promise<int> get_factory_properties_async(
            naming::id_type const& targetgid, components::component_type type)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
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

        ///////////////////////////////////////////////////////////////////////
        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. This is a non-blocking call. The caller needs
        /// to call \a promise#get on the result of this function
        /// to obtain the global id of the newly created object.
        static lcos::promise<naming::id_type, naming::gid_type>
        create_component_async(
            naming::gid_type const& gid, components::component_type type,
            std::size_t count = 1)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_component_action action_type;
            return lcos::eager_future<action_type, naming::id_type>(gid, type, count);
        }

        static lcos::promise<naming::id_type, naming::gid_type>
        create_component_async(
            naming::id_type const& gid, components::component_type type,
            std::size_t count = 1)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_component_action action_type;
            return lcos::eager_future<action_type, naming::id_type>(
                gid.get_gid(), type, count);
        }

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        static naming::id_type create_component(
            naming::gid_type const& gid, components::component_type type,
            std::size_t count = 1)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_component_async(gid, type, count).get();
        }

        static naming::id_type create_component(
            naming::id_type const& gid, components::component_type type,
            std::size_t count = 1)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_component_async(gid.get_gid(), type, count).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. This is a non-blocking call. The caller needs
        /// to call \a promise#get on the result of this function
        /// to obtain the global id of the newly created object. Pass one
        /// generic argument to the constructor.
        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_component_async(
            naming::gid_type const& gid, components::component_type type,
            Arg0 const& arg0)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_one_component_action
                action_type;
            return lcos::eager_future<action_type, naming::id_type>(
                gid, type, components::constructor_argument(arg0));
        }

        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_component_async(
            naming::id_type const& gid, components::component_type type,
            Arg0 const& arg0)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_one_component_action
                action_type;
            return lcos::eager_future<action_type, naming::id_type>(
                gid.get_gid(), type, components::constructor_argument(arg0));
        }

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish. Pass one
        /// generic argument to the constructor.
        template <typename Arg0>
        static naming::id_type create_one_component(
            naming::id_type const& targetgid, components::component_type type,
            Arg0 const& arg0)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_one_component_async(targetgid, type, arg0).get();
        }

        template <typename Arg0>
        static naming::id_type create_one_component(
            naming::gid_type const& gid, components::component_type type,
            Arg0 const& arg0)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_one_component_async(gid, type, arg0).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new memory block using the runtime_support with the
        /// given \a targetgid. This is a non-blocking call. The caller needs
        /// to call \a promise#get on the result of this function
        /// to obtain the global id of the newly created object.
//         template <typename T>
//         static lcos::promise<naming::id_type, naming::gid_type>
//         create_memory_block_async(
//             naming::gid_type const& gid, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // Create an eager_future, execute the required action,
//             // we simply return the initialized promise, the caller needs
//             // to call get() on the return value to obtain the result
//             typedef server::runtime_support::create_memory_block_action action_type;
//             return lcos::eager_future<action_type, naming::id_type>(gid, count, act);
//         }
//
//         template <typename T>
//         static lcos::promise<naming::id_type, naming::gid_type>
//         create_memory_block_async(
//             naming::id_type const& id, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // Create an eager_future, execute the required action,
//             // we simply return the initialized promise, the caller needs
//             // to call get() on the return value to obtain the result
//             typedef server::runtime_support::create_memory_block_action action_type;
//             return lcos::eager_future<action_type, naming::id_type>(
//                 id.get_gid(), count, act);
//         }
//
//         /// Create a new memory block using the runtime_support with the
//         /// given \a targetgid. Block for the creation to finish.
//         template <typename T>
//         static naming::id_type create_memory_block(
//             naming::gid_type const& gid, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // The following get yields control while the action above
//             // is executed and the result is returned to the eager_future
//             return create_memory_block_async(gid, count, act).get();
//         }
//
//         template <typename T>
//         static naming::id_type create_memory_block(
//             naming::id_type const& id, std::size_t count,
//             hpx::actions::manage_object_action<T> const& act)
//         {
//             // The following get yields control while the action above
//             // is executed and the result is returned to the eager_future
//             return create_memory_block_async(id.get_gid(), count, act).get();
//         }

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Config>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_memory_block_async(
            naming::gid_type const& gid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_memory_block_action
                action_type;
            return lcos::eager_future<action_type, naming::id_type>(
                gid, count, act);
        }

        template <typename T, typename Config>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_memory_block_async(
            naming::id_type const& id, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::create_memory_block_action
                action_type;
            return lcos::eager_future<action_type, naming::id_type>(
                id.get_gid(), count, act);
        }

        /// Create a new memory block using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        template <typename T, typename Config>
        static naming::id_type create_memory_block(
            naming::gid_type const& gid, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_memory_block_async(gid, count, act).get();
        }

        template <typename T, typename Config>
        static naming::id_type create_memory_block(
            naming::id_type const& id, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            return create_memory_block_async(id.get_gid(), count, act).get();
        }

        static lcos::promise<bool>
        load_components_async(naming::id_type const& gid)
        {
            typedef server::runtime_support::load_components_action action_type;
            return lcos::eager_future<action_type>(gid.get_gid());
        }

        static bool load_components(naming::id_type const& gid)
        {
            return load_components_async(gid).get();
        }

        static lcos::promise<void>
        call_startup_functions_async(naming::id_type const& gid, bool pre_startup)
        {
            typedef server::runtime_support::call_startup_functions_action action_type;
            return lcos::eager_future<action_type, void>(gid.get_gid(), pre_startup);
        }

        static void call_startup_functions(naming::id_type const& gid, bool pre_startup)
        {
            call_startup_functions_async(gid, pre_startup).get();
        }

        static lcos::promise<void>
        call_shutdown_functions_async(naming::id_type const& gid, bool pre_shutdown)
        {
            typedef server::runtime_support::call_shutdown_functions_action action_type;
            return lcos::eager_future<action_type, void>(gid.get_gid(), pre_shutdown);
        }

        static void call_shutdown_functions(naming::id_type const& gid, bool pre_shutdown)
        {
            call_shutdown_functions_async(gid, pre_shutdown).get();
        }

        static void free_component_sync(components::component_type type,
            naming::gid_type const& gid, boost::uint64_t count)
        {
            free_component_sync(type, gid, naming::gid_type(0, count));
        }

        static void free_component_sync(components::component_type type,
            naming::gid_type const& gid, naming::gid_type const& count)
        {
            typedef server::runtime_support::free_component_action action_type;

            // Determine whether the gid of the component to delete is local or
            // remote
            //naming::resolver_client& agas = appl.get_agas_client();
            if (/*agas.is_bootstrap() || */agas::is_local_address(gid)) {
                // apply locally
                applier::detail::apply_helper3<action_type>::call(
                    applier::get_applier().get_runtime_support_raw_gid().get_lsb(),
                    threads::thread_priority_default, type, gid, count);
            }
            else {
                // apply remotely
                // FIXME: Resolve the locality instead of deducing it from
                // the target GID, otherwise this will break once we start
                // moving objects.
                naming::gid_type prefix = naming::get_locality_from_gid(gid);
                lcos::eager_future<action_type, void>(prefix, type, gid, count).get();
            }
        }

        /// \brief Shutdown the given runtime system
        static lcos::promise<void>
        shutdown_async(naming::id_type const& targetgid, double timeout = -1)
        {
            // Create a promise directly and execute the required action.
            // This action has implemented special response handling as the
            // back-parcel is sent explicitly (and synchronously).
            typedef server::runtime_support::shutdown_action action_type;

            lcos::promise<void> value;
            hpx::applier::apply<action_type>(targetgid, timeout, value.get_gid());
            return value;
        }

        static void shutdown(naming::id_type const& targetgid, double timeout = - 1)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            shutdown_async(targetgid, timeout).get();
        }

        /// \brief Shutdown the runtime systems of all localities
        static void
        shutdown_all(naming::id_type const& targetgid, double timeout = -1)
        {
            hpx::applier::apply<server::runtime_support::shutdown_all_action>(
                targetgid, timeout);
        }

        static void shutdown_all(double timeout = -1)
        {
            hpx::applier::apply<server::runtime_support::shutdown_all_action>(
                hpx::naming::id_type(
                    hpx::applier::get_applier().get_runtime_support_raw_gid(),
                    hpx::naming::id_type::unmanaged), timeout);
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Retrieve configuration information
        /// \brief Terminate the given runtime system
        static lcos::promise<void>
        terminate_async(naming::id_type const& targetgid)
        {
            // Create a promise directly and execute the required action.
            // This action has implemented special response handling as the
            // back-parcel is sent explicitly (and synchronously).
            typedef server::runtime_support::terminate_action action_type;

            lcos::promise<void> value;
            hpx::applier::apply<action_type>(targetgid, value.get_gid());
            return value;
        }

        static void terminate(naming::id_type const& targetgid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            terminate_async(targetgid).get();
        }

        /// \brief Terminate the runtime systems of all localities
        static void
        terminate_all(naming::id_type const& targetgid)
        {
            hpx::applier::apply<server::runtime_support::terminate_all_action>(
                targetgid);
        }

        static void terminate_all()
        {
            hpx::applier::apply<server::runtime_support::terminate_all_action>(
                hpx::naming::id_type(
                    hpx::applier::get_applier().get_runtime_support_raw_gid(),
                    hpx::naming::id_type::unmanaged));
        }

        static void
        update_agas_cache(naming::id_type const& targetgid,
            naming::gid_type const& gid, agas::gva const& g)
        {
            typedef server::runtime_support::update_agas_cache_action
                action_type;
            hpx::applier::apply<action_type>(targetgid, gid, g);
        }

        ///////////////////////////////////////////////////////////////////////
        static void
        garbage_collect_non_blocking(naming::id_type const& targetgid)
        {
            typedef server::runtime_support::garbage_collect_action
                action_type;
            hpx::applier::apply<action_type>(targetgid);
        }

        static lcos::promise<void>
        garbage_collect_async(naming::id_type const& targetgid)
        {
            typedef server::runtime_support::garbage_collect_action
                action_type;
            return lcos::eager_future<action_type, void>(targetgid);
        }

        static void
        garbage_collect(naming::id_type const& targetgid)
        {
            typedef server::runtime_support::garbage_collect_action
                action_type;
            lcos::eager_future<action_type, void>(targetgid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::promise<naming::gid_type>
        create_performance_counter_async(naming::id_type targetgid,
            performance_counters::counter_info const& info)
        {
            typedef server::runtime_support::create_performance_counter_action
                action_type;
            return lcos::eager_future<action_type, naming::gid_type>(targetgid, info);
        }

        static naming::gid_type
        create_performance_counter(naming::id_type targetgid,
            performance_counters::counter_info const& info)
        {
            typedef server::runtime_support::create_performance_counter_action
                action_type;
            return lcos::eager_future<action_type, naming::gid_type>(targetgid, info).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Retrieve configuration information
        static lcos::promise<util::section> get_config_async(
            naming::id_type const& targetgid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::runtime_support::get_config_action action_type;
            return lcos::eager_future<action_type>(targetgid);
        }

        static void get_config(naming::id_type const& targetgid, util::section& ini)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the eager_future
            ini = get_config_async(targetgid).get();
        }
    };

}}}

#endif
