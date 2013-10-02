//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/async.hpp>
#include <hpx/util/remove_reference.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum_params.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a runtime_support class is the client side representation of a
    // \a server#runtime_support component
    struct HPX_EXPORT runtime_support
    {
        ///////////////////////////////////////////////////////////////////////
        /// \brief  The function \a get_factory_properties is used to
        ///         determine, whether instances of the derived component can
        ///         be created in blocks (i.e. more than one instance at once).
        ///         This function is used by the \a distributing_factory to
        ///         determine a correct allocation strategy
        static lcos::future<int> get_factory_properties_async(
            naming::id_type const& targetgid, components::component_type type);

        static int get_factory_properties(naming::id_type const& targetgid,
            components::component_type type);

        ///////////////////////////////////////////////////////////////////////
        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. This is a non-blocking call. The caller needs
        /// to call \a future#get on the result of this function
        /// to obtain the global id of the newly created object.
        template <typename Component>
        static lcos::future<naming::id_type>
        create_component_async(naming::id_type const& gid)
        {
            if (!naming::is_locality(gid))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "stubs::runtime_support::create_component_async",
                    "The id passed as the first argument is not representing"
                        " a locality");
                return lcos::make_ready_future(naming::invalid_id);
            }

            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef typename server::create_component_action0<Component>
                action_type;
            return hpx::async<action_type>(gid);
        }

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        template <typename Component>
        static naming::id_type create_component(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return create_component_async<Component>(gid).get();
        }

#define HPX_RUNTIME_SUPPORT_STUB_REMOVE_REFERENCE(Z, N, D)                    \
        typename util::remove_reference<BOOST_PP_CAT(D, N)>::type             \
/**/
#define HPX_RUNTIME_SUPPORT_STUB_CREATE(Z, N, D)                              \
        template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename Arg)>  \
        static lcos::future<naming::id_type>                                  \
        create_component_async(naming::id_type const& gid,                    \
            HPX_ENUM_FWD_ARGS(N, Arg, arg))                                   \
        {                                                                     \
            if (!naming::is_locality(gid))                                    \
            {                                                                 \
                HPX_THROW_EXCEPTION(bad_parameter,                            \
                    "stubs::runtime_support::create_component_async",         \
                    "The id passed as the first argument is not representing" \
                        " a locality");                                       \
                return lcos::make_ready_future(naming::invalid_id);           \
            }                                                                 \
                                                                              \
            typedef typename                                                  \
                server::BOOST_PP_CAT(create_component_action, N)<             \
                    Component, BOOST_PP_ENUM(N,                               \
                        HPX_RUNTIME_SUPPORT_STUB_REMOVE_REFERENCE, Arg)>      \
                action_type;                                                  \
            return hpx::async<action_type>(gid,                               \
                HPX_ENUM_FORWARD_ARGS(N , Arg, arg));                         \
        }                                                                     \
                                                                              \
        template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename Arg)>  \
        static naming::id_type create_component(                              \
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))       \
        {                                                                     \
            return create_component_async<Component>                          \
                (gid, HPX_ENUM_FORWARD_ARGS(N , Arg, arg)).get();             \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_RUNTIME_SUPPORT_STUB_CREATE
          , _
        )

#undef HPX_RUNTIME_SUPPORT_STUB_CREATE
#undef HPX_RUNTIME_SUPPORT_STUB_REMOVE_REFERENCE

        ///////////////////////////////////////////////////////////////////////
        // copy construct a component 
        template <typename Component>
        static lcos::future<naming::id_type>
        copy_create_component_async(naming::id_type const& gid,
            boost::shared_ptr<Component> const& p, bool local_op)
        {
            if (!naming::is_locality(gid))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "stubs::runtime_support::copy_create_component_async",
                    "The id passed as the first argument is not representing"
                        " a locality");
                return lcos::make_ready_future(naming::invalid_id);
            }

            typedef typename server::copy_create_component_action<Component>
                action_type;
            return hpx::async<action_type>(gid, p, local_op);
        }

        template <typename Component>
        static naming::id_type copy_create_component(naming::id_type const& gid,
            boost::shared_ptr<Component> const& p, bool local_op)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return copy_create_component_async<Component>(gid, p, local_op).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<std::vector<naming::id_type> >
        bulk_create_components_async(
            naming::id_type const& gid, components::component_type type,
            std::size_t count = 1);

        /// Create a new component \a type using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        static std::vector<naming::id_type> bulk_create_components(
            naming::id_type const& gid, components::component_type type,
            std::size_t count = 1);

        ///////////////////////////////////////////////////////////////////////
        /// Create a new memory block using the runtime_support with the
        /// given \a targetgid. This is a non-blocking call. The caller needs
        /// to call \a future#get on the result of this function
        /// to obtain the global id of the newly created object.
        template <typename T, typename Config>
        static lcos::future<naming::id_type>
        create_memory_block_async(
            naming::id_type const& id, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act);

        /// Create a new memory block using the runtime_support with the
        /// given \a targetgid. Block for the creation to finish.
        template <typename T, typename Config>
        static naming::id_type create_memory_block(
            naming::id_type const& id, std::size_t count,
            hpx::actions::manage_object_action<T, Config> const& act)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return create_memory_block_async(id, count, act).get();
        }

        static lcos::future<bool>
        load_components_async(naming::id_type const& gid);
        static bool load_components(naming::id_type const& gid);

        static lcos::future<void>
        call_startup_functions_async(naming::id_type const& gid,
            bool pre_startup);
        static void call_startup_functions(naming::id_type const& gid,
            bool pre_startup);

        static lcos::future<void>
        call_shutdown_functions_async(naming::id_type const& gid,
            bool pre_shutdown);

        static void call_shutdown_functions(naming::id_type const& gid,
            bool pre_shutdown);
        static void free_component_sync(components::component_type type,
            naming::gid_type const& gid, boost::uint64_t count);

        static void free_component_sync(components::component_type type,
            naming::gid_type const& gid, naming::gid_type const& count);

        /// \brief Shutdown the given runtime system
        static lcos::future<void>
        shutdown_async(naming::id_type const& targetgid,
            double timeout = -1);
        static void shutdown(naming::id_type const& targetgid,
            double timeout = - 1);

        /// \brief Shutdown the runtime systems of all localities
        static void
        shutdown_all(naming::id_type const& targetgid, double timeout = -1);

        static void shutdown_all(double timeout = -1);

        ///////////////////////////////////////////////////////////////////////
        /// \brief Retrieve configuration information
        /// \brief Terminate the given runtime system
        static lcos::future<void>
        terminate_async(naming::id_type const& targetgid);

        static void terminate(naming::id_type const& targetgid);

        /// \brief Terminate the runtime systems of all localities
        static void
        terminate_all(naming::id_type const& targetgid);

        static void terminate_all();

        ///////////////////////////////////////////////////////////////////////
        static void
        update_agas_cache_entry(naming::id_type const& targetgid,
            naming::gid_type const& gid, naming::address const& g,
            boost::uint64_t count, boost::uint64_t offset);

        ///////////////////////////////////////////////////////////////////////
        static void
        garbage_collect_non_blocking(naming::id_type const& targetgid);

        static lcos::future<void>
        garbage_collect_async(naming::id_type const& targetgid);

        static void
        garbage_collect(naming::id_type const& targetgid);

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<naming::id_type>
        create_performance_counter_async(naming::id_type targetgid,
            performance_counters::counter_info const& info);
        static naming::id_type
        create_performance_counter(naming::id_type targetgid,
            performance_counters::counter_info const& info,
            error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        /// \brief Retrieve configuration information
        static lcos::future<util::section> get_config_async(
            naming::id_type const& targetgid);
        static void get_config(naming::id_type const& targetgid, util::section& ini);

        ///////////////////////////////////////////////////////////////////////
        /// \brief Retrieve instance count for given component type
        static lcos::future<long> get_instance_count_async(
            naming::id_type const& targetgid, components::component_type type);
        static long get_instance_count(naming::id_type const& targetgid,
            components::component_type type);

        ///////////////////////////////////////////////////////////////////////
        static void
        call_shutdown_functions_async(naming::id_type const& gid,
            naming::locality const& l);
    };
}}}

#endif
