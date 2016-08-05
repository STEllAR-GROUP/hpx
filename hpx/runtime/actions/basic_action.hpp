//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/actions/basic_action.hpp

#ifndef HPX_RUNTIME_ACTIONS_BASIC_ACTION_HPP
#define HPX_RUNTIME_ACTIONS_BASIC_ACTION_HPP

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/detail/invocation_count_registry.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/void_guard.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action, typename F, typename ...Ts>
        struct continuation_thread_function
        {
            HPX_MOVABLE_ONLY(continuation_thread_function);

        public:
            explicit continuation_thread_function(
                std::unique_ptr<continuation> cont,
                naming::address::address_type lva, F&& f, Ts&&... vs)
              : cont_(std::move(cont))
              , lva_(lva)
              , f_(std::forward<F>(f), std::forward<Ts>(vs)...)
            {}

            continuation_thread_function(continuation_thread_function && other)
              : cont_(std::move(other.cont_)), lva_(std::move(other.lva_))
              , f_(std::move(other.f_))
            {}

            HPX_FORCEINLINE threads::thread_state_enum
            operator()(threads::thread_state_ex_enum)
            {
                LTM_(debug) << "Executing " << Action::get_action_name(lva_)
                    << " with continuation(" << cont_->get_id() << ")";

                typedef typename Action::local_result_type local_result_type;

                actions::trigger<local_result_type>(std::move(cont_), f_);
                return threads::terminated;
            }

        private:
            std::unique_ptr<continuation> cont_;
            naming::address::address_type lva_;
            util::detail::deferred<F(Ts&&...)> f_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct is_non_const_reference
          : std::integral_constant<bool,
                std::is_lvalue_reference<T>::value &&
               !std::is_const<typename std::remove_reference<T>::type>::value
            >
        {};
    }

    template <typename Component, typename R, typename ...Args, typename Derived>
    struct basic_action<Component, R(Args...), Derived>
    {
        // Flag the use of raw pointer types as action arguments
        static_assert(
            !util::detail::any_of<std::is_pointer<Args>...>::value,
            "Using raw pointers as arguments for actions is not supported.");

        // Flag the use of array types as action arguments
        static_assert(
            !util::detail::any_of<
                std::is_array<typename std::remove_reference<Args>::type>...
            >::value,
            "Using arrays as arguments for actions is not supported.");

        // Flag the use of non-const reference types as action arguments
        static_assert(
            !util::detail::any_of<
                detail::is_non_const_reference<Args>...
            >::value,
            "Using non-const references as arguments for actions is not supported.");

        typedef Component component_type;
        typedef Derived derived_type;

        // result_type represents the type returned when invoking operator()
        typedef typename traits::promise_local_result<R>::type result_type;

        // The remote_result_type is the remote type for the type_continuation
        typedef typename traits::action_remote_result<R>::type remote_result_type;

        // The local_result_type is the local type for the type_continuation
        typedef
            typename traits::promise_local_result<
                remote_result_type
            >::type local_result_type;

        static const std::size_t arity = sizeof...(Args);
        typedef util::tuple<typename std::decay<Args>::type...> arguments_type;

        typedef void action_tag;

        ///////////////////////////////////////////////////////////////////////
        static std::string get_action_name(naming::address::address_type /*lva*/)
        {
            std::stringstream name;
            name << "action(" << detail::get_action_name<Derived>() << ")";
            return name.str();
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type /*lva*/, Ts&&... /*vs*/);

    protected:
        struct invoker
        {
            typedef
                typename std::conditional<
                    std::is_void<R>::value, util::unused_type, R
                >::type
                result_type;
            template <typename ...Ts>
            HPX_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                return invoke(
                    typename std::is_void<R>::type(), lva, std::forward<Ts>(vs)...);
            }

            template <typename ...Ts>
            HPX_FORCEINLINE result_type invoke(std::true_type,
                naming::address::address_type lva, Ts&&... vs) const
            {
                Derived::invoke(lva, std::forward<Ts>(vs)...);
                return util::unused;
            }

            template <typename ...Ts>
            HPX_FORCEINLINE result_type invoke(std::false_type,
                naming::address::address_type lva, Ts&&... vs) const
            {
                return Derived::invoke(lva, std::forward<Ts>(vs)...);
            }
        };

        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            template <typename ...Ts>
            HPX_FORCEINLINE threads::thread_state_enum
            operator()(naming::address::address_type lva, Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing "
                        << Derived::get_action_name(lva) << ".";

                    // call the function, ignoring the return value
                    Derived::invoke(lva, std::forward<Ts>(vs)...);
                }
                catch (hpx::thread_interrupted const&) { //-V565
                    /* swallow this exception */
                }
                catch (std::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing "
                        << Derived::get_action_name(lva) << ": " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing "
                        << Derived::get_action_name(lva);

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();
                return threads::terminated;
            }
        };

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case no continuation has been supplied.
        template <typename ...Ts>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Ts&&... vs)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, std::forward<Ts>(vs)...));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename ...Ts>
        static threads::thread_function_type
        construct_thread_function(std::unique_ptr<continuation> cont,
            naming::address::address_type lva, Ts&&... vs)
        {
            typedef detail::continuation_thread_function<
                Derived, invoker, naming::address::address_type&, Ts&&...
            > thread_function;

            return traits::action_decorate_function<Derived>::call(lva,
                thread_function(std::move(cont), lva, invoker(),
                    lva, std::forward<Ts>(vs)...));
        }

        // direct execution
        template <typename ...Ts>
        static HPX_FORCEINLINE
        typename invoker::result_type
        execute_function(naming::address::address_type lva, Ts&&... vs)
        {
            LTM_(debug)
                << "basic_action::execute_function"
                << Derived::get_action_name(lva);

            return invoker()(lva, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        typedef traits::is_future<result_type> is_future_pred;

        struct sync_invoke
        {
            template <typename IdOrPolicy, typename ...Ts>
            HPX_FORCEINLINE static result_type call(
                std::false_type, launch policy,
                IdOrPolicy const& id_or_policy, error_code& ec, Ts&&... vs)
            {
                return hpx::async<basic_action>(policy, id_or_policy,
                    std::forward<Ts>(vs)...).get(ec);
            }

            template <typename IdOrPolicy, typename ...Ts>
            HPX_FORCEINLINE static result_type call(
                std::true_type, launch policy,
                IdOrPolicy const& id_or_policy, error_code& /*ec*/, Ts&&... vs)
            {
                return hpx::async<basic_action>(policy, id_or_policy,
                    std::forward<Ts>(vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ...Ts>
        HPX_FORCEINLINE result_type operator()(
            launch policy, naming::id_type const& id,
            error_code& ec, Ts&&... vs) const
        {
            return util::void_guard<result_type>(),
                sync_invoke::call(
                    is_future_pred(), policy, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        HPX_FORCEINLINE result_type operator()(
            naming::id_type const& id, error_code& ec, Ts&&... vs) const
        {
            return (*this)(launch::all, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        HPX_FORCEINLINE result_type operator()(
            launch policy, naming::id_type const& id,
            Ts&&... vs) const
        {
            return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        HPX_FORCEINLINE result_type operator()(
            naming::id_type const& id, Ts&&... vs) const
        {
            return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename DistPolicy, typename ...Ts>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type
        >::type
        operator()(launch policy,
            DistPolicy const& dist_policy, error_code& ec, Ts&&... vs) const
        {
            return util::void_guard<result_type>(),
                sync_invoke::call(
                    is_future_pred(), policy, dist_policy, ec,
                    std::forward<Ts>(vs)...
                );
        }

        template <typename DistPolicy, typename ...Ts>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type
        >::type
        operator()(DistPolicy const& dist_policy, error_code& ec,
            Ts&&... vs) const
        {
            return (*this)(launch::all, dist_policy, ec,
                std::forward<Ts>(vs)...);
        }

        template <typename DistPolicy, typename ...Ts>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type
        >::type
        operator()(launch policy,
            DistPolicy const& dist_policy, Ts&&... vs) const
        {
            return (*this)(launch::all, dist_policy, throws,
                std::forward<Ts>(vs)...);
        }

        template <typename DistPolicy, typename ...Ts>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type
        >::type
        operator()(DistPolicy const& dist_policy, Ts&&... vs) const
        {
            return (*this)(launch::all, dist_policy, throws,
                std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        /// retrieve component type
        static int get_component_type()
        {
            return static_cast<int>(components::get_component_type<Component>());
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::plain_action;
        }

        /// Extract the current invocation count for this action
        static std::int64_t get_invocation_count(bool reset)
        {
            return util::get_and_reset_value(invocation_count_, reset);
        }

    private:
        static boost::atomic<std::int64_t> invocation_count_;

    protected:
        static void increment_invocation_count()
        {
            ++invocation_count_;
        }
    };

    template <typename Component, typename R, typename ...Args, typename Derived>
    boost::atomic<std::int64_t>
        basic_action<Component, R(Args...), Derived>::invocation_count_(0);

    namespace detail
    {
        template <typename Action>
        void register_local_action_invocation_count(
            invocation_count_registry& registry)
        {
            registry.register_class(
                hpx::actions::detail::get_action_name<Action>(),
                &Action::get_invocation_count
            );
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // simple type allowing to distinguish whether an action is the most
        // derived one
        struct this_type {};

        template <typename Action, typename Derived>
        struct action_type
        {
            typedef Derived type;
        };

        template <typename Action>
        struct action_type<Action, this_type>
        {
            typedef Action type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Signature, typename TF, TF F, typename Derived>
    class basic_action_impl;

    ///////////////////////////////////////////////////////////////////////////
    template <typename TF, TF F, typename Derived = detail::this_type>
    struct action
      : basic_action_impl<TF, TF, F,
            typename detail::action_type<
                action<TF, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;

        typedef std::false_type direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename TF, TF F, typename Derived = detail::this_type>
    struct direct_action
      : basic_action_impl<TF, TF, F,
            typename detail::action_type<
                direct_action<TF, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;

        typedef std::true_type direct_execution;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Base template allowing to generate a concrete action type from a function
    // pointer. It is instantiated only if the supplied pointer is not a
    // supported function pointer.
    template <typename TF, TF F, typename Derived = detail::this_type,
        typename Direct = std::false_type>
    struct make_action;

    template <typename TF, TF F, typename Derived>
    struct make_action<TF, F, Derived, std::false_type>
      : action<TF, F, Derived>
    {
        typedef action<TF, F, Derived> type;
    };

    template <typename TF, TF F, typename Derived>
    struct make_action<TF, F, Derived, std::true_type>
      : direct_action<TF, F, Derived>
    {
        typedef direct_action<TF, F, Derived> type;
    };

    template <typename TF, TF F, typename Derived = detail::this_type>
    struct make_direct_action
      : make_action<TF, F, Derived, std::true_type>
    {};

    // Macros usable to refer to an action given the function to expose
    #define HPX_MAKE_ACTION(func)                                             \
        hpx::actions::make_action<decltype(&func), &func>        /**/         \
    /**/
    #define HPX_MAKE_DIRECT_ACTION(func)                                      \
        hpx::actions::make_direct_action<decltype(&func), &func> /**/         \
    /**/

    enum preassigned_action_id
    {
        register_worker_action_id = 0,
        notify_worker_action_id,
        allocate_action_id,
        base_connect_action_id,
        base_disconnect_action_id,
        base_set_event_action_id,
        base_set_exception_action_id,
        broadcast_call_shutdown_functions_action_id,
        broadcast_call_startup_functions_action_id,
        broadcast_symbol_namespace_service_action_id,
        bulk_create_components_action_id,
        call_shutdown_functions_action_id,
        call_startup_functions_action_id,
        component_namespace_bulk_service_action_id,
        component_namespace_service_action_id,
        console_error_sink_action_id,
        console_logging_action_id,
        console_print_action_id,
        create_memory_block_action_id,
        create_performance_counter_action_id,
        dijkstra_termination_action_id,
        free_component_action_id,
        garbage_collect_action_id,
        get_config_action_id,
        get_instance_count_action_id,
        hpx_get_locality_name_action_id,
        hpx_lcos_server_barrier_create_component_action_id,
        hpx_lcos_server_latch_create_component_action_id,
        hpx_lcos_server_latch_wait_action_id,
        list_component_type_action_id,
        list_symbolic_name_action_id,
        load128_action_id,
        load16_action_id,
        load32_action_id,
        load64_action_id,
        load8_action_id,
        load_components_action_id,
        locality_namespace_bulk_service_action_id,
        locality_namespace_service_action_id,
        memory_block_checkin_action_id,
        memory_block_checkout_action_id,
        memory_block_clone_action_id,
        memory_block_get_action_id,
        memory_block_get_config_action_id,
        output_stream_write_async_action_id,
        output_stream_write_sync_action_id,
        performance_counter_get_counter_info_action_id,
        performance_counter_get_counter_value_action_id,
        performance_counter_get_counter_values_array_action_id,
        performance_counter_set_counter_value_action_id,
        performance_counter_reset_counter_value_action_id,
        performance_counter_start_action_id,
        performance_counter_stop_action_id,
        primary_namespace_bulk_service_action_id,
        primary_namespace_service_action_id,
        primary_namespace_route_action_id,
        remove_from_connection_cache_action_id,
        set_value_action_agas_bool_response_type_id,
        set_value_action_agas_id_type_response_type_id,
        shutdown_action_id,
        shutdown_all_action_id,
        store128_action_id,
        store16_action_id,
        store32_action_id,
        store64_action_id,
        store8_action_id,
        symbol_namespace_bulk_service_action_id,
        symbol_namespace_service_action_id,
        terminate_action_id,
        terminate_all_action_id,
        update_agas_cache_entry_action_id,
        register_worker_security_action_id,
        notify_worker_security_action_id,

        base_lco_with_value_gid_get,
        base_lco_with_value_gid_set,
        base_lco_with_value_vector_gid_get,
        base_lco_with_value_vector_gid_set,
        base_lco_with_value_id_get,
        base_lco_with_value_id_set,
        base_lco_with_value_vector_id_get,
        base_lco_with_value_vector_id_set,
        base_lco_with_value_unused_get,
        base_lco_with_value_unused_set,
        base_lco_with_value_float_get,
        base_lco_with_value_float_set,
        base_lco_with_value_double_get,
        base_lco_with_value_double_set,
        base_lco_with_value_int8_get,
        base_lco_with_value_int8_set,
        base_lco_with_value_uint8_get,
        base_lco_with_value_uint8_set,
        base_lco_with_value_int16_get,
        base_lco_with_value_int16_set,
        base_lco_with_value_uint16_get,
        base_lco_with_value_uint16_set,
        base_lco_with_value_int32_get,
        base_lco_with_value_int32_set,
        base_lco_with_value_uint32_get,
        base_lco_with_value_uint32_set,
        base_lco_with_value_int64_get,
        base_lco_with_value_int64_set,
        base_lco_with_value_uint64_get,
        base_lco_with_value_uint64_set,
        base_lco_with_value_uint128_get,
        base_lco_with_value_uint128_set,
        base_lco_with_value_bool_get,
        base_lco_with_value_bool_set,
        base_lco_with_value_hpx_section_get,
        base_lco_with_value_hpx_section_set,
        base_lco_with_value_hpx_counter_info_get,
        base_lco_with_value_hpx_counter_info_set,
        base_lco_with_value_hpx_counter_value_get,
        base_lco_with_value_hpx_counter_value_set,
        base_lco_with_value_hpx_agas_response_get,
        base_lco_with_value_hpx_agas_response_set,
        base_lco_with_value_hpx_agas_response_vector_get,
        base_lco_with_value_hpx_agas_response_vector_set,
        base_lco_with_value_hpx_memory_data_get,
        base_lco_with_value_hpx_memory_data_set,
        base_lco_with_value_std_string_get,
        base_lco_with_value_std_string_set,
        base_lco_with_value_std_bool_ptrdiff_get,
        base_lco_with_value_std_bool_ptrdiff_set,

        last_action_id
    };

    /// \endcond
}}

/// \cond NOINTERNAL

namespace hpx { namespace serialization
{
    template <
        typename Archive,
        typename Component, typename R, typename ...Args, typename Derived
    >
    HPX_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::actions::basic_action<Component, R(Args...), Derived>& t
      , unsigned int const version = 0
    )
    {}
}}

///////////////////////////////////////////////////////////////////////////////
/// \def HPX_DECLARE_ACTION(func, name)
/// \brief Declares an action type
///
#define HPX_DECLARE_ACTION(...)                                               \
    HPX_DECLARE_ACTION_(__VA_ARGS__)                                          \
    /**/

/// \cond NOINTERNAL

#define HPX_DECLARE_DIRECT_ACTION(...)                                        \
    HPX_DECLARE_ACTION(__VA_ARGS__)                                           \
    /**/

#define HPX_DECLARE_ACTION_(...)                                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DECLARE_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)                    \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DECLARE_ACTION_1(func)                                            \
    HPX_DECLARE_ACTION_2(func, BOOST_PP_CAT(func, _action))                   \
    /**/

#define HPX_DECLARE_ACTION_2(func, name) struct name;                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                    \
    HPX_DEFINE_GET_ACTION_NAME_(action, action)                               \
/**/
#define HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                       \
    namespace hpx { namespace actions { namespace detail {                    \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_action_name<action>()                                 \
        {                                                                     \
            return BOOST_PP_STRINGIZE(actionname);                            \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_REGISTER_ACTION_(...)                                             \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)                   \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_ACTION_1(action)                                         \
    HPX_REGISTER_ACTION_2(action, action)                                     \
/**/
#define HPX_REGISTER_ACTION_2(action, actionname)                             \
    HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                           \
    HPX_REGISTER_ACTION_INVOCATION_COUNT(action)                              \
    HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(action)                        \
    namespace hpx { namespace actions {                                       \
        template struct transfer_action<action>;                              \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)               \
    namespace hpx { namespace actions { namespace detail {                    \
        template <> HPX_ALWAYS_EXPORT                                         \
        char const* get_action_name<action>();                                \
    }}}                                                                       \
                                                                              \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct is_action<action>                                              \
          : std::true_type                                                    \
        {};                                                                   \
        template <>                                                           \
        struct needs_automatic_registration<action>                           \
          : std::false_type                                                   \
        {};                                                                   \
    }}                                                                        \
/**/

#define HPX_REGISTER_ACTION_DECLARATION_(...)                                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_ACTION_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_1(action)                             \
    HPX_REGISTER_ACTION_DECLARATION_2(action, action)                         \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_2(action, actionname)                 \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)                   \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_STACK(action, size)                                   \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_stacksize<action>                                       \
        {                                                                     \
            enum { value = size };                                            \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_USES_SMALL_STACK(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_small)            \
/**/
#define HPX_ACTION_USES_MEDIUM_STACK(action)                                  \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_medium)           \
/**/
#define HPX_ACTION_USES_LARGE_STACK(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_large)            \
/**/
#define HPX_ACTION_USES_HUGE_STACK(action)                                    \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_huge)             \
/**/
// This macro is deprecated. It expands to an inline function which will emit a
// warning.
#define HPX_ACTION_DOES_NOT_SUSPEND(action)                                    \
    HPX_DEPRECATED("HPX_ACTION_DOES_NOT_SUSPEND is deprecated and will be "    \
                   "removed in the next release")                              \
    static inline void BOOST_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)();   \
    void BOOST_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)()                  \
    {                                                                          \
        BOOST_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)();                  \
    }                                                                          \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_HAS_PRIORITY(action, priority)                             \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_priority<action>                                        \
        {                                                                     \
            enum { value = priority };                                        \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_HAS_LOW_PRIORITY(action)                                   \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_low)             \
/**/
#define HPX_ACTION_HAS_NORMAL_PRIORITY(action)                                \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_normal)          \
/**/
#define HPX_ACTION_HAS_CRITICAL_PRIORITY(action)                              \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_critical)        \
/**/

/// \endcond

/// \def HPX_REGISTER_ACTION_DECLARATION(action)
///
/// \brief Declare the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION_DECLARATION can be used to declare all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to declare the
/// boilerplate for.
///
/// This macro can be invoked with an optional second parameter. This parameter
/// specifies a unique name of the action to be used for serialization purposes.
/// The second parameter has to be specified if the first parameter is not
/// usable as a plain (non-qualified) C++ identifier, i.e. the first parameter
/// contains special characters which cannot be part of a C++ identifier, such
/// as '<', '>', or ':'.
///
/// \par Example:
///
/// \code
///      namespace app
///      {
///          // Define a simple component exposing one action 'print_greeting'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              void print_greeting ()
///              {
///                  hpx::cout << "Hey, how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greeting_action' representing the action.
///              HPX_DEFINE_COMPONENT_ACTION(server,
///                  print_greeting, print_greeting_action);
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION(app::server::print_greeting_action);
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// be visible in all translation units using the action, thus it is
/// recommended to place it into the header file defining the component.
#define HPX_REGISTER_ACTION_DECLARATION(...)                                  \
    HPX_REGISTER_ACTION_DECLARATION_(__VA_ARGS__)                             \
/**/

/// \def HPX_REGISTER_ACTION(action)
///
/// \brief Define the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION can be used to define all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to define the
/// boilerplate for.
///
/// This macro can be invoked with an optional second parameter. This parameter
/// specifies a unique name of the action to be used for serialization purposes.
/// The second parameter has to be specified if the first parameter is not
/// usable as a plain (non-qualified) C++ identifier, i.e. the first parameter
/// contains special characters which cannot be part of a C++ identifier, such
/// as '<', '>', or ':'.
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION or
/// \a HPX_DEFINE_PLAIN_ACTION macros. It has to occur exactly once for each of
/// the actions, thus it is recommended to place it into the source file defining
/// the component.
///
/// \note Only one of the forms of this macro \a HPX_REGISTER_ACTION or
///       \a HPX_REGISTER_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_REGISTER_ACTION(...)                                              \
    HPX_REGISTER_ACTION_(__VA_ARGS__)                                         \
/**/

/// \def HPX_REGISTER_ACTION_ID(action, actionname, actionid)
///
/// \brief Define the necessary component action boilerplate code and assign a
///        predefined unique id to the action.
///
/// The macro \a HPX_REGISTER_ACTION can be used to define all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to define the
/// boilerplate for.
///
/// The parameter \a actionname specifies an unique name of the action to be
/// used for serialization purposes.
/// The second parameter has to be usable as a plain (non-qualified) C++
/// identifier, it should not contain special characters which cannot be part
/// of a C++ identifier, such as '<', '>', or ':'.
///
/// The parameter \a actionid specifies an unique integer value which will be
/// used to represent the action during serialization.
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION or global actions
/// \a HPX_DEFINE_PLAIN_ACTION macros. It has to occur exactly once for each of
/// the actions, thus it is recommended to place it into the source file defining
/// the component.
///
/// \note Only one of the forms of this macro \a HPX_REGISTER_ACTION or
///       \a HPX_REGISTER_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_REGISTER_ACTION_ID(action, actionname, actionid)                  \
    HPX_REGISTER_ACTION_2(action, actionname)                                 \
    HPX_SERIALIZATION_ADD_CONSTANT_ENTRY(actionname, actionid)                \
/**/

#endif /*HPX_RUNTIME_ACTIONS_BASIC_ACTION_HPP*/
