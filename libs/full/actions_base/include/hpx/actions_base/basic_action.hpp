//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/actions_base/basic_action.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/actions_base/detail/action_factory.hpp>
#include <hpx/actions_base/detail/invocation_count_registry.hpp>
#include <hpx/actions_base/detail/per_action_data_counter_registry.hpp>
#include <hpx/actions_base/preassigned_action_id.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/actions_base/traits/action_stacksize.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/sync.hpp>
#include <hpx/async_local/sync_fwd.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/concurrency/itt_notify.hpp>
#endif

#include <boost/utility/string_ref.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions {

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Action>
        struct action_invoke
        {
            naming::address_type lva;
            naming::component_type comptype;

            template <typename... Ts>
            HPX_FORCEINLINE typename Action::internal_result_type operator()(
                Ts&&... vs) const
            {
                return Action::invoke(lva, comptype, std::forward<Ts>(vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        template <typename Action>
        class thread_function
        {
        public:
            template <typename... Ts>
            explicit thread_function(naming::id_type&& target,
                naming::address_type lva, naming::component_type comptype,
                Ts&&... vs)
              : target_(std::move(target))
              , lva_(lva)
              , comptype_(comptype)
              , args_(std::forward<Ts>(vs)...)
            {
            }

            threads::thread_result_type operator()(
                threads::thread_restart_state)
            {
                try
                {
                    LTM_(debug)
                        << "Executing " << Action::get_action_name(lva_) << ".";

                    // invoke the action, ignoring the return value
                    util::invoke_fused(action_invoke<Action>{lva_, comptype_},
                        std::move(args_));
                }
                catch (hpx::thread_interrupted const&)
                {    //-V565
                     /* swallow this exception */
                }
                catch (std::exception const& e)
                {
                    LTM_(error)
                        << "Unhandled exception while executing "
                        << Action::get_action_name(lva_) << ": " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(std::current_exception());
                }
                catch (...)
                {
                    LTM_(error) << "Unhandled exception while executing "
                                << Action::get_action_name(lva_);

                    // report this error to the console in any case
                    hpx::report_error(std::current_exception());
                }

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();

                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }

        private:
            // This holds the target alive, if necessary.
            naming::id_type target_;
            naming::address_type lva_;
            naming::component_type comptype_;
            typename Action::arguments_type args_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        class continuation_thread_function
        {
        public:
            template <typename... Ts>
            explicit continuation_thread_function(naming::id_type&& target,
                typename Action::continuation_type&& cont,
                naming::address_type lva, naming::component_type comptype,
                Ts&&... vs)
              : target_(std::move(target))
              , cont_(std::move(cont))
              , lva_(lva)
              , comptype_(comptype)
              , args_(std::forward<Ts>(vs)...)
            {
            }

            threads::thread_result_type operator()(
                threads::thread_restart_state)
            {
                LTM_(debug) << "Executing " << Action::get_action_name(lva_)
                            << " with continuation(" << cont_.get_id() << ")";

                actions::trigger(std::move(cont_),
                    util::functional::invoke_fused{},
                    action_invoke<Action>{lva_, comptype_}, std::move(args_));

                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }

        private:
            // This holds the target alive, if necessary.
            naming::id_type target_;
            typename Action::continuation_type cont_;
            naming::address_type lva_;
            naming::component_type comptype_;
            typename Action::arguments_type args_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct is_non_const_reference
          : std::integral_constant<bool,
                std::is_lvalue_reference<T>::value &&
                    !std::is_const<
                        typename std::remove_reference<T>::type>::value>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        inline std::string make_action_name(boost::string_ref action_name)
        {
            std::stringstream name;
            name << "action(" << action_name << ")";
            return name.str();
        }
    }    // namespace detail

    template <typename Component, typename R, typename... Args,
        typename Derived>
    struct basic_action<Component, R(Args...), Derived>
    {
        // Flag the use of raw pointer types as action arguments
        static_assert(!util::any_of<std::is_pointer<Args>...>::value,
            "Using raw pointers as arguments for actions is not supported.");

        // Flag the use of array types as action arguments
        static_assert(
            !util::any_of<std::is_array<
                typename std::remove_reference<Args>::type>...>::value,
            "Using arrays as arguments for actions is not supported.");

        // Flag the use of non-const reference types as action arguments
        static_assert(
            !util::any_of<detail::is_non_const_reference<Args>...>::value,
            "Using non-const references as arguments for actions is not "
            "supported.");

        using component_type = Component;
        using derived_type = Derived;

        // result_type represents the type returned when invoking operator()
        using result_type = typename traits::promise_local_result<R>::type;

        // The remote_result_type is the remote type for the type_continuation
        using remote_result_type =
            typename traits::action_remote_result<R>::type;

        // The local_result_type is the local type for the type_continuation
        using local_result_type =
            typename traits::promise_local_result<remote_result_type>::type;

        using continuation_type =
            hpx::actions::typed_continuation<local_result_type,
                remote_result_type>;

        static constexpr std::size_t arity = sizeof...(Args);

        using internal_result_type = R;
        using arguments_type = hpx::tuple<typename std::decay<Args>::type...>;

        using action_tag = void;

        ///////////////////////////////////////////////////////////////////////
        static std::string get_action_name(naming::address_type /*lva*/)
        {
            return detail::make_action_name(detail::get_action_name<Derived>());
        }

        template <typename... Ts>
        static R invoke(naming::address_type /*lva*/,
            naming::component_type /*comptype*/, Ts&&... /*vs*/);

        template <typename... Ts>
        static remote_result_type invoker(naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            using is_void = typename std::is_void<R>::type;
            return invoker_impl(
                is_void{}, lva, comptype, std::forward<Ts>(vs)...);
        }

    protected:
        template <typename... Ts>
        HPX_FORCEINLINE static remote_result_type invoker_impl(std::true_type,
            naming::address_type lva, naming::component_type comptype,
            Ts&&... vs)
        {
            Derived::invoke(lva, comptype, std::forward<Ts>(vs)...);
            return util::unused;
        }

        template <typename... Ts>
        HPX_FORCEINLINE static remote_result_type invoker_impl(std::false_type,
            naming::address_type lva, naming::component_type comptype,
            Ts&&... vs)
        {
            return Derived::invoke(lva, comptype, std::forward<Ts>(vs)...);
        }

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case no continuation has been supplied.
        template <typename... Ts>
        static threads::thread_function_type construct_thread_function(
            naming::id_type target, naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            if (target &&
                target.get_management_type() == naming::id_type::unmanaged)
                target = {};

            using thread_function = detail::thread_function<Derived>;
            return traits::action_decorate_function<Derived>::call(lva,
                thread_function(
                    std::move(target), lva, comptype, std::forward<Ts>(vs)...));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename... Ts>
        static threads::thread_function_type construct_thread_function(
            naming::id_type target, continuation_type&& cont,
            naming::address_type lva, naming::component_type comptype,
            Ts&&... vs)
        {
            if (target &&
                target.get_management_type() == naming::id_type::unmanaged)
                target = {};

            using thread_function =
                detail::continuation_thread_function<Derived>;
            return traits::action_decorate_function<Derived>::call(lva,
                thread_function(std::move(target), std::move(cont), lva,
                    comptype, std::forward<Ts>(vs)...));
        }

        // direct execution
        template <typename... Ts>
        static HPX_FORCEINLINE remote_result_type execute_function(
            naming::address_type lva, naming::component_type comptype,
            Ts&&... vs)
        {
            LTM_(debug) << "basic_action::execute_function"
                        << Derived::get_action_name(lva);

            return invoker(lva, comptype, std::forward<Ts>(vs)...);
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        template <typename IdOrPolicy, typename Policy, typename... Ts>
        HPX_FORCEINLINE static result_type sync_invoke(Policy const& policy,
            IdOrPolicy const& id_or_policy, error_code&, Ts&&... vs)
        {
            return hpx::sync<basic_action>(
                policy, id_or_policy, std::forward<Ts>(vs)...);
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        HPX_FORCEINLINE result_type operator()(launch policy,
            naming::id_type const& id, error_code& ec, Ts&&... vs) const
        {
            return sync_invoke(policy, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        HPX_FORCEINLINE result_type operator()(
            naming::id_type const& id, error_code& ec, Ts&&... vs) const
        {
            return sync_invoke(launch::sync, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        HPX_FORCEINLINE result_type operator()(
            launch policy, naming::id_type const& id, Ts&&... vs) const
        {
            return sync_invoke(policy, id, throws, std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        HPX_FORCEINLINE result_type operator()(
            naming::id_type const& id, Ts&&... vs) const
        {
            return sync_invoke(
                launch::sync, id, throws, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type>::type
        operator()(launch policy, DistPolicy const& dist_policy, error_code& ec,
            Ts&&... vs) const
        {
            return sync_invoke(
                policy, dist_policy, ec, std::forward<Ts>(vs)...);
        }

        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type>::type
        operator()(
            DistPolicy const& dist_policy, error_code& ec, Ts&&... vs) const
        {
            return sync_invoke(
                launch::sync, dist_policy, ec, std::forward<Ts>(vs)...);
        }

        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type>::type
        operator()(
            launch policy, DistPolicy const& dist_policy, Ts&&... vs) const
        {
            return sync_invoke(
                policy, dist_policy, throws, std::forward<Ts>(vs)...);
        }

        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            result_type>::type
        operator()(DistPolicy const& dist_policy, Ts&&... vs) const
        {
            return sync_invoke(
                launch::sync, dist_policy, throws, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        /// retrieve component type
        static int get_component_type()
        {
            return static_cast<int>(
                components::get_component_type<Component>());
        }

        using direct_execution = std::false_type;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static constexpr actions::action_flavor get_action_type()
        {
            return actions::action_flavor::plain_action;
        }

        /// Extract the current invocation count for this action
        static std::int64_t get_invocation_count(bool reset)
        {
            return util::get_and_reset_value(invocation_count_, reset);
        }

    private:
        static std::atomic<std::int64_t> invocation_count_;

    protected:
        static void increment_invocation_count()
        {
            ++invocation_count_;
        }
    };

    template <typename Component, typename R, typename... Args,
        typename Derived>
    std::atomic<std::int64_t>
        basic_action<Component, R(Args...), Derived>::invocation_count_(0);

    namespace detail {
        template <typename Action>
        void register_local_action_invocation_count(
            invocation_count_registry& registry)
        {
            registry.register_class(
                hpx::actions::detail::get_action_name<Action>(),
                &Action::get_invocation_count);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // simple type allowing to distinguish whether an action is the most
        // derived one
        struct this_type
        {
        };

        template <typename Action, typename Derived>
        struct action_type
        {
            using type = Derived;
        };

        template <typename Action>
        struct action_type<Action, this_type>
        {
            using type = Action;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename TF, TF F, typename Derived = detail::this_type>
    struct action;

    ///////////////////////////////////////////////////////////////////////////
    template <typename TF, TF F, typename Derived = detail::this_type>
    struct direct_action
      : action<TF, F,
            typename detail::action_type<direct_action<TF, F, Derived>,
                Derived>::type>
    {
        using derived_type =
            typename detail::action_type<direct_action, Derived>::type;

        using direct_execution = std::true_type;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static constexpr actions::action_flavor action_flavor()
        {
            return actions::action_flavor::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Base template allowing to generate a concrete action type from a function
    // pointer. It is instantiated only if the supplied pointer is not a
    // supported function pointer.
    template <typename TF, TF F, typename Derived = detail::this_type,
        typename Direct = std::false_type>
    struct make_action
#if defined(HPX_HAVE_ACTION_BASE_COMPATIBILITY)
      : action<TF, F, Derived>
#endif
    {
        using type = action<TF, F, Derived>;
    };

    template <typename TF, TF F, typename Derived>
    struct make_action<TF, F, Derived, std::true_type>
#if defined(HPX_HAVE_ACTION_BASE_COMPATIBILITY)
      : direct_action<TF, F, Derived>
#endif
    {
        using type = direct_action<TF, F, Derived>;
    };

    template <typename TF, TF F, typename Derived = detail::this_type>
    struct make_direct_action : make_action<TF, F, Derived, std::true_type>
    {
    };

// Macros usable to refer to an action given the function to expose
#define HPX_MAKE_ACTION(func)                                                  \
    hpx::actions::make_action<decltype(&func), &func> /**/ /**/
#define HPX_MAKE_DIRECT_ACTION(func)                                           \
    hpx::actions::make_direct_action<decltype(&func), &func> /**/ /**/

        /// \endcond
}}    // namespace hpx::actions

/// \cond NOINTERNAL

///////////////////////////////////////////////////////////////////////////////
/// \def HPX_DECLARE_ACTION(func, name)
/// \brief Declares an action type
///
#define HPX_DECLARE_ACTION(...)                                                \
    HPX_DECLARE_ACTION_(__VA_ARGS__)                                           \
    /**/

/// \cond NOINTERNAL

#define HPX_DECLARE_DIRECT_ACTION(...)                                         \
    HPX_DECLARE_ACTION(__VA_ARGS__)                                            \
    /**/

#define HPX_DECLARE_ACTION_(...)                                               \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_DECLARE_ACTION_, HPX_PP_NARGS(__VA_ARGS__))(  \
        __VA_ARGS__))                                                          \
    /**/

#define HPX_DECLARE_ACTION_1(func)                                             \
    HPX_DECLARE_ACTION_2(func, HPX_PP_CAT(func, _action))                      \
    /**/

#define HPX_DECLARE_ACTION_2(func, name)                                       \
    struct name;                                                               \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ACTION_(...)                                              \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_ACTION_, HPX_PP_NARGS(__VA_ARGS__))( \
        __VA_ARGS__))                                                          \
/**/
#define HPX_REGISTER_ACTION_1(action)                                          \
    HPX_REGISTER_ACTION_2(action, action)                                      \
    /**/

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#define HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname)                     \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT util::itt::string_handle const&              \
                get_action_name_itt<action>()                                  \
                {                                                              \
                    static util::itt::string_handle sh(                        \
                        HPX_PP_STRINGIZE(actionname));                         \
                    return sh;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
/**/
#else
#define HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname)
#endif

#if !defined(HPX_HAVE_NETWORKING)

#define HPX_DEFINE_GET_ACTION_NAME(action) /**/
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#define HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname) /**/
#endif
#define HPX_REGISTER_ACTION_EXTERN_DECLARATION(action) /**/

#define HPX_REGISTER_ACTION_2(action, actionname)                              \
    HPX_REGISTER_ACTION_INVOCATION_COUNT(action)                               \
    HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(action)                         \
    /**/

#define HPX_REGISTER_ACTION_DECLARATION_2(action, actionname) /**/

#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_ACTION_USES_STACK(action, size)  /**/
#define HPX_ACTION_USES_SMALL_STACK(action)  /**/
#define HPX_ACTION_USES_MEDIUM_STACK(action) /**/
#define HPX_ACTION_USES_LARGE_STACK(action)  /**/
#define HPX_ACTION_USES_HUGE_STACK(action)   /**/
#else
#define HPX_ACTION_USES_STACK(action, size)                                    \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct action_stacksize<action>                                    \
            {                                                                  \
                HPX_STATIC_CONSTEXPR threads::thread_stacksize value = size;   \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_ACTION_USES_SMALL_STACK(action)                                    \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize::small_)           \
/**/
#define HPX_ACTION_USES_MEDIUM_STACK(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize::medium)           \
/**/
#define HPX_ACTION_USES_LARGE_STACK(action)                                    \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize::large)            \
/**/
#define HPX_ACTION_USES_HUGE_STACK(action)                                     \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize::huge)             \
/**/
#endif

#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_ACTION_DOES_NOT_SUSPEND(action) /**/
#else
// This macro is deprecated. It expands to an inline function which will emit a
// warning.
#define HPX_ACTION_DOES_NOT_SUSPEND(action)                                    \
    HPX_DEPRECATED_V(1, 4,                                                     \
        "HPX_ACTION_DOES_NOT_SUSPEND is deprecated and will be "               \
        "removed in the future")                                               \
    static inline void HPX_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)();     \
    void HPX_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)()                    \
    {                                                                          \
        HPX_PP_CAT(HPX_ACTION_DOES_NOT_SUSPEND_, action)();                    \
    }                                                                          \
/**/
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_ACTION_HAS_PRIORITY(action, priority)      /**/
#define HPX_ACTION_HAS_LOW_PRIORITY(action)            /**/
#define HPX_ACTION_HAS_NORMAL_PRIORITY(action)         /**/
#define HPX_ACTION_HAS_HIGH_PRIORITY(action)           /**/
#define HPX_ACTION_HAS_HIGH_RECURSIVE_PRIORITY(action) /**/
// obsolete, kept for compatibility
#define HPX_ACTION_HAS_CRITICAL_PRIORITY(action) /**/
#else
///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_HAS_PRIORITY(action, priority)                              \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct action_priority<action>                                     \
            {                                                                  \
                HPX_STATIC_CONSTEXPR threads::thread_priority value =          \
                    priority;                                                  \
            };                                                                 \
            /* make sure the action is not executed directly */                \
            template <>                                                        \
            struct has_decorates_action<action> : std::true_type               \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_ACTION_HAS_LOW_PRIORITY(action)                                    \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority::low)             \
/**/
#define HPX_ACTION_HAS_NORMAL_PRIORITY(action)                                 \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority::normal)          \
/**/
#define HPX_ACTION_HAS_HIGH_PRIORITY(action)                                   \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority::high)            \
/**/
#define HPX_ACTION_HAS_HIGH_RECURSIVE_PRIORITY(action)                         \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority::high_recursive)  \
/**/

// obsolete, kept for compatibility
#define HPX_ACTION_HAS_CRITICAL_PRIORITY(action)                               \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority::high_recursive)  \
/**/
#endif

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
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_REGISTER_ACTION_DECLARATION(...) /**/
#else
#define HPX_REGISTER_ACTION_DECLARATION(...)                                   \
    HPX_REGISTER_ACTION_DECLARATION_(__VA_ARGS__)                              \
    /**/

#define HPX_REGISTER_ACTION_DECLARATION_(...)                                  \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_ACTION_DECLARATION_,                 \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_1(action)                              \
    HPX_REGISTER_ACTION_DECLARATION_2(action, action)                          \
/**/
#endif

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
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_REGISTER_ACTION(...) /**/
#else
#define HPX_REGISTER_ACTION(...)                                               \
    HPX_REGISTER_ACTION_(__VA_ARGS__)                                          \
/**/
#endif

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
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_REGISTER_ACTION_ID(action, actionname, actionid) /**/
#else
#define HPX_REGISTER_ACTION_ID(action, actionname, actionid)                   \
    HPX_REGISTER_ACTION_2(action, actionname)                                  \
    HPX_REGISTER_ACTION_FACTORY_ID(actionname, actionid)                       \
/**/
#endif
