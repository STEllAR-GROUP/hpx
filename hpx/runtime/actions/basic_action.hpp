//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/actions/basic_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_BASIC_ACTION_NOV_14_2008_0711PM)
#define HPX_RUNTIME_ACTIONS_BASIC_ACTION_NOV_14_2008_0711PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#include <boost/mpl/if.hpp>
#include <boost/serialization/access.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/cat.hpp>

#include <sstream>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action, typename F, typename ...Ts>
        struct continuation_thread_function
        {
            HPX_MOVABLE_BUT_NOT_COPYABLE(continuation_thread_function);

        public:
            template <typename F_, typename ...Ts_>
            explicit continuation_thread_function(continuation_type cont,
                naming::address::address_type lva, F_&& f, Ts_&&... vs)
              : cont_(std::move(cont)), lva_(lva)
              , f_(util::deferred_call(
                    std::forward<F_>(f), std::forward<Ts_>(vs)...))
            {}

            continuation_thread_function(continuation_thread_function && other)
              : cont_(std::move(other.cont_)), lva_(std::move(other.lva_))
              , f_(std::move(other.f_))
            {}

            typedef threads::thread_state_enum result_type;

            BOOST_FORCEINLINE result_type operator()(threads::thread_state_ex_enum)
            {
                LTM_(debug) << "Executing " << Action::get_action_name(lva_)
                    << " with continuation(" << cont_->get_gid() << ")";

                actions::trigger(*cont_, f_);
                return threads::terminated;
            }

        private:
            continuation_type cont_;
            naming::address::address_type lva_;
            util::detail::deferred_call_impl<
                typename util::decay<F>::type
              , util::tuple<typename util::decay_unwrap<Ts>::type...>
            > f_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Signature         return type and arguments
    /// \tparam Derived           derived action class
    template <typename Component, typename Signature, typename Derived>
    struct basic_action;

    template <typename Component, typename R, typename ...Args, typename Derived>
    struct basic_action<Component, R(Args...), Derived>
    {
        typedef Component component_type;
        typedef Derived derived_type;

        typedef typename boost::mpl::if_c<
            boost::is_void<R>::value, util::unused_type, R
        >::type result_type;
        typedef typename traits::promise_local_result<R>::type local_result_type;
        typedef typename detail::remote_action_result<R>::type remote_result_type;

        static const std::size_t arity = sizeof...(Args);
        typedef util::tuple<typename util::decay<Args>::type...> arguments_type;

        typedef void action_tag;

        ///////////////////////////////////////////////////////////////////////
        static std::string get_action_name(naming::address::address_type /*lva*/)
        {
            std::stringstream name;
            name << "action(" << detail::get_action_name<Derived>() << ")";
            return name.str();
        }

        static bool is_target_valid(naming::id_type const& id)
        {
            return true;        // by default we don't do any verification
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type /*lva*/, Ts&&... /*vs*/);

    protected:
        struct invoker
        {
            template <typename ...Ts>
            typename boost::disable_if_c<
                (boost::is_void<R>::value && util::detail::pack<Ts...>::size >= 0),
                result_type
            >::type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                return Derived::invoke(lva, std::forward<Ts>(vs)...);
            }

            template <typename ...Ts>
            typename boost::enable_if_c<
                (boost::is_void<R>::value && util::detail::pack<Ts...>::size >= 0),
                result_type
            >::type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                Derived::invoke(lva, std::forward<Ts>(vs)...);
                return util::unused;
            }
        };

        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
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
                catch (hpx::exception const& e) {
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
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Ts&&... vs)
        {
            typedef detail::continuation_thread_function<
                Derived, invoker, naming::address::address_type, Ts...
            > thread_function;

            return traits::action_decorate_function<Derived>::call(lva,
                thread_function(std::move(cont), lva, invoker(),
                    lva, std::forward<Ts>(vs)...));
        }

        // direct execution
        template <typename ...Ts>
        static BOOST_FORCEINLINE result_type
        execute_function(naming::address::address_type lva, Ts&&... vs)
        {
            LTM_(debug)
                << "basic_action::execute_function"
                << Derived::get_action_name(lva);

            return invoker()(lva, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        typedef typename traits::is_future<local_result_type>::type is_future_pred;

        template <typename LocalResult>
        struct sync_invoke
        {
            template <typename ...Ts>
            BOOST_FORCEINLINE static LocalResult call(
                boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
                naming::id_type const& id, error_code& ec, Ts&&... vs)
            {
                return hpx::async<basic_action>(policy, id,
                    std::forward<Ts>(vs)...).get(ec);
            }

            template <typename ...Ts>
            BOOST_FORCEINLINE static LocalResult call(
                boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
                naming::id_type const& id, error_code& /*ec*/, Ts&&... vs)
            {
                return hpx::async<basic_action>(policy, id,
                    std::forward<Ts>(vs)...);
            }
        };

        template <typename ...Ts>
        BOOST_FORCEINLINE local_result_type operator()(
            BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& id,
            error_code& ec, Ts&&... vs) const
        {
            return util::void_guard<local_result_type>(),
                sync_invoke<local_result_type>::call(
                    is_future_pred(), policy, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        BOOST_FORCEINLINE local_result_type operator()(
            naming::id_type const& id, error_code& ec, Ts&&... vs) const
        {
            return (*this)(launch::all, id, ec, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        BOOST_FORCEINLINE local_result_type operator()(
            BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& id,
            Ts&&... vs) const
        {
            return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        BOOST_FORCEINLINE local_result_type operator()(
            naming::id_type const& id, Ts&&... vs) const
        {
            return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
        }

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

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // simple type allowing to distinguish whether an action is the most
        // derived one
        struct this_type {};

        template <typename Action, typename Derived>
        struct action_type
          : boost::mpl::if_<boost::is_same<Derived, this_type>, Action, Derived>
        {};
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

        typedef boost::mpl::false_ direct_execution;
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

        typedef boost::mpl::true_ direct_execution;

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
        typename Direct = boost::mpl::false_>
    struct make_action;

    template <typename TF, TF F, typename Derived>
    struct make_action<TF, F, Derived, boost::mpl::false_>
      : action<TF, F, Derived>
    {
        typedef action<TF, F, Derived> type;
    };

    template <typename TF, TF F, typename Derived>
    struct make_action<TF, F, Derived, boost::mpl::true_>
      : direct_action<TF, F, Derived>
    {
        typedef direct_action<TF, F, Derived> type;
    };

    template <typename TF, TF F, typename Derived = detail::this_type>
    struct make_direct_action
      : make_action<TF, F, Derived, boost::mpl::true_>
    {};

    // Macros usable to refer to an action given the function to expose
    #define HPX_MAKE_ACTION(func)                                             \
        hpx::actions::make_action<decltype(&func), &func>        /**/         \
    /**/
    #define HPX_MAKE_DIRECT_ACTION(func)                                      \
        hpx::actions::make_direct_action<decltype(&func), &func> /**/         \
    /**/

    /// \endcond
}}

#include <hpx/config/warnings_suffix.hpp>

/// \cond NOINTERNAL

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_HELPER(action, actionname)                          \
    hpx::actions::detail::register_base_helper<action>                        \
            BOOST_PP_CAT(                                                     \
                BOOST_PP_CAT(__hpx_action_register_base_helper_, __LINE__),   \
                _##actionname);                                               \
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

#define HPX_ACTION_REGISTER_ACTION_FACTORY(Action, Name)                      \
    static ::hpx::actions::detail::action_registration<Action>                \
        const BOOST_PP_CAT(Name, _action_factory_registration) =              \
        ::hpx::actions::detail::action_registration<Action>();                \
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
    HPX_ACTION_REGISTER_ACTION_FACTORY(hpx::actions::transfer_action<action>, \
        actionname)                                                           \
    HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                           \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)              \
    namespace hpx { namespace actions { namespace detail {                    \
        template <> HPX_ALWAYS_EXPORT                                         \
        char const* get_action_name<action>();                                \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(action)              \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<action>                           \
          : boost::mpl::false_                                                \
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
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)                  \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(                         \
        hpx::actions::transfer_action<action>)                                \
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
#define HPX_ACTION_DOES_NOT_SUSPEND(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_nostack)          \
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
///          // Define a simple component exposing one action 'print_greating'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              void print_greating ()
///              {
///                  hpx::cout << "Hey, how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greating_action' representing the action.
///              HPX_DEFINE_COMPONENT_ACTION(server, print_greating, print_greating_action);
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION(app::server::print_greating_action);
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// be visible in all translation units using the action, thus it is
/// recommended to place it into the header file defining the component.
#define HPX_REGISTER_ACTION_DECLARATION(...)                                  \
    HPX_REGISTER_ACTION_DECLARATION_(__VA_ARGS__)                             \
/**/

/// \def HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(template, action)
///
/// \brief Declare the necessary component action boilerplate code for actions
///        taking template type arguments.
///
/// The macro \a HPX_REGISTER_ACTION_DECLARATION_TEMPLATE can be used to
/// declare all the boilerplate code which is required for proper functioning
/// of component actions in the context of HPX, if those actions take template
/// type arguments.
///
/// The parameter \a template specifies the list of template type declarations
/// for the action type. This argument has to be wrapped into an additional
/// pair of parenthesis.
///
/// The parameter \a action is the type of the action to declare the
/// boilerplate for. This argument has to be wrapped into an additional pair
/// of parenthesis.
///
/// \par Example:
///
/// \code
///      namespace app
///      {
///          // Define a simple component exposing one action 'print_greating'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              template <typename T>
///              void print_greating (T t)
///              {
///                  hpx::cout << "Hey " << t << ", how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greating_action' representing the action.
///
///              // Actions with template arguments (like print_greating<>()
///              // above) require special type definitions. The simplest way
///              // to define such an action type is by deriving from the HPX
///              // facility make_action:
///              template <typename T>
///              struct print_greating_action
///                : hpx::actions::make_action<
///                      void (server::*)(T), &server::template print_greating<T>,
///                      print_greating_action<T> >
///              {};
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION_TEMPLATE((template T), (app::server::print_greating_action<T>));
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined as above. It has to be visible in all translation units using the
/// action, thus it is recommended to place it into the header file defining the
/// component.
#define HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(TEMPLATE, TYPE)              \
    HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(TEMPLATE, TYPE)                \
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
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// occur exactly once for each of the actions, thus it is recommended to
/// place it into the source file defining the component. There is no need
/// to use this macro for actions which have template type arguments (see
/// \a HPX_REGISTER_ACTION_DECLARATION_TEMPLATE)
#define HPX_REGISTER_ACTION(...)                                              \
    HPX_REGISTER_ACTION_(__VA_ARGS__)                                         \
/**/

#endif
