//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_action_implementations.hpp"))              \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
    BOOST_PP_COMMA_IF(n)                                                      \
    util::detail::move_if_no_ref<                                             \
        typename util::detail::remove_reference<Arguments>::type::            \
            BOOST_PP_CAT(member_type, n)>::call(data. BOOST_PP_CAT(a, n))     \
    /**/
#define HPX_REMOVE_QUALIFIERS(z, n, data)                                     \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename detail::remove_qualifiers<BOOST_PP_CAT(T, n)>::type          \
    /**/

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class BOOST_PP_CAT(base_result_action, N)
      : public action<
            Component, Action, Result,
            BOOST_PP_CAT(hpx::util::tuple, N)<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef BOOST_PP_CAT(hpx::util::tuple, N)<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            result_type operator()(
                naming::address::address_type lva,
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    // The arguments are moved here. This function is called from a
                    // bound functor. In order to do true perfect forwarding in an
                    // asynchronous operation. These bound variables must be moved
                    // out of the bound object.
                    (get_lva<Component>::call(lva)->*F)(
                        HPX_ENUM_MOVE_ARGS(N, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();

                        // report this error to the console in any case
                        hpx::report_error(boost::current_exception());
                    }
                }

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();
                return threads::terminated;
            }
        };

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case no continuation has been supplied.
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case a continuation has been supplied
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct BOOST_PP_CAT(result_action, N)
      : BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(result_action, N)<
                    Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(result_action, N)<
                Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
                    Priority>,
            Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    namespace detail
    {
        template <typename Obj, typename Result,
            BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct synthesize_const_mf<Obj,
            Result (*)(BOOST_PP_ENUM_PARAMS(N, T))>
        {
            typedef Result (Obj::*type)(BOOST_PP_ENUM_PARAMS(N, T)) const;
        };

        template <typename Obj, typename Result,
            BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct synthesize_const_mf<Obj,
            Result (Obj::*)(BOOST_PP_ENUM_PARAMS(N, T)) const>
        {
            typedef Result (Obj::*type)(BOOST_PP_ENUM_PARAMS(N, T)) const;
        };

        template <typename Result, BOOST_PP_ENUM_PARAMS(N, typename T)>
        typename boost::mpl::identity<Result (*)(BOOST_PP_ENUM_PARAMS(N, T))>::type
        replicate_type(Result (*p)(BOOST_PP_ENUM_PARAMS(N, T)));
    }
#endif

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(result_action, N)<
            Component, Result, BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived>
    {};

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(result_action, N)<
            Component const, Result,
            BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived>
    {};

#else

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::false_>
        : boost::mpl::identity<BOOST_PP_CAT(result_action, N)<
            Component, Result, BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived> >
    {};

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : boost::mpl::identity<BOOST_PP_CAT(result_action, N)<
            Component const, Result,
            BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        typename Derived = detail::this_type>
    struct BOOST_PP_CAT(direct_result_action, N)
      : BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_result_action, N)<
                    Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_result_action, N)<
                Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_result_action, N)<
            Component, Result, BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived>
    {};

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_result_action, N)<
            Component const, Result,
            BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived>
    {};
#else
    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::true_>
      : boost::mpl::identity<BOOST_PP_CAT(direct_result_action, N)<
            Component, Result, BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived> >
    {};

    template <typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : boost::mpl::identity<BOOST_PP_CAT(direct_result_action, N)<
            Component const, Result,
            BOOST_PP_CAT(component_result_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class BOOST_PP_CAT(base_action, N)
      : public action<
            Component, Action, util::unused_type,
            BOOST_PP_CAT(hpx::util::tuple, N)<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef BOOST_PP_CAT(hpx::util::tuple, N)<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            result_type operator()(
                naming::address::address_type lva,
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    // The arguments are moved here. This function is called from a
                    // bound functor. In order to do true perfect forwarding in an
                    // asynchronous operation. These bound variables must be moved
                    // out of the bound object.
                    (get_lva<Component>::call(lva)->*F)(
                        HPX_ENUM_MOVE_ARGS(N, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();

                        // report this error to the console in any case
                        hpx::report_error(boost::current_exception());
                    }
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
        // instantiate the base_actionN type. This is used by the applier in
        // case no continuation has been supplied.
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_actionN type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct BOOST_PP_CAT(action, N)
      : BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(action, N)<
                    Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(action, N)<
                Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F, Priority>,
            Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(action, N)<
            Component, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived>
    {};

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(action, N)<
            Component const, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived>
    {};
#else
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::false_>
      : boost::mpl::identity<BOOST_PP_CAT(action, N)<
            Component, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived> >
    {};

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : boost::mpl::identity<BOOST_PP_CAT(action, N)<
            Component const, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, threads::thread_priority_default,
            Derived> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        typename Derived = detail::this_type>
    struct BOOST_PP_CAT(direct_action, N)
      : BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_action, N)<
                    Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_action, N)<
                Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
            return util::unused;
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_action, N)<
            Component, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived>
    {};

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_action, N)<
            Component const, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived>
    {};
#else
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)),
            F, Derived, boost::mpl::true_>
      : boost::mpl::identity<BOOST_PP_CAT(direct_action, N)<
            Component, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived> >
    {};

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : boost::mpl::identity<BOOST_PP_CAT(direct_action, N)<
            Component const, BOOST_PP_CAT(component_action_arg, N),
            BOOST_PP_ENUM_PARAMS(N, T), F, Derived> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <
        typename Component, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority,
        typename Derived>
    struct BOOST_PP_CAT(result_action, N)<Component, void, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>
      : BOOST_PP_CAT(action, N)<Component, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>
    {};
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT
// #undef HPX_ACTION_ARGUMENT
#undef N

#endif

