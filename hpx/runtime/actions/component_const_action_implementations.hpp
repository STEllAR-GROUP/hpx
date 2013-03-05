//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONST_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_RUNTIME_ACTIONS_CONST_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

// generate platform specific code
#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (4, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_const_action_implementations.hpp", 1))     \
    /**/

#include BOOST_PP_ITERATE()
#endif

// now generate the rest, which is platform independent
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/component_const_action_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/component_const_action_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (4, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_const_action_implementations.hpp", 2))     \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

namespace hpx { namespace actions { namespace detail
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
}}}

#endif

#if BOOST_PP_ITERATION_FLAGS() == 2

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
        typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    class BOOST_PP_CAT(base_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : public action<
            Component const, Result,
            BOOST_PP_CAT(hpx::util::tuple, N)<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef BOOST_PP_CAT(hpx::util::tuple, N)<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component const, result_type, arguments_type, Derived>
            base_type;

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    // The arguments are moved here. This function is called from a
                    // bound functor. In order to do true perfect forwarding in an
                    // asynchronous operation. These bound variables must be moved
                    // out of the bound object.
                    (get_lva<Component const>::call(lva)->*F)(
                        HPX_ENUM_MOVE_ARGS(N, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component const>::call(lva)) << "): " << e.what();

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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    lva, BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)), lva));
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
            return boost::move(Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component const>::call(lva),
                        boost::forward<Arguments>(args)), lva));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct BOOST_PP_CAT(result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : BOOST_PP_CAT(base_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                BOOST_PP_CAT(result_action, N)<
                    Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(result_action, N), Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
    {
        typedef BOOST_PP_CAT(result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct BOOST_PP_CAT(direct_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : BOOST_PP_CAT(base_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_result_action, N)<
                    Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_result_action, N), Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            return (get_lva<Component const>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
    {
        typedef BOOST_PP_CAT(direct_result_action, N)<
            Result (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <
        typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    class BOOST_PP_CAT(base_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : public action<
            Component const, util::unused_type,
            BOOST_PP_CAT(hpx::util::tuple, N)<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef BOOST_PP_CAT(hpx::util::tuple, N)<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component const, result_type, arguments_type, Derived>
            base_type;

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    // The arguments are moved here. This function is called from a
                    // bound functor. In order to do true perfect forwarding in an
                    // asynchronous operation. These bound variables must be moved
                    // out of the bound object.
                    (get_lva<Component const>::call(lva)->*F)(
                        HPX_ENUM_MOVE_ARGS(N, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component const>::call(lva)) << "): " << e.what();

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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)), lva));
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
            return boost::move(Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component const>::call(lva),
                        boost::forward<Arguments>(args)), lva));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct BOOST_PP_CAT(action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : BOOST_PP_CAT(base_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                BOOST_PP_CAT(action, N)<
                    void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(action, N), Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::false_>
      : BOOST_PP_CAT(action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
    {
        typedef BOOST_PP_CAT(action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct BOOST_PP_CAT(direct_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : BOOST_PP_CAT(base_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_action, N)<
                    void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_action, N), Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            (get_lva<Component const>::call(lva)->*F)(
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

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    struct make_action<void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const,
            F, Derived, boost::mpl::true_>
      : BOOST_PP_CAT(direct_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
    {
        typedef BOOST_PP_CAT(direct_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <
        typename Component,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct BOOST_PP_CAT(result_action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : BOOST_PP_CAT(action, N)<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
    {};
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT

#endif // #if BOOST_PP_ITERATION_FLAGS() == 2

#undef N

#endif

