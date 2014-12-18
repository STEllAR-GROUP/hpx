//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONST_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_RUNTIME_ACTIONS_CONST_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

// now generate the rest, which is platform independent
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/component_const_action_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/component_const_action_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_const_action_implementations.hpp"))        \
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

#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
    BOOST_PP_COMMA_IF(n)                                                      \
    util::get<n>(std::forward<Arguments>(data))                             \
    /**/
#define HPX_REMOVE_QUALIFIERS(z, n, data)                                     \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename util::decay<BOOST_PP_CAT(T, n)>::type                        \
    /**/

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with result
    template <
        typename Component, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        R (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    class basic_action_impl<
            R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : public basic_action<
            Component const, R(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)), Derived>
            base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

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

                    (get_lva<Component const>::call(lva)->*F)(
                        HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                }
                catch (hpx::thread_interrupted const&) {
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";

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
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the
        // applier in case no continuation has been supplied.
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the
        // applier in case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "basic_action_impl" << N
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            return (get_lva<Component const>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        R (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct action<
            R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        R (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>,
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
    //  N parameter version, no result type
    template <
        typename Component, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)) const, typename Derived>
    class basic_action_impl<
            void (Component::*)(BOOST_PP_ENUM_PARAMS(N, T)) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)), Derived>
            base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

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

                    (get_lva<Component const>::call(lva)->*F)(
                        HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                }
                catch (hpx::thread_interrupted const&) {
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";

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
        // instantiate the basic_action_impl type. This is used by the applier in
        // case no continuation has been supplied.
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "basic_action_impl" << N
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            (get_lva<Component const>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
            return util::unused;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT

#undef N

#endif
