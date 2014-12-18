//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_IMPLEMENTATIONS_NOV_14_2008_0811PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_IMPLEMENTATIONS_NOV_14_2008_0811PM

#include <hpx/config/forceinline.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/plain_action_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/plain_action_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/plain_action_implementations.hpp"))                  \
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
    //  N parameter version, with R
    template <
        typename R, BOOST_PP_ENUM_PARAMS(N, typename T),
        R (*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    class basic_action_impl<R (*)(BOOST_PP_ENUM_PARAMS(N, T)), F, Derived>
      : public basic_action<
            components::server::plain_function<Derived>,
            R(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)),
            Derived>
    {
    public:
        typedef basic_action<
            components::server::plain_function<Derived>,
            R(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)), Derived>
            base_type;

        // Only localities are valid targets for a plain action
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
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
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";

                    // call the function, ignoring the return value
                    F(HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                }
                catch (hpx::thread_interrupted const&) {
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";

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
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_function(
                    cont, F, std::forward<Arguments>(args)));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            return F(BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T),
        void (*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>
    class basic_action_impl<void (*)(BOOST_PP_ENUM_PARAMS(N, T)), F, Derived>
      : public basic_action<
            components::server::plain_function<Derived>,
            util::unused_type(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)),
            Derived>
    {
    public:
        typedef basic_action<
            components::server::plain_function<Derived>,
            util::unused_type(BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)), Derived>
            base_type;

        // Only localities are valid targets for a plain action
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
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
                HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";

                    // call the function, ignoring the return value
                    F(HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                }
                catch (hpx::thread_interrupted const&) {
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";

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
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args)));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_function_void(
                    cont, F, std::forward<Arguments>(args)));
        }

        //  direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            F(BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
            return util::unused;
        }
    };
}}

// Disabling the guid initialization stuff for plain actions
namespace hpx { namespace traits
{
    template <typename R, BOOST_PP_ENUM_PARAMS(N, typename Arg),
        R (*F)(BOOST_PP_ENUM_PARAMS(N, Arg)), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::action<
                    R (*)(BOOST_PP_ENUM_PARAMS(N, Arg)), F, Derived> >, Enable>
      : boost::mpl::false_
    {};

    template <typename R, BOOST_PP_ENUM_PARAMS(N, typename Arg),
        R (*F)(BOOST_PP_ENUM_PARAMS(N, Arg)), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::direct_action<
                    R (*)(BOOST_PP_ENUM_PARAMS(N, Arg)), F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT
#undef N

#endif

