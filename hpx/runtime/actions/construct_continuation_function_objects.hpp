//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_OBJECTS_MAY_08_2012_0610PM)
#define HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_OBJECTS_MAY_08_2012_0610PM

#include <hpx/config/forceinline.hpp>
#include <hpx/util/move.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

namespace detail
{
    template <typename Action, int N>
    struct construct_continuation_thread_function_voidN;

    template <typename Action, int N>
    struct construct_continuation_thread_functionN;
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/construct_continuation_function_objects.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/construct_continuation_function_objects_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/construct_continuation_function_objects.hpp"))       \
    /**/

#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
    BOOST_PP_COMMA_IF(n)                                                      \
    util::get<n>(std::forward<Arguments>(data))                               \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_ACTION_DIRECT_ARGUMENT

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a continuation_thread_function will be registered as the thread
    /// function of a thread. It encapsulates the execution of the
    /// original function (given by \a func), and afterwards triggers all
    /// continuations using the result value obtained from the execution
    /// of the original thread function.
    template <typename Action>
    struct BOOST_PP_CAT(continuation_thread_function_void_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Func
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func && func
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";

                func(HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                cont->trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    /// The \a construct_continuation_thread_function is a helper function
    /// for constructing the wrapped thread function needed for
    /// continuation support
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, N>
    {
        template <typename Func, typename Arguments>
        static threads::thread_function_type
        call(continuation_type cont, Func && func,
            Arguments && args)
        {
            return util::bind(util::one_shot(
                BOOST_PP_CAT(continuation_thread_function_void_, N)<Action>()),
                cont, std::forward<Func>(func)
              BOOST_PP_COMMA_IF(N)
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct BOOST_PP_CAT(continuation_thread_function_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Func
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func && func
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";

                cont->trigger(std::forward<typename Action::result_type>(
                    func(HPX_ENUM_FORWARD_ARGS(N, Arg, arg))
                ));
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Action>
    struct construct_continuation_thread_functionN<Action, N>
    {
        template <typename Func, typename Arguments>
        static threads::thread_function_type
        call(continuation_type cont, Func && func,
            Arguments && args)
        {
            return util::bind(util::one_shot(
                BOOST_PP_CAT(continuation_thread_function_, N)<Action>()),
                cont, std::forward<Func>(func)
              BOOST_PP_COMMA_IF(N)
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }
    };
}

#undef N

#endif // !BOOST_PP_IS_ITERATING
