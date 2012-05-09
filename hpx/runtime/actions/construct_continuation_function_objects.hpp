//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_OBJECTS_MAY_08_2012_0610PM)
#define HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_OBJECTS_MAY_08_2012_0610PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

#include <boost/fusion/include/size.hpp>
#include <boost/type_traits/remove_reference.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace detail
{
    template <typename Action, int N>
    struct construct_continuation_thread_function_voidN;

    template <typename Action, int N>
    struct construct_continuation_thread_functionN;

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/construct_continuation_function_objects.hpp"))       \
    /**/

#include BOOST_PP_ITERATE()

}

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()
#define M BOOST_PP_DEC(N)

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/
#define HPX_MOVE_ARGS(z, n, _)                                                \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::move(BOOST_PP_CAT(arg, n))                                 \
    /**/
#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
        BOOST_PP_COMMA_IF(n) boost::move(util::get_argument_from_pack<n>(data)) \
    /**/

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
        result_type operator()(continuation_type cont, Func const& func
          BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                func(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, args));
                cont->trigger();
            }
            catch (hpx::exception const&) {
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                BOOST_PP_CAT(continuation_thread_function_void_, N)<Action>(),
                cont, boost::forward<Func>(func)
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
        result_type operator()(continuation_type cont, Func const& func
          BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(boost::move(
                    func(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _))
                ));
            }
            catch (hpx::exception const&) {
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                BOOST_PP_CAT(continuation_thread_function_, N)<Action>(),
                cont, boost::forward<Func>(func)
              BOOST_PP_COMMA_IF(N)
                    BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
        }
    };

#undef HPX_FWD_ARGS
#undef HPX_MOVE_ARGS
#undef HPX_ACTION_DIRECT_ARGUMENT
#undef M
#undef N

#endif // !BOOST_PP_IS_ITERATING
