//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM)
#define HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM

#include <hpx/hpx_fwd.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/
#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

#define HPX_MOVE_ARGS(z, n, _)                                                \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::move(BOOST_PP_CAT(arg, n))                                 \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/construct_continuation_functions.hpp"))              \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_FWD_ARGS
#undef HPX_FORWARD_ARGS
#undef HPX_MOVE_ARGS

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()
#define M BOOST_PP_DEC(N)

    ///////////////////////////////////////////////////////////////////////////
    /// The \a continuation_thread_function will be registered as the thread
    /// function of a thread. It encapsulates the execution of the
    /// original function (given by \a func), and afterwards triggers all
    /// continuations using the result value obtained from the execution
    /// of the original thread function.
    struct BOOST_PP_CAT(continuation_thread_function_void_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Func
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont, Func const& func
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                            << detail::get_action_name<derived_type>()
                            << ") with continuation("
                            << cont->get_raw_gid()
                            << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                func(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _));
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
    template <typename Func
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_function_void(
        continuation_type cont, BOOST_FWD_REF(Func) func
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_function_void_, N)(),
            cont, boost::forward<Func>(func)
            BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct BOOST_PP_CAT(continuation_thread_function_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Func
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont, Func const& func
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                            << detail::get_action_name<derived_type>()
                            << ") with continuation("
                            << cont->get_raw_gid()
                            << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(boost::move(func(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _))));
            }
            catch (hpx::exception const&) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Func
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_function(
        continuation_type cont, BOOST_FWD_REF(Func) func
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_function_, N)(),
            cont, boost::forward<Func>(func)
            BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    // special version for member function pointer
    struct BOOST_PP_CAT(continuation_thread_object_function_void_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont,
            void (Object::* func)(BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)),
            Object* obj
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_raw_gid() << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                (obj->*func)(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)) const,
            Component* obj
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_raw_gid() << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                (obj->*func)(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Object
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(BOOST_PP_ENUM_PARAMS(N, FArg)), Object* obj
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_void_, N)(),
            cont, func, obj BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    template <typename Object
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(BOOST_PP_ENUM_PARAMS(N, FArg)) const,
        Component* obj
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_void_, N)(),
            cont, func, obj BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct BOOST_PP_CAT(continuation_thread_object_function_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)),
            Component* obj
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_raw_gid() << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(
                    boost::move((obj->*func)(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _))));
            }
            catch (hpx::exception const&) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)) const,
            Component* obj
            BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_raw_gid() << ")";
                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(
                    boost::move((obj->*func)(BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _))));
            }
            catch (hpx::exception const&) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Object
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(BOOST_PP_ENUM_PARAMS(N, FArg)), Component* obj
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_, N)(),
            cont, func, obj BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    template <typename Object
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(BOOST_PP_ENUM_PARAMS(N, FArg)) const,
        Component* obj
        BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_, N)(),
            cont, func, obj BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

#undef M
#undef N

#endif // !BOOST_PP_IS_ITERATING
