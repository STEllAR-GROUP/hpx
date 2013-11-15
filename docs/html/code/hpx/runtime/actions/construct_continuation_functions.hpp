//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM)
#define HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM

#include <hpx/config/forceinline.hpp>
#include <hpx/util/move.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/construct_continuation_functions.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/construct_continuation_functions_" HPX_LIMIT_STR ".hpp")
#endif

#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
    BOOST_PP_COMMA_IF(n)                                                      \
    util::get<n>(boost::forward<Arguments_>(data))                            \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/construct_continuation_functions.hpp"))              \
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

    ///////////////////////////////////////////////////////////////////////////
    // special version for member function pointer
    struct BOOST_PP_CAT(continuation_thread_object_function_void_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)),
            Object* obj
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                (obj->*func)(HPX_ENUM_MOVE_ARGS(N, arg));
                cont->trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)) const,
            Component* obj
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                (obj->*func)(HPX_ENUM_MOVE_ARGS(N, arg));
                cont->trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Object, typename Arguments_
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(BOOST_PP_ENUM_PARAMS(N, FArg)), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_void_, N)(),
            cont, func, obj
          BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
    }

    template <typename Object, typename Arguments_
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(BOOST_PP_ENUM_PARAMS(N, FArg)) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_void_, N)(),
            cont, func, obj
          BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct BOOST_PP_CAT(continuation_thread_object_function_, N)
    {
        typedef threads::thread_state_enum result_type;

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)),
            Component* obj
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(boost::forward<Result>(
                    (obj->*func)(HPX_ENUM_MOVE_ARGS(N, arg))
                ));
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }

        template <typename Object
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
            BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, FArg, arg)) const,
            Component* obj
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";

                // The arguments are moved here. This function is called from a
                // bound functor. In order to do true perfect forwarding in an
                // asynchronous operation. These bound variables must be moved
                // out of the bound object.
                cont->trigger(boost::forward<Result>(
                    (obj->*func)(HPX_ENUM_MOVE_ARGS(N, arg))
                ));
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    template <typename Object, typename Arguments_
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(BOOST_PP_ENUM_PARAMS(N, FArg)), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_, N)(),
            cont, func, obj
          BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
    }

    template <typename Object, typename Arguments_
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename FArg)>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(BOOST_PP_ENUM_PARAMS(N, FArg)) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            BOOST_PP_CAT(continuation_thread_object_function_, N)(),
            cont, func, obj
          BOOST_PP_COMMA_IF(N)
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, args));
    }

#undef N

#endif // !BOOST_PP_IS_ITERATING
