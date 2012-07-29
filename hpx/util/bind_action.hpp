//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_BIND_ACTION_HPP
#define HPX_UTIL_BIND_ACTION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/include/async.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_BIND_EVAL(Z, N, D)                                           \
    hpx::util::detail::eval(env, BOOST_PP_CAT(arg, N))                        \
/**/

#define HPX_UTIL_BIND_REMOVE_REFERENCE(Z, N, D)                               \
    typename boost::remove_const<                                             \
        typename detail::remove_reference<BOOST_PP_CAT(D, N)>::type>::type    \
/**/

#define HPX_UTIL_BIND_REFERENCE(Z, N, D)                                      \
    typename detail::env_value_type<BOOST_FWD_REF(BOOST_PP_CAT(D, N))>::type  \
/**/

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/util/preprocessed/bind_action.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/bind_action_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                       \
          , <hpx/util/bind_action.hpp>                                        \
        )                                                                     \
    )                                                                         \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

#undef HPX_UTIL_BIND_EVAL
#undef HPX_UTIL_BIND_REMOVE_REFERENCE
#undef HPX_UTIL_BIND_REFERENCE

namespace boost { namespace serialization
{
    // serialization of placeholders is trivial, just provide empty functions
    template <int N>
    void serialize(hpx::util::portable_binary_iarchive&,
        hpx::util::placeholders::arg<N>&, unsigned int const)
    {}
    template <int N>
    void serialize(hpx::util::portable_binary_oarchive&,
        hpx::util::placeholders::arg<N>&, unsigned int const)
    {}
}}

#endif

#else  // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_FRAME_ITERATION(1)
#define NN N

#define HPX_UTIL_BIND_INIT_MEMBER(Z, N, D)                                    \
    BOOST_PP_CAT(arg, N)(boost::forward<BOOST_PP_CAT(A, N)>(BOOST_PP_CAT(a, N)))\
/**/
#define HPX_UTIL_BIND_MEMBER(Z, N, D)                                         \
    BOOST_PP_CAT(Arg, N) BOOST_PP_CAT(arg, N);                                \
/**/

#define HPX_UTIL_BIND_INIT_COPY_MEMBER(Z, N, D)                               \
    BOOST_PP_CAT(arg, N)(other.BOOST_PP_CAT(arg, N))                          \
/**/

#define HPX_UTIL_BIND_INIT_MOVE_MEMBER(Z, N, D)                               \
    BOOST_PP_CAT(arg, N)(boost::move(other.BOOST_PP_CAT(arg, N)))             \
/**/

#define HPX_UTIL_BIND_ASSIGN_MEMBER(Z, N, D)                                  \
    BOOST_PP_CAT(arg, N) = other.BOOST_PP_CAT(arg, N);                        \
/**/

#define HPX_UTIL_BIND_MOVE_MEMBER(Z, N, D)                                    \
    BOOST_PP_CAT(arg, N) = boost::move(other.BOOST_PP_CAT(arg, N));           \
/**/

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // actions
    namespace detail
    {
        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound_action, N)
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;

            // default constructor is needed for serialization
            BOOST_PP_CAT(bound_action, N)()
            {}

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_action, N)(
                HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_action, N)(
                    BOOST_PP_CAT(bound_action, N) const & other)
                : BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_action, N)(BOOST_RV_REF(
                    BOOST_PP_CAT(bound_action, N)) other)
                : BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_action, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound_action, N)) other)
            {
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound_action, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound_action, N)) other)
            {
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            // apply() invokes the embedded action fully asynchronously
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

#define HPX_UTIL_BIND_ACTION_APPLY(Z, N, D)                                   \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                            \
    bool apply(HPX_ENUM_FWD_ARGS(N, A, a))                                     \
    {                                                                         \
        typedef                                                               \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                  \
            >                                                                 \
            env_type;                                                         \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                          \
        return hpx::apply<Action>(                                            \
            hpx::util::detail::eval(env, arg0)                                \
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                 \
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));            \
    }                                                                         \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                            \
    bool apply(HPX_ENUM_FWD_ARGS(N, A, a)) const                               \
    {                                                                         \
        typedef                                                               \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                  \
            >                                                                 \
            env_type;                                                         \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                          \
        return hpx::apply<Action>(                                            \
            hpx::util::detail::eval(env, arg0)                                \
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                 \
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));            \
    }                                                                         \
/**/
            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_ACTION_APPLY, _
            )
#undef HPX_UTIL_BIND_ACTION_APPLY

            // async() invokes the embedded action asynchronously and returns
            // a future representing the result of the embedded operation
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

#define HPX_UTIL_BIND_ACTION_ASYNC(Z, N, D)                                   \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                            \
    hpx::lcos::future<result_type>                                            \
    async(HPX_ENUM_FWD_ARGS(N, A, a))                                          \
    {                                                                         \
        typedef                                                               \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                  \
            >                                                                 \
            env_type;                                                         \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                          \
        return hpx::async<Action>(                                            \
            hpx::util::detail::eval(env, arg0)                                \
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                 \
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));            \
    }                                                                         \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                            \
    hpx::lcos::future<result_type>                                            \
    async(HPX_ENUM_FWD_ARGS(N, A, a)) const                                    \
    {                                                                         \
        typedef                                                               \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                  \
            >                                                                 \
            env_type;                                                         \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                          \
        return hpx::async<Action>(                                            \
            hpx::util::detail::eval(env, arg0)                                \
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                 \
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));            \
    }                                                                         \
/**/
            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_ACTION_ASYNC, _
            )
#undef HPX_UTIL_BIND_ACTION_ASYNC

            // The operator()() invokes the embedded action synchronously.
            result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _)).get();
            }

            result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _)).get();
            }

#define BOOST_PP_ITERATION_PARAMS_2                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                                  \
          , <hpx/util/detail/bind_action_functor_operator.hpp>                  \
        )                                                                       \
    )                                                                           \
 /**/
#include BOOST_PP_ITERATE()

            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)
        };

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Env
          , typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        typename BOOST_PP_CAT(detail::bound_action, N)<
                Action BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
        >::result_type
        eval(
            Env & env
          , BOOST_PP_CAT(detail::bound_action, N)<
                Action
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    BOOST_PP_CAT(detail::bound_action, N)<
                        Action
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
                    >
                >(f)(
                    env
                 );
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound_action, N)<
        Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_action, N)<
                Action
              BOOST_PP_COMMA_IF(N)
                  BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            > (HPX_ENUM_FORWARD_ARGS(N, A, a));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound_action, N)<
        Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_action, N)<
                Derived
              BOOST_PP_COMMA_IF(N)
                  BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            > (HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the bound object, just serialize members
#define HPX_UTIL_BIND_SERIALIZE_MEMBER(Z, N, _) ar & BOOST_PP_CAT(bound.arg, N);

    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
            Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
            Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

#undef HPX_UTIL_BIND_SERIALIZE_MEMBER
}}

#undef HPX_UTIL_BIND_ASSIGN_MEMBER
#undef HPX_UTIL_BIND_INIT_MOVE_MEMBER
#undef HPX_UTIL_BIND_INIT_COPY_MEMBER
#undef HPX_UTIL_BIND_MEMBER
#undef HPX_UTIL_BIND_INIT_MEMBER

#undef NN
#undef N

#endif
