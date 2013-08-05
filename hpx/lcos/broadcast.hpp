//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_BROADCAST_HPP
#define HPX_LCOS_BROADCAST_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_any.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <vector>

namespace hpx { namespace lcos {
    namespace impl
    {
        template <typename Action>
        struct broadcast_result
        {
            typedef
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<
                        Action
                    >::remote_result_type
                >::type
                action_result;
            typedef
                typename boost::mpl::if_<
                    boost::is_same<void, action_result>
                  , void
                  , std::vector<action_result>
                >::type
                type;
        };

        template <typename Action, int N>
        struct make_broadcast_action_impl;

        template <
            typename Action
        >
        struct make_broadcast_action
          : make_broadcast_action_impl<
                Action
              , boost::fusion::result_of::size<
                    typename Action::arguments_type
                >::value
            >
        {};
    }
}}

/**
 * FIXME: generate partially preprocessed headers
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/broadcast.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/broadcast_" HPX_LIMIT_STR ".hpp")
#endif
*/

#define HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS(Z, N, D)                    \
    typename boost::fusion::result_of::value_at_c<                              \
        typename Action::arguments_type, N                                      \
    >::type                                                                     \
/**/

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for broadcast
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/broadcast.hpp>))               \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS

/*
#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
*/

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION(Action)                       \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Action)                                        \
    )                                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION(Action)                                   \
    HPX_REGISTER_PLAIN_ACTION(                                                  \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Action)                                        \
    )                                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(Action, Name)               \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Name)                                          \
    )                                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_2(Action, Name)                           \
    HPX_REGISTER_PLAIN_ACTION(                                                  \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Name)                                          \
    )                                                                           \
/**/

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos {
    namespace impl
    {

        template <
            typename Action
            BOOST_PP_COMMA_IF(N)
            BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        typename broadcast_result<Action>::type
        BOOST_PP_CAT(broadcast_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , boost::mpl::true_
        )
        {
            if(ids.size() == 0) return;

            hpx::id_type this_id = ids[0];

            if(ids.size() == 1)
            {
                act(
                    ids[0]
                    BOOST_PP_COMMA_IF(N)
                    BOOST_PP_ENUM_PARAMS(N, a)
                );
                return;
            }

            if(ids.size() == 2)
            {
                hpx::future<void> f = hpx::async(
                        act
                      , ids[1]
                      BOOST_PP_COMMA_IF(N)
                      BOOST_PP_ENUM_PARAMS(N, a)
                    );
                act(
                    ids[0]
                  BOOST_PP_COMMA_IF(N)
                  BOOST_PP_ENUM_PARAMS(N, a)
                );
                hpx::wait(f);
                return;
            }

            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(2);
            std::size_t half = (ids.size() / 2) + 1;
            if(half == 1) half = 2;
            std::vector<hpx::id_type>
                ids_first(ids.begin() + 1, ids.begin() + half);
            std::vector<hpx::id_type>
                ids_second(ids.begin() + half, ids.end());

            typedef
                typename impl::make_broadcast_action<
                    Action
                >::type
                broadcast_impl_action;

            hpx::id_type id = hpx::naming::get_locality_from_id(ids_first[0]);
            broadcast_futures.push_back(
                hpx::async<broadcast_impl_action>(
                    id
                  , act
                  , boost::move(ids_first)
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , boost::integral_constant<bool, true>()
                )
            );

            if(ids_second.size() > 0)
            {
                hpx::id_type id = hpx::naming::get_locality_from_id(ids_second[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_second)
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , boost::integral_constant<bool, true>()
                    )
                );
            }

            act(
                ids[0]
              BOOST_PP_COMMA_IF(N)
              BOOST_PP_ENUM_PARAMS(N, a)
            );

            while(!broadcast_futures.empty())
            {
                HPX_STD_TUPLE<int, hpx::future<void> >
                    f_res = hpx::wait_any(broadcast_futures);
                int part = HPX_STD_GET(0, f_res);
                broadcast_futures.erase(broadcast_futures.begin() + part);
            }
        }

        template <
            typename Action
            BOOST_PP_COMMA_IF(N)
            BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        typename broadcast_result<Action>::type
        BOOST_PP_CAT(broadcast_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::type
                result_type;

            if(ids.size() == 0) return result_type();

            hpx::id_type this_id = ids[0];
            
            result_type res(ids.size());
            if(ids.size() == 1)
            {
                res[0]
                    = boost::move(
                        act(
                            ids[0]
                            BOOST_PP_COMMA_IF(N)
                            BOOST_PP_ENUM_PARAMS(N, a)
                        )
                    );
                return boost::move(res);
            }

            if(ids.size() == 2)
            {
                hpx::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<
                            Action
                        >::remote_result_type
                    >::type
                > f = hpx::async(
                        act
                      , ids[1]
                      BOOST_PP_COMMA_IF(N)
                      BOOST_PP_ENUM_PARAMS(N, a)
                    );
                res[0]
                    = boost::move(
                        act(
                            ids[0]
                          BOOST_PP_COMMA_IF(N)
                          BOOST_PP_ENUM_PARAMS(N, a)
                        )
                    );
                res[1] = f.move();
                return boost::move(res);
            }

            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(2);
            std::size_t half = (ids.size() / 2) + 1;
            if(half == 1) half = 2;
            std::vector<hpx::id_type>
                ids_first(ids.begin() + 1, ids.begin() + half);
            std::vector<hpx::id_type>
                ids_second(ids.begin() + half, ids.end());

            typedef
                typename impl::make_broadcast_action<
                    Action
                >::type
                broadcast_impl_action;

            hpx::id_type id = hpx::naming::get_locality_from_id(ids_first[0]);
            broadcast_futures.push_back(
                hpx::async<broadcast_impl_action>(
                    id
                  , act
                  , boost::move(ids_first)
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , boost::integral_constant<bool, false>()
                )
            );

            if(ids_second.size() > 0)
            {
                hpx::id_type id = hpx::naming::get_locality_from_id(ids_second[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_second)
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , boost::integral_constant<bool, false>()
                    )
                );
            }

            res[0]
                = boost::move(
                    act(
                        ids[0]
                      BOOST_PP_COMMA_IF(N)
                      BOOST_PP_ENUM_PARAMS(N, a)
                    )
                );

            int part_finished = -1;
            while(!broadcast_futures.empty())
            {
                HPX_STD_TUPLE<int, hpx::future<result_type> >
                    f_res = hpx::wait_any(broadcast_futures);
                int part = HPX_STD_GET(0, f_res);
                result_type tmp(boost::move(HPX_STD_GET(1, f_res).move()));
                broadcast_futures.erase(broadcast_futures.begin() + part);
                if(((part_finished == -1) && (part == 0)) || (part_finished == 1))
                {
                    std::copy(tmp.begin(), tmp.end(), res.begin() + 1);
                    part_finished = 0;
                    continue;
                }
                if(((part_finished == -1) && (part == 1)) || (part_finished == 0))
                {
                    std::copy(tmp.begin(), tmp.end(), res.begin() + half);
                    part_finished = 1;
                    continue;
                }
            }

            return boost::move(res);
        }


        template <
            typename Action
            BOOST_PP_COMMA_IF(N)
            BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IsVoid
        >
        struct BOOST_PP_CAT(broadcast_invoker, N)
        {
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
              , IsVoid
            )
            {
                return
                    BOOST_PP_CAT(broadcast_impl, N)(
                        act
                      , ids
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , IsVoid()
                    );
            }
        };

        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, N>
        {
            typedef
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<Action>::remote_result_type
                >::type
                action_result;

            typedef BOOST_PP_CAT(broadcast_invoker, N)<
                        Action
                        BOOST_PP_COMMA_IF(N)
                        BOOST_PP_ENUM(
                            N
                          , HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS
                          , _
                        )
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;

            typedef
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
                type;
        };
    }

    template <typename Action
        BOOST_PP_COMMA_IF(N)
        BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    hpx::future<
        typename impl::broadcast_result<Action>::type
    > broadcast(std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        hpx::id_type dest = hpx::naming::get_locality_from_id(ids[0]);

        typedef
            typename impl::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , boost::move(ids)
                BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , typename boost::is_same<void, action_result>::type()
            );
    }

}}


#endif
