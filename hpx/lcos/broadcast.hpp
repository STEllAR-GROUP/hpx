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
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <vector>

namespace hpx { namespace lcos {
    namespace detail
    {
        template <typename Action>
        struct broadcast_with_index
        {
            typedef typename Action::arguments_type arguments_type;
        };

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

        template <typename Action>
        struct broadcast_result<broadcast_with_index<Action> >
          : broadcast_result<Action>
        {};

        template <typename Action, int N>
        struct make_broadcast_action_impl;

        template <typename Action>
        struct make_broadcast_action
          : make_broadcast_action_impl<
                Action
              , boost::fusion::result_of::size<
                    typename Action::arguments_type
                >::value
            >
        {};

        template <typename Action>
        struct make_broadcast_action<broadcast_with_index<Action> >
          : make_broadcast_action_impl<
                broadcast_with_index<Action>
              , boost::fusion::result_of::size<
                    typename Action::arguments_type
                >::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        inline void return_void(hpx::future<std::vector<hpx::future<void> > >)
        {
            // todo: verify validity of all futures in the vector
        }

        template <
            typename Result
        >
        std::vector<Result>
        wrap_into_vector(hpx::future<Result> r)
        {
            return std::vector<Result>(1, r.move());
        }

        template <
            typename Result
        >
        std::vector<Result>
        return_result_type(hpx::future<std::vector<hpx::future<std::vector<Result> > > > r)
        {
            std::vector<Result> res;
            std::vector<hpx::future<std::vector<Result> > > fres = boost::move(r.move());

            BOOST_FOREACH(hpx::future<std::vector<Result> >& f, fres)
            {
                std::vector<Result> t = boost::move(f.move());
                res.reserve(res.capacity() + t.size());
                boost::move(t.begin(), t.end(), std::back_inserter(res));
            }

            return boost::move(res);
        }
    }
}}

/*
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/broadcast.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/broadcast_" HPX_LIMIT_STR ".hpp")
#endif
*/

#define HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS(Z, N, D)                  \
    typename boost::fusion::result_of::value_at_c<                            \
        typename Action::arguments_type, N                                    \
    >::type                                                                   \
/**/

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for broadcast
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/broadcast.hpp>))             \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS

/*
#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
*/

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION(Action)                     \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Action)                                      \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_ACTION(Action)                                 \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Action)                                      \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(Action, Name)             \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_2(Action, Name)                         \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION(Action)          \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_, Action)                                      \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION(Action)                      \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_, Action)                                      \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_2(Action, Name)  \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_2(Action, Name)              \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
/**/

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                )
            );
        }

        template <
            typename Action
          , typename Futures
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename Futures
          , typename Cont
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(Action act, Futures& futures, Cont cont, hpx::id_type id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                ).then(cont)
            );
        }

        template <
            typename Action
          , typename Futures
          , typename Cont
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, Cont cont, hpx::id_type id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx
                ).then(cont)
            );
        }

        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        //hpx::future<void>
        void
        BOOST_PP_CAT(broadcast_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;// hpx::lcos::make_ready_future();

            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);

            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , global_idx
            );

            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());

                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;

                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }

                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }

            //return hpx::when_all(broadcast_futures).then(&return_void);
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }

        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        //hpx::future<typename broadcast_result<Action>::type>
        typename broadcast_result<Action>::type
        BOOST_PP_CAT(broadcast_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;

            //if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            if(ids.empty()) return result_type();

            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);

            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , global_idx
            );

            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());

                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;

                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }

                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }

            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IsVoid
        >
        struct BOOST_PP_CAT(broadcast_invoker, N)
        {
            //static hpx::future<typename broadcast_result<Action>::type>
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    BOOST_PP_CAT(broadcast_impl, N)(
                        act
                      , ids
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , global_idx
                      , IsVoid()
                    );
            }
        };

        template <typename Action>
        struct make_broadcast_action_impl<Action, N>
        {
            typedef
                typename broadcast_result<Action>::action_result
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

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);

        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;

        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , 0
              , typename boost::is_same<void, action_result>::type()
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        return broadcast<Derived>(
                ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
            );
    }
    
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
                ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
            );
    }
}}

#endif
