//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_REDUCE_SEP_28_2013_1105AM)
#define HPX_LCOS_REDUCE_SEP_28_2013_1105AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_any.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/decay.hpp>

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/assert.hpp>

namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        struct reduce_with_index
        {
            typedef typename Action::arguments_type arguments_type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        struct reduce_result
          : traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type>
        {};

        template <typename Action>
        struct reduce_result<reduce_with_index<Action> >
          : reduce_result<Action>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, int N>
        struct make_reduce_action_impl;

        template <typename Action>
        struct make_reduce_action
          : make_reduce_action_impl<
                Action
              , boost::fusion::result_of::size<
                    typename Action::arguments_type
                >::value
            >
        {};

        template <typename Action>
        struct make_reduce_action<reduce_with_index<Action> >
          : make_reduce_action_impl<
                reduce_with_index<Action>
              , boost::fusion::result_of::size<
                    typename Action::arguments_type
                >::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename ReduceOp>
        struct perform_reduction
        {
            perform_reduction(ReduceOp const& reduce_op)
              : reduce_op_(reduce_op)
            {}

            Result operator()(hpx::future<std::vector<hpx::future<Result> > >& r) const
            {
                std::vector<hpx::future<Result> > fres = boost::move(r.move());

                BOOST_ASSERT(!fres.empty());

                if (fres.size() == 1)
                    return fres[0].move();

                Result res = reduce_op_(fres[0].move(), fres[1].move());
                for (std::size_t i = 2; i != fres.size(); ++i)
                    res = reduce_op_(res, fres[i].get());

                return boost::move(res);
            }

            ReduceOp const& reduce_op_;
        };
    }
}}

/*
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/reduce.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/reduce_" HPX_LIMIT_STR ".hpp")
#endif
*/

#define HPX_LCOS_REDUCE_EXTRACT_ACTION_ARGUMENTS(Z, N, D)                     \
    typename boost::fusion::result_of::value_at_c<                            \
        typename Action::arguments_type, N                                    \
    >::type                                                                   \
/**/

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for reduce
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/reduce.hpp>))                \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_REDUCE_EXTRACT_ACTION_ARGUMENTS

/*
#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
*/

#define HPX_REGISTER_REDUCE_ACTION_DECLARATION(Action, ReduceOp)              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            template reduce_invoker<ReduceOp>::type                           \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_ACTION(Action, ReduceOp)                          \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            template reduce_invoker<ReduceOp>::type                           \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_ACTION_DECLARATION_2(Action, ReduceOp, Name)      \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            template reduce_invoker<ReduceOp>::type                           \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_ACTION_2(Action, ReduceOp, Name)                  \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            template reduce_invoker<ReduceOp>::type                           \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION(Action, ReduceOp)   \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::template reduce_invoker<ReduceOp>::type                            \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION(Action, ReduceOp)               \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::template reduce_invoker<ReduceOp>::type                            \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_2(Action, ReduceOp, Name) \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::template reduce_invoker<ReduceOp>::type                            \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_2(Action, ReduceOp, Name)       \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::template reduce_invoker<ReduceOp>::type                            \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        void
        reduce_invoke(Action /*act*/
          , Futures& futures
          , hpx::id_type const& id
          BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                )
            );
        }

        template <
            typename Action
          , typename Futures
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                  , global_idx
                )
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename ReduceOp
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        typename reduce_result<Action>::type
        BOOST_PP_CAT(reduce_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;

            if(ids.empty()) return result_type();

            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);

            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;

                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                          , global_idx + 1
                        )
                    );
                }

                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                          , global_idx + half
                        )
                    );
                }
            }

            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename ReduceOp
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        struct BOOST_PP_CAT(reduce_invoker, N)
        {
            //static hpx::future<typename reduce_result<Action>::type>
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
              , std::size_t global_idx
            )
            {
                return
                    BOOST_PP_CAT(reduce_impl, N)(
                        act
                      , ids
                      , reduce_op
                      BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                      , global_idx
                    );
            }
        };

        template <typename Action>
        struct make_reduce_action_impl<Action, N>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;

            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;

                typedef BOOST_PP_CAT(reduce_invoker, N)<
                        Action
                      , reduce_op_type
                      BOOST_PP_ENUM_TRAILING(
                            N
                          , HPX_LCOS_REDUCE_EXTRACT_ACTION_ARGUMENTS
                          , _
                        )
                    >
                    reduce_invoker_type;

                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename ReduceOp
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);

        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;

        typedef
            typename detail::reduce_result<Action>::type
            action_result;

        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
              , 0
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Action
      , typename ReduceOp
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }
}}

#endif
