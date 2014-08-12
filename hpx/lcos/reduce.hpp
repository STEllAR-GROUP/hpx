//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file reduce.hpp

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
    /// \brief Perform a distributed reduction operation
    ///
    /// The function hpx::lcos::reduce performs a distributed reduction
    /// operation over results returned from action invocations on a given set
    /// of global identifiers. The action can be either a plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param reduce_op [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the reduction operation performed on its arguments.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall reduction operation.
    ///
    template <typename Action, typename ReduceOp, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ...))>
    reduce(
        std::vector<hpx::id_type> const & ids
      , ReduceOp&& reduce_op
      , ArgN argN, ...);

    /// \brief Perform a distributed reduction operation
    ///
    /// The function hpx::lcos::reduce_with_index performs a distributed reduction
    /// operation over results returned from action invocations on a given set
    /// of global identifiers. The action can be either plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// The function passes the index of the global identifier in the given
    /// list of identifiers as the last argument to the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param reduce_op [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the reduction operation performed on its arguments.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall reduction operation.
    ///
    template <typename Action, typename ReduceOp, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ..., std::size_t))>
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , ReduceOp&& reduce_op
      , ArgN argN, ...);
}}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_REDUCE_SEP_28_2013_1105AM)
#define HPX_LCOS_REDUCE_SEP_28_2013_1105AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/calculate_fanout.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/cat.hpp>

#if !defined(HPX_REDUCE_FANOUT)
#define HPX_REDUCE_FANOUT 16
#endif

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
              , util::tuple_size<typename Action::arguments_type>::value
            >
        {};

        template <typename Action>
        struct make_reduce_action<reduce_with_index<Action> >
          : make_reduce_action_impl<
                reduce_with_index<Action>
              , util::tuple_size<typename Action::arguments_type>::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename ReduceOp>
        struct perform_reduction
        {
            perform_reduction(ReduceOp const& reduce_op)
              : reduce_op_(reduce_op)
            {}

            Result operator()(
                hpx::future<std::vector<hpx::future<Result> > > r) const
            {
                std::vector<hpx::future<Result> > fres = std::move(r.get());

                HPX_ASSERT(!fres.empty());

                if (fres.size() == 1)
                    return fres[0].get();

                Result res = reduce_op_(fres[0].get(), fres[1].get());
                for (std::size_t i = 2; i != fres.size(); ++i)
                    res = reduce_op_(res, fres[i].get());

                return std::move(res);
            }

            ReduceOp const& reduce_op_;
        };
    }
}}


#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/reduce.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/reduce_" HPX_LIMIT_STR ".hpp")
#endif

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

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_REDUCE_ACTION_DECLARATION(...)                           \
    HPX_REGISTER_REDUCE_ACTION_DECLARATION_(__VA_ARGS__)                      \
/**/
#define HPX_REGISTER_REDUCE_ACTION_DECLARATION_(...)                          \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_REDUCE_ACTION_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_REDUCE_ACTION_DECLARATION_2(Action, ReduceOp)            \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            reduce_invoker<ReduceOp>::type                                    \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/
#define HPX_REGISTER_REDUCE_ACTION_DECLARATION_3(Action, ReduceOp, Name)      \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            reduce_invoker<ReduceOp>::type                                    \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_REDUCE_ACTION(...)                                       \
    HPX_REGISTER_REDUCE_ACTION_(__VA_ARGS__)                                  \
/**/
#define HPX_REGISTER_REDUCE_ACTION_(...)                                      \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_REDUCE_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)            \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_REDUCE_ACTION_2(Action, ReduceOp)                        \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            reduce_invoker<ReduceOp>::type                                    \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/
#define HPX_REGISTER_REDUCE_ACTION_3(Action, ReduceOp, Name)                  \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<Action>::                     \
            reduce_invoker<ReduceOp>::type                                    \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION(...)                \
    HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_(__VA_ARGS__)           \
/**/
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_(...)               \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_,                   \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_2(Action, ReduceOp) \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::reduce_invoker<ReduceOp>::type                                     \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION_3(Action, ReduceOp, Name) \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::reduce_invoker<ReduceOp>::type                                     \
      , BOOST_PP_CAT(reduce_, Name)                                           \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION(...)                            \
    HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_(__VA_ARGS__)                       \
/**/
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_(...)                           \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__) \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_2(Action, ReduceOp)             \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::reduce_invoker<ReduceOp>::type                                     \
      , BOOST_PP_CAT(BOOST_PP_CAT(reduce_, Action), ReduceOp)                 \
    )                                                                         \
/**/
#define HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_3(Action, ReduceOp, Name)       \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_reduce_action<                              \
            ::hpx::lcos::detail::reduce_with_index<Action>                    \
        >::reduce_invoker<ReduceOp>::type                                     \
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
          , ReduceOp && reduce_op
          BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;

            if(ids.empty()) return result_type();

            std::size_t const local_fanout = HPX_REDUCE_FANOUT;
            std::size_t local_size = (std::min)(ids.size(), local_fanout);
            std::size_t fanout = util::calculate_fanout(ids.size(), local_fanout);

            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(local_size + (ids.size()/fanout) + 1);
            for(std::size_t i = 0; i != local_size; ++i)
            {
                reduce_invoke(
                    act
                  , reduce_futures
                  , ids[i]
                  BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                  , global_idx + i
                );
            }

            if(ids.size() > local_fanout)
            {
                std::size_t applied = local_fanout;
                std::vector<hpx::id_type>::const_iterator it =
                    ids.begin() + local_fanout;

                typedef
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;

                while(it != ids.end())
                {
                    HPX_ASSERT(ids.size() >= applied);

                    std::size_t next_fan = (std::min)(fanout, ids.size() - applied);
                    std::vector<hpx::id_type> ids_next(it, it + next_fan);

                    hpx::id_type id(ids_next[0]);
                    reduce_futures.push_back(
                        hpx::async_colocated<reduce_impl_action>(
                            id
                          , act
                          , std::move(ids_next)
                          , reduce_op
                          BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                          , global_idx + applied
                        )
                    );

                    applied += next_fan;
                    it += next_fan;
                }
            }

            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                get();
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
      , ReduceOp && reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;

        typedef
            typename detail::reduce_result<Action>::type
            action_result;

        return
            hpx::async_colocated<reduce_impl_action>(
                ids[0]
              , Action()
              , ids
              , std::forward<ReduceOp>(reduce_op)
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
      , ReduceOp && reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<Derived>(
                ids
              , std::forward<ReduceOp>(reduce_op)
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
      , ReduceOp && reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , std::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
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
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , ReduceOp && reduce_op
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , std::forward<ReduceOp>(reduce_op)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }
}}

#endif

#endif // DOXYGEN
