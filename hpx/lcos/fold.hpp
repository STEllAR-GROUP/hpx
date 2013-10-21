//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file fold.hpp

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
    /// \brief Perform a distributed fold operation
    ///
    /// The function hpx::lcos::fold performs a distributed folding
    /// operation over results returned from action invocations on a given set
    /// of global identifiers. The action can be either a plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param fold_op   [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the folding operation performed on its arguments.
    /// \param init      [in] The initial value to be used for the folding
    ///                  operation
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the action invocation.
    ///
    /// \note            The type of the initial value must be convertible to
    ///                  the result type returned from the invoked action.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall folding operation.
    ///
    template <typename Action, typename FoldOp, typename Init, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ...))>
    fold(
        std::vector<hpx::id_type> const & ids
      , FoldOp&& fold_op
      , Init&& init
      , ArgN argN, ...);

    /// \brief Perform a distributed folding operation
    ///
    /// The function hpx::lcos::fold_with_index performs a distributed folding
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
    /// \param fold_op [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the folding operation performed on its arguments.
    /// \param init      [in] The initial value to be used for the folding
    ///                  operation
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the action invocation.
    ///
    /// \note            The type of the initial value must be convertible to
    ///                  the result type returned from the invoked action.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall folding operation.
    ///
    template <typename Action, typename FoldOp, typename Init, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ..., std::size_t))>
    fold_with_index(
        std::vector<hpx::id_type> const & ids
      , FoldOp&& fold_op
      , Init&& init
      , ArgN argN, ...);

    /// \brief Perform a distributed inverse folding operation
    ///
    /// The function hpx::lcos::inverse_fold performs an inverse distributed folding
    /// operation over results returned from action invocations on a given set
    /// of global identifiers. The action can be either a plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param fold_op   [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the folding operation performed on its arguments.
    /// \param init      [in] The initial value to be used for the folding
    ///                  operation
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the action invocation.
    ///
    /// \note            The type of the initial value must be convertible to
    ///                  the result type returned from the invoked action.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall folding operation.
    ///
    template <typename Action, typename FoldOp, typename Init, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ...))>
    inverse_fold(
        std::vector<hpx::id_type> const & ids
      , FoldOp&& fold_op
      , Init&& init
      , ArgN argN, ...);

    /// \brief Perform a distributed inverse folding operation
    ///
    /// The function hpx::lcos::inverse_fold_with_index performs an inverse
    /// distributed folding operation over results returned from action
    /// invocations on a given set of global identifiers. The action can be
    /// either plain action (in
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
    /// \param fold_op [in] A binary function expecting two results as
    ///                  returned from the action invocations. The function
    ///                  (or function object) is expected to return the result
    ///                  of the folding operation performed on its arguments.
    /// \param init      [in] The initial value to be used for the folding
    ///                  operation
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the action invocation.
    ///
    /// \note            The type of the initial value must be convertible to
    ///                  the result type returned from the invoked action.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall folding operation.
    ///
    template <typename Action, typename FoldOp, typename Init, typename ArgN, ...>
    hpx::future<decltype(Action(hpx::id_type, ArgN, ..., std::size_t))>
    inverse_fold_with_index(
        std::vector<hpx::id_type> const & ids
      , FoldOp&& fold_op
      , Init&& init
      , ArgN argN, ...);
}}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_FOLD_SEP_29_2013_1442AM)
#define HPX_LCOS_FOLD_SEP_29_2013_1442AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_any.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/count_num_args.hpp>

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
        struct fold_with_index
        {
            typedef typename Action::arguments_type arguments_type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        struct fold_result
          : traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type>
        {};

        template <typename Action>
        struct fold_result<fold_with_index<Action> >
          : fold_result<Action>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, int N>
        struct make_fold_action_impl;

        template <typename Action>
        struct make_fold_action
          : make_fold_action_impl<
                Action
              , util::tuple_size<typename Action::arguments_type>::value
            >
        {};

        template <typename Action>
        struct make_fold_action<fold_with_index<Action> >
          : make_fold_action_impl<
                fold_with_index<Action>
              , util::tuple_size<typename Action::arguments_type>::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename FoldOp>
        struct perform_folding
        {
            perform_folding(FoldOp const& fold_op, Result const& init)
              : fold_op_(fold_op), init_(init)
            {}

            Result operator()(hpx::future<std::vector<hpx::future<Result> > >& r) const
            {
                std::vector<hpx::future<Result> > fres = boost::move(r.move());
                BOOST_ASSERT(!fres.empty());

                // we're at the beginning of the folding chain, incroporate the initial
                // value
                if (fres.size() == 1)
                    return fold_op_(init_, fres[0].move());

                // in the middle of the folding chain we simply apply the folding
                // operation to the two values we received.
                BOOST_ASSERT(fres.size() == 2);
                return fold_op_(fres[1].move(), fres[2].move());
            }

            FoldOp const& fold_op_;
            Result const& init_;
        };
    }
}}


//#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
//#  include <hpx/lcos/preprocessed/fold.hpp>
//#else
//
//#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
//#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/fold_" HPX_LIMIT_STR ".hpp")
//#endif

#define HPX_LCOS_FOLD_EXTRACT_ACTION_ARGUMENTS(Z, N, D)                       \
    typename boost::fusion::result_of::value_at_c<                            \
        typename Action::arguments_type, N                                    \
    >::type                                                                   \
/**/

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for fold
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/fold.hpp>))                  \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_FOLD_EXTRACT_ACTION_ARGUMENTS

//#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
//#  pragma wave option(output: null)
//#endif
//
//#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_ACTION_DECLARATION(...)                             \
    HPX_REGISTER_FOLD_ACTION_DECLARATION_(__VA_ARGS__)                        \
/**/
#define HPX_REGISTER_FOLD_ACTION_DECLARATION_(...)                            \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_FOLD_ACTION_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)  \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_ACTION_DECLARATION_2(Action, FoldOp)                \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , BOOST_PP_CAT(BOOST_PP_CAT(fold_, Action), FoldOp)                     \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_ACTION_DECLARATION_3(Action, FoldOp, Name)          \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , BOOST_PP_CAT(fold_, Name)                                             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_ACTION(...)                                         \
    HPX_REGISTER_FOLD_ACTION_(__VA_ARGS__)                                    \
/**/
#define HPX_REGISTER_FOLD_ACTION_(...)                                        \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_FOLD_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)              \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_ACTION_2(Action, FoldOp)                            \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , BOOST_PP_CAT(BOOST_PP_CAT(fold_, Action), FoldOp)                     \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_ACTION_3(Action, FoldOp, Name)                      \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , BOOST_PP_CAT(fold_, Name)                                             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION(...)                  \
    HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_(__VA_ARGS__)             \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_(...)                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_,                     \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_2(Action, FoldOp)     \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , BOOST_PP_CAT(BOOST_PP_CAT(fold_, Action), FoldOp)                     \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_3(Action, FoldOp, Name) \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , BOOST_PP_CAT(fold_, Name)                                             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION(...)                              \
    HPX_REGISTER_FOLD_WITH_INDEX_ACTION_(__VA_ARGS__)                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_(...)                             \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_FOLD_WITH_INDEX_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)   \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_2(Action, FoldOp)                 \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , BOOST_PP_CAT(BOOST_PP_CAT(fold_, Action), FoldOp)                     \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_3(Action, FoldOp, Name)           \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , BOOST_PP_CAT(fold_, Name)                                             \
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
        fold_invoke(Action /*act*/
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
        fold_invoke(fold_with_index<Action>
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
          , typename FoldOp
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        typename fold_result<Action>::type
        BOOST_PP_CAT(fold_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(FoldOp) fold_op
          , typename fold_result<Action>::type const& init
          BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
          , long global_idx
        )
        {
            typedef
                typename fold_result<Action>::type
                result_type;

            if(ids.empty()) return result_type();

            std::vector<hpx::future<result_type> > fold_futures;
            fold_futures.reserve(2);

            // first kick off the possibly remote operation
            id_type id_first = ids.front();
            if(ids.size() > 1)
            {
                std::vector<id_type> ids_next(ids.begin()+1, ids.end());

                typedef
                    typename detail::make_fold_action<Action>::
                        template fold_invoker<FoldOp>::type
                    fold_impl_action;

                if(!ids.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_next.front());
                    fold_futures.push_back(
                        hpx::async<fold_impl_action>(
                            id
                          , act
                          , boost::move(ids_next)
                          , fold_op
                          , init
                          BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                          , global_idx + 1
                        )
                    );
                }
            }

            // now perform the local operation
            fold_invoke(
                act
              , fold_futures
              , id_first
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
              , std::abs(global_idx)
            );

            return hpx::when_all(fold_futures).
                then(perform_folding<result_type, FoldOp>(fold_op, init)).
                move();
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename FoldOp
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
        >
        struct BOOST_PP_CAT(fold_invoker, N)
        {
            typedef
                typename fold_result<Action>::type
                result_type;

            static result_type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , FoldOp const& fold_op
              , result_type const& init
              BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a)
              , long global_idx
            )
            {
                return
                    BOOST_PP_CAT(fold_impl, N)(
                        act
                      , ids
                      , fold_op
                      , init
                      BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
                      , global_idx
                    );
            }
        };

        template <typename Action>
        struct make_fold_action_impl<Action, N>
        {
            typedef
                typename fold_result<Action>::type
                action_result;

            template <typename FoldOp>
            struct fold_invoker
            {
                typedef
                    typename util::decay<FoldOp>::type
                    fold_op_type;

                typedef BOOST_PP_CAT(fold_invoker, N)<
                        Action
                      , fold_op_type
                      BOOST_PP_ENUM_TRAILING(
                            N
                          , HPX_LCOS_FOLD_EXTRACT_ACTION_ARGUMENTS
                          , _
                        )
                    >
                    fold_invoker_type;

                typedef
                    typename HPX_MAKE_ACTION_TPL(fold_invoker_type::call)::type
                    type;
            };
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    fold(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        typedef
            typename detail::fold_result<Action>::type
            action_result;

        if (ids.empty())
        {
            return make_error_future<action_result>(
                HPX_GET_EXCEPTION(bad_parameter, "hpx::lcos::fold",
                    "empty list of targets for fold operation"));
        }

        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);

        typedef
            typename detail::make_fold_action<Action>::
                template fold_invoker<FoldOp>::type
            fold_impl_action;

        return
            hpx::async<fold_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
              , 0
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    fold(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return fold<Derived>(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Action
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    fold_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return fold<detail::fold_with_index<Action> >(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    fold_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return fold<detail::fold_with_index<Derived> >(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    inverse_fold(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        typedef
            typename detail::fold_result<Action>::type
            action_result;

        if (ids.empty()) 
        {
            return make_error_future<action_result>(
                HPX_GET_EXCEPTION(bad_parameter,
                    "hpx::lcos::inverse_fold",
                    "empty list of targets for fold operation"));
        }

        std::vector<id_type> inverted_ids;
        std::reverse_copy(ids.begin(), ids.end(), std::back_inserter(inverted_ids));

        hpx::id_type dest = hpx::get_colocation_id_sync(inverted_ids[0]);

        typedef
            typename detail::make_fold_action<Action>::
                template fold_invoker<FoldOp>::type
            fold_impl_action;

        return
            hpx::async<fold_impl_action>(
                dest
              , Action()
              , inverted_ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
              , -static_cast<long>(inverted_ids.size()-1)
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    inverse_fold(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return inverse_fold<Derived>(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Action
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    inverse_fold_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return inverse_fold<detail::fold_with_index<Action> >(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename FoldOp
      , typename Init
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    inverse_fold_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(FoldOp) fold_op
      , BOOST_FWD_REF(Init) init
      BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(N, A, const & a))
    {
        return inverse_fold<detail::fold_with_index<Derived> >(
                ids
              , boost::forward<FoldOp>(fold_op)
              , boost::forward<Init>(init)
              BOOST_PP_ENUM_TRAILING_PARAMS(N, a)
            );
    }
}}

#endif

#endif // DOXYGEN
