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

#if !defined(HPX_LCOS_FOLD_SEP_29_2013_1442AM)
#define HPX_LCOS_FOLD_SEP_29_2013_1442AM

#include <hpx/config.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/expand.hpp>
#include <hpx/util/detail/pp/nargs.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <utility>
#include <vector>

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
                typename hpx::traits::extract_action<
                    Action
                >::remote_result_type>
        {};

        template <typename Action>
        struct fold_result<fold_with_index<Action> >
          : fold_result<Action>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename FoldOp
          , typename ...Ts
        >
        typename fold_result<Action>::type
        fold_impl(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , FoldOp && fold_op
          , typename fold_result<Action>::type const& init
          , long global_idx
          , Ts const&... vs
        );

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename Futures
          , typename ...Ts
        >
        void
        fold_invoke(Action /*act*/
          , Futures& futures
          , hpx::id_type const& id
          , std::size_t
          , Ts const&... vs)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , vs...
                )
            );
        }

        template <
            typename Action
          , typename Futures
          , typename ...Ts
        >
        void
        fold_invoke(fold_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , std::size_t global_idx
          , Ts const&... vs)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , vs...
                  , global_idx
                )
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename FoldOp
          , typename ...Ts
        >
        struct fold_invoker
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
              , long global_idx
              , Ts const&... vs
            )
            {
                return
                    fold_impl(
                        act
                      , ids
                      , fold_op
                      , init
                      , global_idx
                      , vs...
                    );
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename Is>
        struct make_fold_action_impl;

        template <typename Action, std::size_t ...Is>
        struct make_fold_action_impl<Action, util::detail::pack_c<std::size_t, Is...> >
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

                typedef detail::fold_invoker<
                        Action
                      , fold_op_type
                      , typename util::tuple_element<
                            Is, typename Action::arguments_type
                        >::type...
                    >
                    fold_invoker_type;

                typedef
                    typename HPX_MAKE_ACTION(fold_invoker_type::call)::type
                    type;
            };
        };

        template <typename Action>
        struct make_fold_action
          : make_fold_action_impl<
                Action
              , typename util::detail::make_index_pack<Action::arity>::type
            >
        {};

        template <typename Action>
        struct make_fold_action<fold_with_index<Action> >
          : make_fold_action_impl<
                fold_with_index<Action>
              , typename util::detail::make_index_pack<Action::arity - 1>::type
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename FoldOp>
        struct perform_folding
        {
            perform_folding(FoldOp const& fold_op, Result const& init)
              : fold_op_(fold_op), init_(init)
            {}

            Result operator()(
                hpx::future<std::vector<hpx::future<Result> > > r) const
            {
                std::vector<hpx::future<Result> > fres = std::move(r.get());
                HPX_ASSERT(!fres.empty());

                // we're at the beginning of the folding chain, incorporate the
                // initial value
                if (fres.size() == 1)
                    return fold_op_(init_, fres[0].get());

                // in the middle of the folding chain we simply apply the folding
                // operation to the two values we received.
                HPX_ASSERT(fres.size() == 2);
                return fold_op_(fres[0].get(), fres[1].get());
            }

            FoldOp const& fold_op_;
            Result const& init_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename FoldOp
          , typename ...Ts
        >
        typename fold_result<Action>::type
        fold_impl(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , FoldOp && fold_op
          , typename fold_result<Action>::type const& init
          , long global_idx
          , Ts const&... vs
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
                    fold_futures.push_back(
                        hpx::detail::async_colocated<fold_impl_action>(
                            ids_next.front()
                          , act
                          , std::move(ids_next)
                          , fold_op
                          , init
                          , global_idx + 1
                          , vs...
                        )
                    );
                }
            }

            // now perform the local operation
            fold_invoke(
                act
              , fold_futures
              , id_first
              , std::abs(global_idx)
              , vs...
            );

            return hpx::when_all(fold_futures).
                then(perform_folding<result_type, FoldOp>(fold_op, init)).
                get();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename FoldOp
      , typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    fold(
        std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        typedef
            typename detail::fold_result<Action>::type
            action_result;

        if (ids.empty())
        {
            return hpx::make_exceptional_future<action_result>(
                HPX_GET_EXCEPTION(bad_parameter, "hpx::lcos::fold",
                    "empty list of targets for fold operation"));
        }

        typedef
            typename detail::make_fold_action<Action>::
                template fold_invoker<FoldOp>::type
            fold_impl_action;

        return
            hpx::detail::async_colocated<fold_impl_action>(
                ids[0]
              , Action()
              , ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , 0
              , vs...
            );
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename FoldOp, typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    fold(
        hpx::actions::basic_action<Component, Signature, Derived> /* act */
      , std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return fold<Derived>(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }

    template <
        typename Action
      , typename FoldOp
      , typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    fold_with_index(
        std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return fold<detail::fold_with_index<Action> >(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename FoldOp, typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    fold_with_index(
        hpx::actions::basic_action<Component, Signature, Derived> /* act */
      , std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return fold<detail::fold_with_index<Derived> >(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename FoldOp
      , typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    inverse_fold(
        std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        typedef
            typename detail::fold_result<Action>::type
            action_result;

        if (ids.empty())
        {
            return hpx::make_exceptional_future<action_result>(
                HPX_GET_EXCEPTION(bad_parameter,
                    "hpx::lcos::inverse_fold",
                    "empty list of targets for inverse_fold operation"));
        }

        std::vector<id_type> inverted_ids;
        std::reverse_copy(ids.begin(), ids.end(), std::back_inserter(inverted_ids));

        typedef
            typename detail::make_fold_action<Action>::
                template fold_invoker<FoldOp>::type
            fold_impl_action;

        if (ids.empty())
        {
            return hpx::make_exceptional_future<action_result>(
                    hpx::exception(hpx::bad_parameter,
                        "array of targets is empty")
                );
        }

        return
            hpx::detail::async_colocated<fold_impl_action>(
                inverted_ids[0]
              , Action()
              , inverted_ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , -static_cast<long>(inverted_ids.size()-1)
              , vs...
            );
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename FoldOp, typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    inverse_fold(
        hpx::actions::basic_action<Component, Signature, Derived> /* act */
      , std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return inverse_fold<Derived>(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }

    template <
        typename Action
      , typename FoldOp
      , typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Action>::type
    >
    inverse_fold_with_index(
        std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return inverse_fold<detail::fold_with_index<Action> >(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename FoldOp, typename Init
      , typename ...Ts
    >
    hpx::future<
        typename detail::fold_result<Derived>::type
    >
    inverse_fold_with_index(
        hpx::actions::basic_action<Component, Signature, Derived> /* act */
      , std::vector<hpx::id_type> const & ids
      , FoldOp && fold_op
      , Init && init
      , Ts const&... vs)
    {
        return inverse_fold<detail::fold_with_index<Derived> >(
                ids
              , std::forward<FoldOp>(fold_op)
              , std::forward<Init>(init)
              , vs...
            );
    }
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_ACTION_DECLARATION(...)                             \
    HPX_REGISTER_FOLD_ACTION_DECLARATION_(__VA_ARGS__)                        \
/**/
#define HPX_REGISTER_FOLD_ACTION_DECLARATION_(...)                            \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_FOLD_ACTION_DECLARATION_, HPX_PP_NARGS(__VA_ARGS__)      \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_ACTION_DECLARATION_2(Action, FoldOp)                \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , HPX_PP_CAT(HPX_PP_CAT(fold_, Action), FoldOp)                         \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_ACTION_DECLARATION_3(Action, FoldOp, Name)          \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , HPX_PP_CAT(fold_, Name)                                               \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_ACTION(...)                                         \
    HPX_REGISTER_FOLD_ACTION_(__VA_ARGS__)                                    \
/**/
#define HPX_REGISTER_FOLD_ACTION_(...)                                        \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_FOLD_ACTION_, HPX_PP_NARGS(__VA_ARGS__)                  \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_ACTION_2(Action, FoldOp)                            \
    HPX_REGISTER_ACTION(                                                      \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , HPX_PP_CAT(HPX_PP_CAT(fold_, Action), FoldOp)                         \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_ACTION_3(Action, FoldOp, Name)                      \
    HPX_REGISTER_ACTION(                                                      \
        ::hpx::lcos::detail::make_fold_action<Action>::                       \
            fold_invoker<FoldOp>::type                                        \
      , HPX_PP_CAT(fold_, Name)                                               \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION(...)                  \
    HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_(__VA_ARGS__)             \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_(...)                 \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_,                     \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_2(Action, FoldOp)     \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , HPX_PP_CAT(HPX_PP_CAT(fold_, Action), FoldOp)                         \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_DECLARATION_3(Action, FoldOp, Name) \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , HPX_PP_CAT(fold_, Name)                                               \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION(...)                              \
    HPX_REGISTER_FOLD_WITH_INDEX_ACTION_(__VA_ARGS__)                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_(...)                             \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_FOLD_WITH_INDEX_ACTION_, HPX_PP_NARGS(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_2(Action, FoldOp)                 \
    HPX_REGISTER_ACTION(                                                      \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , HPX_PP_CAT(HPX_PP_CAT(fold_, Action), FoldOp)                         \
    )                                                                         \
/**/
#define HPX_REGISTER_FOLD_WITH_INDEX_ACTION_3(Action, FoldOp, Name)           \
    HPX_REGISTER_ACTION(                                                      \
        ::hpx::lcos::detail::make_fold_action<                                \
            ::hpx::lcos::detail::fold_with_index<Action>                      \
        >::fold_invoker<FoldOp>::type                                         \
      , HPX_PP_CAT(fold_, Name)                                               \
    )                                                                         \
/**/

#endif

#endif // DOXYGEN
