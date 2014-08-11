//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file broadcast.hpp

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
    /// \brief Perform a distributed broadcast operation
    ///
    /// The function hpx::lcos::broadcast performs a distributed broadcast
    /// operation resulting in action invocations on a given set
    /// of global identifiers. The action can be either a plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// The given action is invoked asynchronously on all given identifiers,
    /// and the arguments ArgN are passed along to those invocations.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall reduction operation.
    ///
    /// \note            If decltype(Action(...)) is void, then the result of
    ///                  this function is future<void>.
    ///
    template <typename Action, typename ArgN, ...>
    hpx::future<std::vector<decltype(Action(hpx::id_type, ArgN, ...))> >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , ArgN argN, ...);

    /// \brief Perform an asynchronous (fire&forget) distributed broadcast operation
    ///
    /// The function hpx::lcos::broadcast_apply performs an asynchronous
    /// (fire&forget) distributed broadcast operation resulting in action
    /// invocations on a given set of global identifiers. The action can be
    /// either a plain action (in which case the global identifiers have to
    /// refer to localities) or a component action (in which case the global
    /// identifiers have to refer to instances of a component type which
    /// exposes the action.
    ///
    /// The given action is invoked asynchronously on all given identifiers,
    /// and the arguments ArgN are passed along to those invocations.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    template <typename Action, typename ArgN, ...>
    void
    broadcast_apply(
        std::vector<hpx::id_type> const & ids
      , ArgN argN, ...);

    /// \brief Perform a distributed broadcast operation
    ///
    /// The function hpx::lcos::broadcast_with_index performs a distributed broadcast
    /// operation resulting in action invocations on a given set
    /// of global identifiers. The action can be either a plain action (in
    /// which case the global identifiers have to refer to localities) or a
    /// component action (in which case the global identifiers have to refer
    /// to instances of a component type which exposes the action.
    ///
    /// The given action is invoked asynchronously on all given identifiers,
    /// and the arguments ArgN are passed along to those invocations.
    ///
    /// The function passes the index of the global identifier in the given
    /// list of identifiers as the last argument to the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    /// \returns         This function returns a future representing the result
    ///                  of the overall reduction operation.
    ///
    /// \note            If decltype(Action(...)) is void, then the result of
    ///                  this function is future<void>.
    ///
    template <typename Action, typename ArgN, ...>
    hpx::future<std::vector<decltype(Action(hpx::id_type, ArgN, ..., std::size_t))> >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , ArgN argN, ...);

    /// \brief Perform an asynchronous (fire&forget) distributed broadcast operation
    ///
    /// The function hpx::lcos::broadcast_apply_with_index performs an asynchronous
    /// (fire&forget) distributed broadcast operation resulting in action
    /// invocations on a given set of global identifiers. The action can be
    /// either a plain action (in which case the global identifiers have to
    /// refer to localities) or a component action (in which case the global
    /// identifiers have to refer to instances of a component type which
    /// exposes the action.
    ///
    /// The given action is invoked asynchronously on all given identifiers,
    /// and the arguments ArgN are passed along to those invocations.
    ///
    /// The function passes the index of the global identifier in the given
    /// list of identifiers as the last argument to the action.
    ///
    /// \param ids       [in] A list of global identifiers identifying the
    ///                  target objects for which the given action will be
    ///                  invoked.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  by const reference) which will be forwarded to the
    ///                  action invocation.
    ///
    template <typename Action, typename ArgN, ...>
    void
    broadcast_apply_with_index(
        std::vector<hpx::id_type> const & ids
      , ArgN argN, ...);
}}
#else

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_BROADCAST_HPP
#define HPX_LCOS_BROADCAST_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/lcos/async_colocated.hpp>
#include <hpx/runtime/applier/apply_colocated.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/calculate_fanout.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/cat.hpp>

#if !defined(HPX_BROADCAST_FANOUT)
#define HPX_BROADCAST_FANOUT 16
#endif

namespace hpx { namespace lcos
{
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

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, int N>
        struct make_broadcast_action_impl;

        template <typename Action>
        struct make_broadcast_action
          : make_broadcast_action_impl<
                Action
              , util::tuple_size<typename Action::arguments_type>::value
            >
        {};

        template <typename Action>
        struct make_broadcast_action<broadcast_with_index<Action> >
          : make_broadcast_action_impl<
                broadcast_with_index<Action>
              , util::tuple_size<typename Action::arguments_type>::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, int N>
        struct make_broadcast_apply_action_impl;

        template <typename Action>
        struct make_broadcast_apply_action
          : make_broadcast_apply_action_impl<
                Action
              , util::tuple_size<typename Action::arguments_type>::value
            >
        {};

        template <typename Action>
        struct make_broadcast_apply_action<broadcast_with_index<Action> >
          : make_broadcast_apply_action_impl<
                broadcast_with_index<Action>
              , util::tuple_size<typename Action::arguments_type>::value - 1
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        inline void return_void(
            hpx::future<std::vector<hpx::future<void> > >)
        {
            // todo: verify validity of all futures in the vector
        }

        template <
            typename Result
        >
        std::vector<Result>
        wrap_into_vector(hpx::future<Result> r)
        {
            return std::vector<Result>(1, r.get());
        }

        template <
            typename Result
        >
        std::vector<Result>
        return_result_type(
            hpx::future<std::vector<hpx::future<std::vector<Result> > > > r)
        {
            std::vector<Result> res;
            std::vector<hpx::future<std::vector<Result> > > fres = std::move(r.get());

            BOOST_FOREACH(hpx::future<std::vector<Result> >& f, fres)
            {
                std::vector<Result> t = std::move(f.get());
                res.reserve(res.capacity() + t.size());
                std::move(t.begin(), t.end(), std::back_inserter(res));
            }

            return std::move(res);
        }
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/broadcast.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/broadcast_" HPX_LIMIT_STR ".hpp")
#endif

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

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(...)                  \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_(__VA_ARGS__)             \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_(...)                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_,                     \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_1(Action)             \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_2(Action, Action)         \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION_2(Action, Name)       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_apply_action<Action>::type        \
      , BOOST_PP_CAT(broadcast_apply_, Name)                                  \
    )                                                                         \
    HPX_REGISTER_APPLY_COLOCATED_DECLARATION(                                 \
        ::hpx::lcos::detail::make_broadcast_apply_action<Action>::type        \
      , BOOST_PP_CAT(apply_colocated_broadcast_, Name)                        \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_APPLY_ACTION(...)                              \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_(__VA_ARGS__)                         \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_ACTION_(...)                             \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_APPLY_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)   \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_APPLY_ACTION_1(Action)                         \
    HPX_REGISTER_BROADCAST_APPLY_ACTION_2(Action, Action)                     \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_ACTION_2(Action, Name)                   \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_apply_action<Action>::type        \
      , BOOST_PP_CAT(broadcast_apply_, Name)                                  \
    )                                                                         \
    HPX_REGISTER_APPLY_COLOCATED(                                             \
        ::hpx::lcos::detail::make_broadcast_apply_action<Action>::type        \
      , BOOST_PP_CAT(apply_colocated_broadcast_, Name)                        \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION(...)       \
    HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_(__VA_ARGS__)  \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_(...)      \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_,          \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_1(Action)  \
    HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_2(             \
        Action, Action)                                                       \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION_2(         \
        Action, Name)                                                         \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_apply_action<                     \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_apply_with_index_, Name)                       \
    )                                                                         \
    HPX_REGISTER_APPLY_COLOCATED_DECLARATION(                                 \
        ::hpx::lcos::detail::make_broadcast_apply_action<                     \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(apply_colocated_broadcast_with_index_, Name)             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION(...)                   \
    HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_(__VA_ARGS__)              \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_(...)                  \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_,                      \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_1(Action)              \
    HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_2(Action, Action)          \
/**/
#define HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_2(Action, Name)        \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_apply_action<                     \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_apply_with_index_, Name)                       \
    )                                                                         \
    HPX_REGISTER_APPLY_COLOCATED(                                             \
        ::hpx::lcos::detail::make_broadcast_apply_action<                     \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(apply_colocated_broadcast_with_index_, Name)             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION(...)                        \
    HPX_REGISTER_BROADCAST_ACTION_DECLARATION_(__VA_ARGS__)                   \
/**/
#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION_(...)                       \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_ACTION_DECLARATION_,                           \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION_1(Action)                   \
    HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(Action, Action)               \
/**/
#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(Action, Name)             \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
    HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(                                 \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(async_colocated_broadcast_, Name)                        \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_ACTION(...)                                    \
    HPX_REGISTER_BROADCAST_ACTION_(__VA_ARGS__)                               \
/**/
#define HPX_REGISTER_BROADCAST_ACTION_(...)                                   \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION_1(Action)                               \
    HPX_REGISTER_BROADCAST_ACTION_2(Action, Action)                           \
/**/
#define HPX_REGISTER_BROADCAST_ACTION_2(Action, Name)                         \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(broadcast_, Name)                                        \
    )                                                                         \
    HPX_REGISTER_ASYNC_COLOCATED(                                             \
        ::hpx::lcos::detail::make_broadcast_action<Action>::type              \
      , BOOST_PP_CAT(async_colocated_broadcast_, Name)                        \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION(...)             \
    HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_(__VA_ARGS__)        \
/**/
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_(...)            \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_,                \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_1(Action)        \
    HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_2(Action, Action)    \
/**/
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_DECLARATION_2(Action, Name)  \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_with_index_, Name)                             \
    )                                                                         \
    HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(                                 \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(async_colocated_broadcast_with_index_, Name)             \
    )                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION(...)                         \
    HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_(__VA_ARGS__)                    \
/**/
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_(...)                        \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_,                            \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_1(Action)                    \
    HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_2(Action, Action)                \
/**/
#define HPX_REGISTER_BROADCAST_WITH_INDEX_ACTION_2(Action, Name)              \
    HPX_REGISTER_PLAIN_ACTION(                                                \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(broadcast_with_index_, Name)                             \
    )                                                                         \
    HPX_REGISTER_ASYNC_COLOCATED(                                             \
        ::hpx::lcos::detail::make_broadcast_action<                           \
            ::hpx::lcos::detail::broadcast_with_index<Action>                 \
        >::type                                                               \
      , BOOST_PP_CAT(async_colocated_broadcast_with_index_, Name)             \
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
        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          , typename Futures
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
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
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
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
        broadcast_invoke(Action act, Futures& futures, Cont && cont
          , hpx::id_type const& id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                ).then(std::forward<Cont>(cont))
            );
        }

        template <
            typename Action
          , typename Futures
          , typename Cont
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , Cont && cont
          , hpx::id_type const& id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx
                ).then(std::forward<Cont>(cont))
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke_apply(Action act
          , hpx::id_type const& id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t)
        {
            hpx::apply(
                act
              , id
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
            );
        }

        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        broadcast_invoke_apply(broadcast_with_index<Action>
          , hpx::id_type const& id
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx)
        {
            hpx::apply(
                Action()
              , id
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , global_idx
            );
        }

        ///////////////////////////////////////////////////////////////////////
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

            std::size_t const local_fanout = HPX_BROADCAST_FANOUT;
            std::size_t local_size = (std::min)(ids.size(), local_fanout);
            std::size_t fanout = util::calculate_fanout(ids.size(), local_fanout);

            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(local_size + (ids.size()/fanout) + 1);
            for(std::size_t i = 0; i != local_size; ++i)
            {
                broadcast_invoke(
                    act
                  , broadcast_futures
                  , ids[i]
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx + i
                );
            }

            if(ids.size() > local_fanout)
            {
                std::size_t applied = local_fanout;
                std::vector<hpx::id_type>::const_iterator it =
                    ids.begin() + local_fanout;

                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;

                while(it != ids.end())
                {
                    HPX_ASSERT(ids.size() >= applied);

                    std::size_t next_fan = (std::min)(fanout, ids.size() - applied);
                    std::vector<hpx::id_type> ids_next(it, it + next_fan);

                    hpx::id_type id(ids_next[0]);
                    broadcast_futures.push_back(
                        hpx::async_colocated<broadcast_impl_action>(
                            id
                          , act
                          , std::move(ids_next)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + applied
                          , boost::integral_constant<bool, true>::type()
                        )
                    );

                    applied += next_fan;
                    it += next_fan;
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

            std::size_t const local_fanout = HPX_BROADCAST_FANOUT;
            std::size_t local_size = (std::min)(ids.size(), local_fanout);
            std::size_t fanout = util::calculate_fanout(ids.size(), local_fanout);

            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(local_size + (ids.size()/fanout) + 1);
            for(std::size_t i = 0; i != local_size; ++i)
            {
                broadcast_invoke(
                    act
                  , broadcast_futures
                  , &wrap_into_vector<action_result>
                  , ids[i]
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx + i
                );
            }

            if(ids.size() > local_fanout)
            {
                std::size_t applied = local_fanout;
                std::vector<hpx::id_type>::const_iterator it =
                    ids.begin() + local_fanout;

                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;

                while(it != ids.end())
                {
                    HPX_ASSERT(ids.size() >= applied);

                    std::size_t next_fan = (std::min)(fanout, ids.size() - applied);
                    std::vector<hpx::id_type> ids_next(it, it + next_fan);

                    hpx::id_type id(ids_next[0]);
                    broadcast_futures.push_back(
                        hpx::async_colocated<broadcast_impl_action>(
                            id
                          , act
                          , std::move(ids_next)
                          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                          , global_idx + applied
                          , boost::integral_constant<bool, false>::type()
                        )
                    );

                    applied += next_fan;
                    it += next_fan;
                }
            }

            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        void
        BOOST_PP_CAT(broadcast_apply_impl, N)(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
          , std::size_t global_idx
        )
        {
            if(ids.empty()) return;

            std::size_t const local_fanout = HPX_BROADCAST_FANOUT;
            std::size_t local_size = (std::min)(ids.size(), local_fanout);

            for(std::size_t i = 0; i != local_size; ++i)
            {
                broadcast_invoke_apply(
                    act
                  , ids[i]
                  BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                  , global_idx + i
                );
            }

            if(ids.size() > local_fanout)
            {
                std::size_t applied = local_fanout;
                std::vector<hpx::id_type>::const_iterator it =
                    ids.begin() + local_fanout;

                typedef
                    typename detail::make_broadcast_apply_action<
                        Action
                    >::type
                    broadcast_impl_action;

                std::size_t fanout = util::calculate_fanout(ids.size(), local_fanout);
                while(it != ids.end())
                {
                    HPX_ASSERT(ids.size() >= applied);

                    std::size_t next_fan = (std::min)(fanout, ids.size() - applied);
                    std::vector<hpx::id_type> ids_next(it, it + next_fan);

                    hpx::id_type id(ids_next[0]);
                    hpx::apply_colocated<broadcast_impl_action>(
                        id
                      , act
                      , std::move(ids_next)
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , global_idx + applied
                    );

                    applied += next_fan;
                    it += next_fan;
                }
            }
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

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
        >
        struct BOOST_PP_CAT(broadcast_apply_invoker, N)
        {
            static void
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)
              , std::size_t global_idx
            )
            {
                return
                    BOOST_PP_CAT(broadcast_apply_impl, N)(
                        act
                      , ids
                      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
                      , global_idx
                    );
            }
        };

        template <typename Action>
        struct make_broadcast_apply_action_impl<Action, N>
        {
            typedef BOOST_PP_CAT(broadcast_apply_invoker, N)<
                        Action
                      BOOST_PP_COMMA_IF(N)
                        BOOST_PP_ENUM(
                            N
                          , HPX_LCOS_BROADCAST_EXTRACT_ACTION_ARGUMENTS
                          , _
                        )
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
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;

        return
            hpx::async_colocated<broadcast_impl_action>(
                ids[0]
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

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    void
    broadcast_apply(
        std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        typedef
            typename detail::make_broadcast_apply_action<Action>::type
            broadcast_impl_action;

        hpx::apply_colocated<broadcast_impl_action>(
                ids[0]
              , Action()
              , ids
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
              , 0
            );
    }

    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    void
    broadcast_apply(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        broadcast_apply<Derived>(
            ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
        );
    }

    ///////////////////////////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    void
    broadcast_apply_with_index(
        std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        broadcast_apply<detail::broadcast_with_index<Action> >(
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
    void
    broadcast_apply_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /* act */
      , std::vector<hpx::id_type> const & ids
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        broadcast_apply<detail::broadcast_with_index<Derived> >(
            ids
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a)
        );
    }
}}

#endif
#endif // DOXYGEN
