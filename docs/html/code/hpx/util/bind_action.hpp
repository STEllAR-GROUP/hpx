//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_BIND_ACTION_HPP
#define HPX_UTIL_BIND_ACTION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/add_rvalue_reference.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action, typename BoundArgs, typename UnboundArgs
          , typename Enable = void
        >
        struct bind_action_apply_impl;

        template <typename Action, typename BoundArgs, typename UnboundArgs>
        BOOST_FORCEINLINE
        bool
        bind_action_apply(
            BoundArgs& bound_args
          , BOOST_FWD_REF(UnboundArgs) unbound_args
        )
        {
            return
                bind_action_apply_impl<Action, BoundArgs, UnboundArgs>::call(
                    bound_args
                  , boost::forward<UnboundArgs>(unbound_args)
                );
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Action, typename BoundArgs, typename UnboundArgs
          , typename Enable = void
        >
        struct bind_action_async_impl;

        template <typename Action, typename BoundArgs, typename UnboundArgs>
        BOOST_FORCEINLINE
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type
        >
        bind_action_async(
            BoundArgs& bound_args
          , BOOST_FWD_REF(UnboundArgs) unbound_args
        )
        {
            return
                bind_action_async_impl<Action, BoundArgs, UnboundArgs>::call(
                    bound_args
                  , boost::forward<UnboundArgs>(unbound_args)
                );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        BOOST_FORCEINLINE
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type
        bind_action_invoke(
            BoundArgs& bound_args
          , BOOST_FWD_REF(UnboundArgs) unbound_args
        )
        {
            return
                bind_action_async_impl<Action, BoundArgs, UnboundArgs>::call(
                    bound_args
                  , boost::forward<UnboundArgs>(unbound_args)
                ).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs>
        class bound_action
        {
            BOOST_COPYABLE_AND_MOVABLE(bound_action);
            
        public:
            typedef
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<Action>::result_type
                >::type
                result_type;

        public:
            // default constructor is needed for serialization
            bound_action()
            {}

            template <typename BoundArgs_>
            explicit bound_action(
                Action /*action*/
              , BOOST_FWD_REF(BoundArgs_) bound_args
            ) : _bound_args(boost::forward<BoundArgs_>(bound_args))
            {}

            bound_action(bound_action const& other)
              : _bound_args(other._bound_args)
            {}
            bound_action(BOOST_RV_REF(bound_action) other)
              : _bound_args(boost::move(other._bound_args))
            {}

            // bring in the definition for all overloads for apply, async, operator()
            #include <hpx/util/detail/define_bind_action_function_operators.hpp>

        public: // exposition-only
            BoundArgs _bound_args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    typename boost::enable_if_c<
        traits::is_action<typename util::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<>
        >
    >::type
    bind()
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<>
            >
            result_type;

        return result_type(Action(), util::forward_as_tuple());
    }
    
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
    >
    detail::bound_action<Derived, util::tuple<> >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action
    )
    {
        typedef
            detail::bound_action<Derived, util::tuple<> >
            result_type;

        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple()
            );
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Action, typename BoundArgs>
    struct is_bind_expression<util::detail::bound_action<Action, BoundArgs> >
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename BoundArgs>
    struct is_bound_action<util::detail::bound_action<Action, BoundArgs> >
      : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the bound action object
    template <typename Action, typename BoundArgs>
    void serialize(
        ::hpx::util::portable_binary_iarchive& ar
      , ::hpx::util::detail::bound_action<Action, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar >> bound._bound_args;
    }

    template <typename Action, typename BoundArgs>
    void serialize(
        ::hpx::util::portable_binary_oarchive& ar
      , ::hpx::util::detail::bound_action<Action, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar << bound._bound_args;
    }
}}

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/preprocessed/bind_action.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/bind_action_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_FUNCTION_ARGUMENT_LIMIT                                   \
              , <hpx/util/bind_action.hpp>                                    \
            )                                                                 \
        )                                                                     \
        /**/
#       include BOOST_PP_ITERATE()

#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(output: null)
#       endif
#   endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    namespace detail
    {
#       define HPX_UTIL_BIND_EVAL_TYPE(Z, N, D)                               \
        typename detail::bind_eval_impl<                                      \
            typename util::tuple_element<N, BoundArgs>::type                  \
          , UnboundArgs                                                       \
        >::type                                                               \
        /**/
#       define HPX_UTIL_BIND_EVAL(Z, N, D)                                    \
        detail::bind_eval(                                                    \
            util::get<N>(bound_args)                                          \
          , boost::forward<UnboundArgs>(unbound_args)                         \
        )                                                                     \
        /**/
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == N
            >::type
        >
        {
            typedef bool type;

            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _)
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == N
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;

            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _)
                    );
            }
        };
#       undef HPX_UTIL_BIND_EVAL_TYPE
#       undef HPX_UTIL_BIND_EVAL
    }

#   define HPX_UTIL_BIND_DECAY(Z, N, D)                                       \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename T)>
    typename boost::enable_if_c<
        traits::is_action<typename util::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
        >
    >::type
    bind(HPX_ENUM_FWD_ARGS(N, T, t))
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
            >
            result_type;

        return
            result_type(
                Action()
              , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, T, t))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , BOOST_PP_ENUM_PARAMS(N, typename T)>
    detail::bound_action<
        Derived
      , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, HPX_ENUM_FWD_ARGS(N, T, t))
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
            >
            result_type;

        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, T, t))
            );
    }
#   undef HPX_UTIL_BIND_DECAY
}}

#undef N

#endif
