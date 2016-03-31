//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_BIND_ACTION_HPP
#define HPX_UTIL_BIND_ACTION_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl
        {
            typedef bool type;

            template <std::size_t ...Is>
            static HPX_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , BoundArgs& bound_args, UnboundArgs&& unbound_args
            )
            {
                return hpx::apply<Action>(bind_eval<Action>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename Action, typename BoundArgs, typename UnboundArgs>
        HPX_FORCEINLINE
        bool
        bind_action_apply(BoundArgs& bound_args, UnboundArgs&& unbound_args)
        {
            return bind_action_apply_impl<Action, BoundArgs, UnboundArgs>::call(
                typename detail::make_index_pack<
                    util::tuple_size<BoundArgs>::value
                >::type(),
                bound_args, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl
        {
            typedef bool type;

            template <std::size_t ...Is>
            static HPX_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , naming::id_type const& cont
              , BoundArgs& bound_args, UnboundArgs&& unbound_args
            )
            {
                return hpx::apply_c<Action>(cont, bind_eval<Action>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename Action, typename BoundArgs, typename UnboundArgs>
        HPX_FORCEINLINE
        bool
        bind_action_apply_cont(naming::id_type const& cont,
            BoundArgs& bound_args, UnboundArgs&& unbound_args
        )
        {
            return bind_action_apply_cont_impl<
                    Action, BoundArgs, UnboundArgs
                >::call(
                    typename detail::make_index_pack<
                        util::tuple_size<BoundArgs>::value
                    >::type(), cont,
                    bound_args, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl2
        {
            typedef bool type;

            template <typename Continuation, std::size_t ...Is>
            static HPX_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , Continuation && cont
              , BoundArgs& bound_args, UnboundArgs&& unbound_args
            )
            {
                return hpx::apply<Action>(std::forward<Continuation>(cont),
                    bind_eval<Action>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename Action, typename Continuation, typename BoundArgs,
            typename UnboundArgs>
        HPX_FORCEINLINE
        typename boost::enable_if_c<
            traits::is_continuation<Continuation>::value, bool
        >::type
        bind_action_apply_cont2(Continuation && cont,
            BoundArgs& bound_args, UnboundArgs&& unbound_args)
        {
            return bind_action_apply_cont_impl2<
                    Action, BoundArgs, UnboundArgs
                >::call(
                    typename detail::make_index_pack<
                        util::tuple_size<BoundArgs>::value
                    >::type(), std::forward<Continuation>(cont),
                    bound_args, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl
        {
            typedef lcos::future<typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type> type;

            template <std::size_t ...Is>
            static HPX_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , BoundArgs& bound_args, UnboundArgs&& unbound_args
            )
            {
                return hpx::async<Action>(bind_eval<Action>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename Action, typename BoundArgs, typename UnboundArgs>
        HPX_FORCEINLINE
        lcos::future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
        bind_action_async(BoundArgs& bound_args, UnboundArgs&& unbound_args)
        {
            return bind_action_async_impl<Action, BoundArgs, UnboundArgs>::call(
                typename detail::make_index_pack<
                    util::tuple_size<BoundArgs>::value
                >::type(),
                bound_args, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        HPX_FORCEINLINE
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
        bind_action_invoke(
            BoundArgs& bound_args
          , UnboundArgs&& unbound_args
        )
        {
            return bind_action_async_impl<Action, BoundArgs, UnboundArgs>::call(
                typename detail::make_index_pack<
                    util::tuple_size<BoundArgs>::value
                >::type(),
                bound_args, std::forward<UnboundArgs>(unbound_args)).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename BoundArgs>
        class bound_action //-V690
        {
        public:
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type result_type;

        public:
            // default constructor is needed for serialization
            bound_action()
            {}

            template <typename BoundArgs_>
            explicit bound_action(Action /*action*/, BoundArgs_&& bound_args)
              : _bound_args(std::forward<BoundArgs_>(bound_args))
            {}

            bound_action(bound_action const& other)
              : _bound_args(other._bound_args)
            {}
            bound_action(bound_action&& other)
              : _bound_args(std::move(other._bound_args))
            {}

            template <typename ...Us>
            HPX_FORCEINLINE
            bool
            apply(Us&&... us) const
            {
                return detail::bind_action_apply<Action>(
                    _bound_args, util::forward_as_tuple(std::forward<Us>(us)...));
            }

            template <typename ...Us>
            HPX_FORCEINLINE
            bool
            apply_c(naming::id_type const& contgid, Us&&... us) const
            {
                return detail::bind_action_apply_cont<Action>(contgid,
                    _bound_args, util::forward_as_tuple(std::forward<Us>(us)...));
            }

            template <typename Continuation, typename ...Us>
            HPX_FORCEINLINE
            typename boost::enable_if_c<
                traits::is_continuation<Continuation>::value, bool
            >::type
            apply_c(Continuation && cont, Us&&... us) const
            {
                return detail::bind_action_apply_cont2<Action>
                        (std::forward<Continuation>(cont),
                    _bound_args, util::forward_as_tuple(std::forward<Us>(us)...));
            }

            template <typename ...Us>
            HPX_FORCEINLINE
            hpx::lcos::future<result_type>
            async(Us&&... us) const
            {
                return detail::bind_action_async<Action>(
                    _bound_args, util::forward_as_tuple(std::forward<Us>(us)...));
            }

            template <typename ...Us>
            HPX_FORCEINLINE
            result_type
            operator()(Us&&... us) const
            {
                return detail::bind_action_invoke<Action>(
                    _bound_args, util::forward_as_tuple(std::forward<Us>(us)...));
            }

        public: // exposition-only
            BoundArgs _bound_args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_action<typename util::decay<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<Ts>::type...>
        >
    >::type
    bind(Ts&&... vs)
    {
        typedef detail::bound_action<
            typename util::decay<Action>::type,
            util::tuple<typename util::decay<Ts>::type...>
        > result_type;

        return result_type(Action(),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename ...Ts>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<Ts>::type...>
    >
    bind(
        hpx::actions::basic_action<Component, Signature, Derived> action,
        Ts&&... vs)
    {
        typedef detail::bound_action<
            Derived,
            util::tuple<typename util::decay<Ts>::type...>
        > result_type;

        return result_type(static_cast<Derived const&>(action),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename BoundArgs>
    struct is_bind_expression<util::detail::bound_action<Action, BoundArgs> >
      : boost::mpl::true_
    {};

    template <typename Action, typename BoundArgs>
    struct is_bound_action<util::detail::bound_action<Action, BoundArgs> >
      : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    // serialization of the bound action object
    template <typename Action, typename BoundArgs>
    void serialize(
        ::hpx::serialization::input_archive& ar
      , ::hpx::util::detail::bound_action<Action, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar >> bound._bound_args;
    }

    template <typename Action, typename BoundArgs>
    void serialize(
        ::hpx::serialization::output_archive& ar
      , ::hpx::util::detail::bound_action<Action, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar << bound._bound_args;
    }
}}

#endif
