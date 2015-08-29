//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct cannot_be_called {};

        struct not_enough_arguments {};

        template <typename F>
        class one_shot_wrapper;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct bind_eval_bound_impl
        {
            typedef T& type;

            template <typename UnboundArgs>
            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs&& /*unbound_args*/)
            {
                return t;
            }
        };

        template <typename F, typename T>
        struct bind_eval_bound_impl<one_shot_wrapper<F>, T>
        {
            typedef T&& type;

            template <typename UnboundArgs>
            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs&& /*unbound_args*/)
            {
                return std::move(t);
            }
        };

        template <
            typename F
          , typename T, typename UnboundArgs
          , typename Enable = void
        >
        struct bind_eval_impl
          : bind_eval_bound_impl<F, T>
        {};

        template <std::size_t I, typename UnboundArgs, typename Enable = void>
        struct bind_eval_placeholder_impl
        {
            typedef typename util::tuple_element<
                I
              , typename util::decay<UnboundArgs>::type
            >::type&& type;

            template <typename T>
            static BOOST_FORCEINLINE
            type call(T& /*t*/, UnboundArgs&& unbound_args)
            {
                return util::get<I>(std::forward<UnboundArgs>(unbound_args));
            }
        };
        template <std::size_t I, typename UnboundArgs>
        struct bind_eval_placeholder_impl<
            I, UnboundArgs
          , typename boost::enable_if_c<
                I >= util::tuple_size<UnboundArgs>::value
            >::type
        >
        {
            typedef not_enough_arguments type;

            template <typename T>
            static BOOST_FORCEINLINE
            type call(T& /*t*/, UnboundArgs&& /*unbound_args*/)
            {
                return not_enough_arguments();
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                traits::is_placeholder<
                    typename boost::remove_const<T>::type
                >::value != 0
            >::type
        > : bind_eval_placeholder_impl<
                traits::is_placeholder<
                    typename boost::remove_const<T>::type
                >::value - 1
              , UnboundArgs
            >
        {};

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                traits::is_bind_expression<
                    typename boost::remove_const<T>::type
                >::value
            >::type
        >
        {
            typedef typename util::invoke_fused_result_of<
                T&(UnboundArgs)
            >::type type;

            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs&& unbound_args)
            {
                return util::invoke_fused_r<type>
                    (t, std::forward<UnboundArgs>(unbound_args));
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                boost::is_reference_wrapper<
                    typename boost::remove_const<T>::type
                >::value
            >::type
        >
        {
            typedef typename boost::unwrap_reference<T>::type& type;

            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs&& /*unbound_args*/)
            {
                return t.get();
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        BOOST_FORCEINLINE
        typename bind_eval_impl<F, T, UnboundArgs>::type
        bind_eval(T& t, UnboundArgs&& unbound_args)
        {
            return bind_eval_impl<F, T, UnboundArgs>::call
                    (t, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl;

        template <typename F, typename ...Ts, typename UnboundArgs>
        struct bind_invoke_impl<
            F, util::tuple<Ts...>, UnboundArgs
        >
        {
            typedef typename util::detail::result_of_or<
                  F(typename bind_eval_impl<F, Ts, UnboundArgs>::type...)
                , cannot_be_called
            >::type type;

            template <std::size_t ...Is>
            static BOOST_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , F& f, util::tuple<Ts...>& bound_args
              , UnboundArgs&& unbound_args
            )
            {
                return util::invoke_r<type>(f, bind_eval<F>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename F, typename ...Ts, typename UnboundArgs>
        struct bind_invoke_impl<
            F, util::tuple<Ts...> const, UnboundArgs
        >
        {
            typedef typename util::detail::result_of_or<
                F(typename bind_eval_impl<F, Ts const, UnboundArgs>::type...)
              , cannot_be_called
            >::type type;

            template <std::size_t ...Is>
            static BOOST_FORCEINLINE
            type call(
                detail::pack_c<std::size_t, Is...>
              , F& f, util::tuple<Ts...> const& bound_args
              , UnboundArgs&& unbound_args
            )
            {
                return util::invoke_r<type>(f, bind_eval<F>(
                    util::get<Is>(bound_args),
                    std::forward<UnboundArgs>(unbound_args))...);
            }
        };

        template <typename F, typename ...Ts, typename UnboundArgs>
        struct bind_invoke_impl<
            one_shot_wrapper<F> const, util::tuple<Ts...>, UnboundArgs
        >
        {
            typedef cannot_be_called type;
        };

        template <typename F, typename ...Ts, typename UnboundArgs>
        struct bind_invoke_impl<
            one_shot_wrapper<F> const, util::tuple<Ts...> const, UnboundArgs
        >
        {
            typedef cannot_be_called type;
        };

        template <typename F, typename BoundArgs, typename UnboundArgs>
        BOOST_FORCEINLINE
        typename bind_invoke_impl<F, BoundArgs, UnboundArgs>::type
        bind_invoke(
            F& f, BoundArgs& bound_args
          , UnboundArgs&& unbound_args
        )
        {
            return bind_invoke_impl<F, BoundArgs, UnboundArgs>::call(
                typename detail::make_index_pack<
                    util::tuple_size<BoundArgs>::value
                >::type()
              , f, bound_args
              , std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F>
        class one_shot_wrapper //-V690
        {
        public:
#           if !defined(HPX_DISABLE_ASSERTS)
            // default constructor is needed for serialization
            one_shot_wrapper()
              : _called(false)
            {}

            explicit one_shot_wrapper(F const& f)
              : _f(f)
              , _called(false)
            {}
            explicit one_shot_wrapper(F&& f)
              : _f(std::move(f))
              , _called(false)
            {}

            one_shot_wrapper(one_shot_wrapper const& other)
              : _f(other._f)
              , _called(other._called)
            {}
            one_shot_wrapper(one_shot_wrapper&& other)
              : _f(std::move(other._f))
              , _called(other._called)
            {
                other._called = true;
            }

            void check_call()
            {
                HPX_ASSERT(!_called);

                _called = true;
            }
#           else
            // default constructor is needed for serialization
            one_shot_wrapper()
            {}

            explicit one_shot_wrapper(F const& f)
              : _f(f)
            {}
            explicit one_shot_wrapper(F&& f)
              : _f(std::move(f))
            {}

            one_shot_wrapper(one_shot_wrapper const& other)
              : _f(other._f)
            {}
            one_shot_wrapper(one_shot_wrapper&& other)
              : _f(std::move(other._f))
            {}

            void check_call()
            {}
#           endif

            template <typename>
            struct result;

            template <typename This, typename ...Ts>
            struct result<This(Ts...)>
              : util::detail::result_of_or<F(Ts...), cannot_be_called>
            {};

            template <typename This, typename ...Ts>
            struct result<This const(Ts...)>
              : boost::mpl::identity<cannot_be_called>
            {};

            template <typename ...Ts>
            BOOST_FORCEINLINE
            typename result<one_shot_wrapper(Ts...)>::type
            operator()(Ts&&... vs)
            {
                typedef typename result<
                    one_shot_wrapper(Ts...)
                >::type result_type;

                check_call();
                return util::invoke_r<result_type>(_f, std::forward<Ts>(vs)...);
            }

        public: // exposition-only
            F _f;
#           if !defined(HPX_DISABLE_ASSERTS)
            bool _called;
#           endif
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename BoundArgs>
        class bound //-V690
        {
        public:
            // default constructor is needed for serialization
            bound()
            {}

            template <typename F_, typename BoundArgs_>
            explicit bound(F_&& f, BoundArgs_&& bound_args)
              : _f(std::forward<F_>(f))
              , _bound_args(std::forward<BoundArgs_>(bound_args))
            {}

            bound(bound const& other)
              : _f(other._f)
              , _bound_args(other._bound_args)
            {}
            bound(bound&& other)
              : _f(std::move(other._f))
              , _bound_args(std::move(other._bound_args))
            {}

            template <typename>
            struct result;

            template <typename This, typename ...Ts>
            struct result<This(Ts...)>
              : bind_invoke_impl<F, BoundArgs, util::tuple<Ts&&...> >
            {};

            template <typename This, typename ...Ts>
            struct result<This const(Ts...)>
              : bind_invoke_impl<F const, BoundArgs const, util::tuple<Ts&&...> >
            {};

            template <typename ...Us>
            BOOST_FORCEINLINE
            typename result<bound(Us...)>::type
            operator()(Us&&... us)
            {
                return detail::bind_invoke(_f, _bound_args,
                    util::forward_as_tuple(std::forward<Us>(us)...));
            }

            template <typename ...Us>
            BOOST_FORCEINLINE
            typename result<bound const(Us...)>::type
            operator()(Us&&... us) const
            {
                return detail::bind_invoke(_f, _bound_args,
                    util::forward_as_tuple(std::forward<Us>(us)...));
            }

        public: // exposition-only
            F _f;
            BoundArgs _bound_args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <std::size_t I>
        struct placeholder
        {
            static std::size_t const value = I;
        };
    }

    namespace placeholders
    {
        detail::placeholder<1> const _1 = {};
        detail::placeholder<2> const _2 = {};
        detail::placeholder<3> const _3 = {};
        detail::placeholder<4> const _4 = {};
        detail::placeholder<5> const _5 = {};
        detail::placeholder<6> const _6 = {};
        detail::placeholder<7> const _7 = {};
        detail::placeholder<8> const _8 = {};
        detail::placeholder<9> const _9 = {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    typename boost::disable_if_c<
        traits::is_action<typename util::decay<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<Ts>::type...>
        >
    >::type
    bind(F&& f, Ts&&... vs)
    {
        typedef detail::bound<
            typename util::decay<F>::type,
            util::tuple<typename util::decay<Ts>::type...>
        > result_type;

        return result_type(std::forward<F>(f),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    detail::one_shot_wrapper<typename util::decay<F>::type>
    one_shot(F&& f)
    {
        typedef
            detail::one_shot_wrapper<typename util::decay<F>::type>
            result_type;

        return result_type(std::forward<F>(f));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename BoundArgs>
    struct is_bind_expression<util::detail::bound<F, BoundArgs> >
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I>
    struct is_placeholder<util::detail::placeholder<I> >
      : util::detail::placeholder<I>
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    // serialization of the bound object
    template <typename F, typename BoundArgs>
    void serialize(
        ::hpx::serialization::input_archive& ar
      , ::hpx::util::detail::bound<F, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar >> bound._f;
        ar >> bound._bound_args;
    }

    template <typename F, typename BoundArgs>
    void serialize(
        ::hpx::serialization::output_archive& ar
      , ::hpx::util::detail::bound<F, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar << bound._f;
        ar << bound._bound_args;
    }

    // serialization of the bound object
    template <typename F>
    void serialize(
        ::hpx::serialization::input_archive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const /*version*/)
    {
        ar >> one_shot_wrapper._f;
        ar >> one_shot_wrapper._called;
    }

    template <typename F>
    void serialize(
        ::hpx::serialization::output_archive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const /*version*/)
    {
        ar << one_shot_wrapper._f;
        ar << one_shot_wrapper._called;
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <std::size_t I>
    void serialize(
        ::hpx::serialization::input_archive& ar
      , ::hpx::util::detail::placeholder<I>& bound
      , unsigned int const /*version*/)
    {}

    template <std::size_t I>
    void serialize(
        ::hpx::serialization::output_archive& ar
      , ::hpx::util::detail::placeholder<I>& bound
      , unsigned int const /*version*/)
    {}
}}

#endif
