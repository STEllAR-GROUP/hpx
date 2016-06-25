//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/type_traits/integral_constant.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <std::size_t I>
        struct placeholder
        {
            static std::size_t const value = I;
        };

        template <>
        struct placeholder<0>; // not a valid placeholder
    }

    namespace placeholders
    {
        HPX_STATIC_CONSTEXPR detail::placeholder<1> _1 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<2> _2 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<3> _3 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<4> _4 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<5> _5 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<6> _6 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<7> _7 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<8> _8 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<9> _9 = {};
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F>
        class one_shot_wrapper;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct bind_eval_bound_impl
        {
            typedef T& type;

            template <typename Us>
            static HPX_HOST_DEVICE HPX_FORCEINLINE
            type call(T& t, Us&& /*unbound*/)
            {
                return t;
            }
        };

        template <typename F, typename T>
        struct bind_eval_bound_impl<one_shot_wrapper<F>, T>
        {
            typedef T&& type;

            template <typename Us>
            static HPX_HOST_DEVICE HPX_FORCEINLINE
            type call(T& t, Us&& /*unbound*/)
            {
                return std::forward<T>(t);
            }
        };

        template <
            typename F, typename T, typename Us,
            typename Enable = void
        >
        struct bind_eval_impl
          : bind_eval_bound_impl<F, T>
        {};

        template <
            std::size_t I, typename Us,
            typename Enable = void
        >
        struct bind_eval_placeholder_impl
        {};

        template <std::size_t I, typename Us>
        struct bind_eval_placeholder_impl<I, Us,
            typename std::enable_if<
                (I < util::tuple_size<Us>::value)
            >::type
        >
        {
            typedef typename util::tuple_element<
                I, typename util::decay<Us>::type
            >::type&& type;

            template <typename T>
            static HPX_HOST_DEVICE HPX_FORCEINLINE
            type call(T&& /*t*/, Us&& unbound)
            {
                return util::get<I>(std::forward<Us>(unbound));
            }
        };

        template <typename F, typename T, typename Us>
        struct bind_eval_impl<F, T,  Us,
            typename std::enable_if<
                traits::is_placeholder<T>::value != 0
            >::type
        > : bind_eval_placeholder_impl<
                (std::size_t)traits::is_placeholder<T>::value - 1, Us
            >
        {};

        template <typename F, typename T, typename Us>
        struct bind_eval_impl<F, T, Us,
            typename std::enable_if<
                traits::is_bind_expression<T>::value
            >::type
        >
        {
            typedef typename util::detail::fused_result_of<
                T&(Us&&)
            >::type type;

            static HPX_HOST_DEVICE HPX_FORCEINLINE
            type call(T& t, Us&& unbound)
            {
                return util::invoke_fused(t, std::forward<Us>(unbound));
            }
        };

        template <typename F, typename T, typename Us>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename bind_eval_impl<F, T, Us>::type
        bind_eval(T& t, Us&& unbound)
        {
            return bind_eval_impl<F, T, Us>::call(
                t, std::forward<Us>(unbound));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Ts>
        struct is_simple_bind;

        template <typename ...Ts>
        struct is_simple_bind<util::tuple<Ts...> >
          : detail::none_of<
                traits::is_placeholder<Ts>...,
                traits::is_bind_expression<Ts>...
            >
        {};

        template <typename ...Ts>
        struct is_simple_bind<util::tuple<Ts...> const>
          : is_simple_bind<util::tuple<Ts const...> >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us>
        struct bound_result_of_impl;

        template <typename F, typename ...Ts, typename Us>
        struct bound_result_of_impl<F, util::tuple<Ts...>, Us>
          : util::result_of<
                F&(typename bind_eval_impl<F, Ts, Us>::type&&...)
            >
        {};

        template <typename F, typename Ts>
        struct bound_result_of_simple_impl;

        template <typename F, typename ...Ts>
        struct bound_result_of_simple_impl<F, util::tuple<Ts...> >
          : util::result_of<F&(Ts&...)>
        {};

        template <typename F, typename ...Ts>
        struct bound_result_of_simple_impl<one_shot_wrapper<F>, util::tuple<Ts...> >
          : util::result_of<F&&(Ts&&...)>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us>
        struct bound_result_of
          : std::conditional<
                !detail::is_simple_bind<Ts>::value,
                bound_result_of_impl<F, Ts, Us>,
                bound_result_of_simple_impl<F, Ts>
            >::type
        {};

        template <typename F, typename ...Ts, typename Us>
        struct bound_result_of<F, util::tuple<Ts...> const, Us>
          : bound_result_of<F, util::tuple<Ts const...>, Us>
        {};

        template <typename F, typename ...Ts, typename Us>
        struct bound_result_of<
            one_shot_wrapper<F> const, util::tuple<Ts...> const, Us>
        {}; // one-shot wrapper is not const callable

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us, std::size_t ...Is>
        HPX_HOST_DEVICE
        typename std::enable_if<
            !detail::is_simple_bind<Ts>::value,
            typename bound_result_of<F, Ts, Us>::type
        >::type bound_impl(F& f, Ts& bound, Us&& unbound,
            pack_c<std::size_t, Is...>)
        {
            return util::invoke(f,
                detail::bind_eval<F, typename util::tuple_element<Is, Ts>::type>(
                    util::get<Is>(bound), std::forward<Us>(unbound))...);
        }

        template <typename F, typename Ts, typename Us, std::size_t ...Is>
        HPX_HOST_DEVICE
        typename std::enable_if<
            detail::is_simple_bind<Ts>::value,
            typename bound_result_of<F, Ts, Us>::type
        >::type bound_impl(F& f, Ts& bound, Us&& /*unbound*/,
            pack_c<std::size_t, Is...>)
        {
            return util::invoke(f, util::get<Is>(bound)...);
        }

        template <typename F, typename Ts, typename Us, std::size_t ...Is>
        HPX_HOST_DEVICE
        typename std::enable_if<
            detail::is_simple_bind<Ts>::value,
            typename bound_result_of<one_shot_wrapper<F>, Ts, Us>::type
        >::type bound_impl(one_shot_wrapper<F>& f, Ts& bound, Us&& /*unbound*/,
            pack_c<std::size_t, Is...>)
        {
            return util::invoke(std::move(f), util::get<Is>(std::move(bound))...);
        }

        template <typename T>
        class bound;

        template <typename F, typename ...Ts>
        class bound<F(Ts...)>
        {
        public:
            bound() {} // needed for serialization

            explicit bound(F&& f, Ts&&... vs)
              : _f(std::forward<F>(f))
              , _args(std::forward<Ts>(vs)...)
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)
            bound(bound const&) = default;
            bound(bound&&) = default;
#else
            bound(bound const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            bound(bound&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}
#endif

            HPX_DELETE_COPY_ASSIGN(bound);
            HPX_DELETE_MOVE_ASSIGN(bound);

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename bound_result_of<
                typename std::decay<F>::type,
                util::tuple<typename util::decay_unwrap<Ts>::type...>,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs)
            {
                return detail::bound_impl(_f, _args,
                    util::forward_as_tuple(std::forward<Us>(vs)...),
                    typename detail::make_index_pack<sizeof...(Ts)>::type());
            }

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename bound_result_of<
                typename std::decay<F>::type const,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) const
            {
                return detail::bound_impl(_f, _args,
                    util::forward_as_tuple(std::forward<Us>(vs)...),
                    typename detail::make_index_pack<sizeof...(Ts)>::type());
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & _f;
                ar & _args;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<
                        typename std::decay<F>::type
                    >::call(_f);
            }

        private:
            typename std::decay<F>::type _f;
            util::tuple<typename util::decay_unwrap<Ts>::type...> _args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    inline typename std::enable_if<
        !traits::is_action<typename util::decay<F>::type>::value
      , detail::bound<F(Ts&&...)>
    >::type
    bind(F&& f, Ts&&... vs)
    {
        return detail::bound<F(Ts&&...)>(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
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

            one_shot_wrapper(one_shot_wrapper&& other)
              : _f(std::move(other._f))
            {}

            void check_call()
            {}
#           endif

            template <typename ...Ts>
            HPX_HOST_DEVICE inline
            typename util::result_of<F&&(Ts&&...)>::type
            operator()(Ts&&... vs)
            {
                check_call();
                return util::invoke(std::move(_f), std::forward<Ts>(vs)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & _f;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

        public: // exposition-only
            F _f;
#           if !defined(HPX_DISABLE_ASSERTS)
            bool _called;
#           endif
        };
    }

    template <typename F>
    inline detail::one_shot_wrapper<typename util::decay<F>::type>
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
    template <typename T>
    struct is_bind_expression<util::detail::bound<T> >
      : boost::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I>
    struct is_placeholder<util::detail::placeholder<I> >
      : boost::integral_constant<int, I>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct get_function_address<util::detail::bound<Sig> >
    {
        static std::size_t
            call(util::detail::bound<Sig> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_address<util::detail::one_shot_wrapper<F> >
    {
        static std::size_t
            call(util::detail::one_shot_wrapper<F> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    // serialization of the bound object
    template <typename Archive, typename T>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::bound<T>& bound
      , unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }

    template <typename Archive, typename F>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const version = 0)
    {
        one_shot_wrapper.serialize(ar, version);
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <typename Archive, std::size_t I>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::placeholder<I>& /*placeholder*/
      , unsigned int const /*version*/ = 0)
    {}
}}

#endif
