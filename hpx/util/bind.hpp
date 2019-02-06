//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/one_shot.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
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
        template <
            typename T, typename Us,
            typename TD = typename std::decay<T>::type,
            typename Enable = void
        >
        struct bind_eval_impl
        {
            typedef T&& type;

            static HPX_CONSTEXPR HPX_HOST_DEVICE
            type call(T&& t, Us&& /*unbound*/)
            {
                return std::forward<T>(t);
            }
        };

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
                I, typename std::decay<Us>::type
            >::type&& type;

            template <typename T>
            static HPX_CONSTEXPR HPX_HOST_DEVICE
            type call(T&& /*t*/, Us&& unbound)
            {
                return util::get<I>(std::forward<Us>(unbound));
            }
        };

        template <typename T, typename Us, typename TD>
        struct bind_eval_impl<T, Us, TD,
            typename std::enable_if<
                traits::is_placeholder<TD>::value != 0
            >::type
        > : bind_eval_placeholder_impl<
                (std::size_t)traits::is_placeholder<TD>::value - 1, Us
            >
        {};

        template <typename T, typename Us, typename TD>
        struct bind_eval_impl<T, Us, TD,
            typename std::enable_if<
                traits::is_bind_expression<TD>::value
            >::type
        >
        {
            typedef typename util::detail::invoke_fused_result<T, Us>::type type;

            static HPX_CONSTEXPR HPX_HOST_DEVICE
            type call(T&& t, Us&& unbound)
            {
                return util::invoke_fused(
                    std::forward<T>(t), std::forward<Us>(unbound));
            }
        };

        template <typename T, typename Us>
        HPX_CONSTEXPR HPX_HOST_DEVICE
        typename bind_eval_impl<T, Us>::type
        bind_eval(T&& t, Us&& unbound)
        {
            return bind_eval_impl<T, Us>::call(
                std::forward<T>(t), std::forward<Us>(unbound));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us>
        struct invoke_bound_result_impl;

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result_impl<F, util::tuple<Ts...>, Us>
          : util::invoke_result<
                F, typename bind_eval_impl<Ts, Us>::type...
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us>
        struct invoke_bound_result;

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result<F&, util::tuple<Ts...>&, Us>
          : invoke_bound_result_impl<F&, util::tuple<Ts&...>, Us>
        {};

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result<F const&, util::tuple<Ts...> const&, Us>
          : invoke_bound_result_impl<F const&, util::tuple<Ts const&...>, Us>
        {};

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result<F&&, util::tuple<Ts...>&&, Us>
          : invoke_bound_result_impl<F&&, util::tuple<Ts&&...>, Us>
        {};

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result<F const&&, util::tuple<Ts...> const&&, Us>
          : invoke_bound_result_impl<F const&&, util::tuple<Ts const&&...>, Us>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t ...Is, typename F, typename Ts, typename Us>
        HPX_CONSTEXPR HPX_HOST_DEVICE
        typename invoke_bound_result<F&&, Ts&&, Us>::type
        bound_impl(pack_c<std::size_t, Is...>,
            F&& f, Ts&& bound, Us&& unbound)
        {
            using invoke_impl = typename detail::dispatch_invoke<F>::type;
            return invoke_impl{std::forward<F>(f)}(
                detail::bind_eval(
                    util::get<Is>(std::forward<Ts>(bound)),
                    std::forward<Us>(unbound))...);
        }

        template <typename F, typename ...Ts>
        class bound
        {
        public:
            bound() {} // needed for serialization

            template <typename F_, typename ...Ts_, typename =
                typename std::enable_if<
                    !std::is_same<typename std::decay<F_>::type, bound>::value
                >::type>
            HPX_CONSTEXPR explicit bound(F_&& f, Ts_&&... vs)
              : _f(std::forward<F_>(f))
              , _args(std::forward<Ts_>(vs)...)
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound(bound const&) = default;
            bound(bound&&) = default;
#else
            HPX_HOST_DEVICE bound(bound const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            HPX_HOST_DEVICE bound(bound&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}
#endif

            bound& operator=(bound const&) = delete;

            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                typename std::decay<F>::type&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) &
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_impl(index_pack{},
                    _f, _args, util::forward_as_tuple(std::forward<Us>(vs)...));
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                typename std::decay<F>::type const&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) const&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_impl(index_pack{},
                    _f, _args, util::forward_as_tuple(std::forward<Us>(vs)...));
            }

            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                typename std::decay<F>::type&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&&,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) &&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_impl(index_pack{},
                    std::move(_f), std::move(_args),
                    util::forward_as_tuple(std::forward<Us>(vs)...));
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                typename std::decay<F>::type const&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&&,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) const&&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_impl(index_pack{},
                    std::move(_f), std::move(_args),
                    util::forward_as_tuple(std::forward<Us>(vs)...));
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

            char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<
                        typename std::decay<F>::type
                    >::call(_f);
#else
                return nullptr;
#endif
            }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<
                        typename std::decay<F>::type
                    >::call(_f);
#else
                static util::itt::string_handle sh("bound");
                return sh;
#endif
            }
#endif

        private:
            typename std::decay<F>::type _f;
            util::tuple<typename util::decay_unwrap<Ts>::type...> _args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    HPX_CONSTEXPR typename std::enable_if<
        !traits::is_action<typename std::decay<F>::type>::value
      , detail::bound<
            typename std::decay<F>::type,
            typename std::decay<Ts>::type...>
    >::type
    bind(F&& f, Ts&&... vs)
    {
        typedef detail::bound<
            typename std::decay<F>::type,
            typename std::decay<Ts>::type...
        > result_type;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    struct is_bind_expression<util::detail::bound<F, Ts...> >
      : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I>
    struct is_placeholder<util::detail::placeholder<I> >
      : std::integral_constant<int, I>
    {};

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename ...Ts>
    struct get_function_address<util::detail::bound<F, Ts...> >
    {
        static std::size_t
            call(util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    struct get_function_annotation<util::detail::bound<F, Ts...> >
    {
        static char const*
            call(util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename ...Ts>
    struct get_function_annotation_itt<util::detail::bound<F, Ts...> >
    {
        static util::itt::string_handle
            call(util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    // serialization of the bound object
    template <typename Archive, typename F, typename ...Ts>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::bound<F, Ts...>& bound
      , unsigned int const version = 0)
    {
        bound.serialize(ar, version);
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
