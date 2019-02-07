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
        template <std::size_t I>
        struct bind_eval_placeholder
        {
            template <typename T, typename Us>
            static HPX_CONSTEXPR HPX_HOST_DEVICE
            typename util::tuple_element<
                I, typename std::remove_reference<Us>::type
            >::type&& call(T&& /*t*/, Us&& unbound)
            {
                return util::get<I>(std::forward<Us>(unbound));
            }
        };

        template <
            typename T, typename TD = typename std::decay<T>::type,
            typename Enable = void>
        struct bind_eval
        {
            template <typename Us>
            static HPX_CONSTEXPR HPX_HOST_DEVICE
            T&& call(T&& t, Us&& /*unbound*/)
            {
                return std::forward<T>(t);
            }
        };

        template <typename T, typename TD>
        struct bind_eval<T, TD,
            typename std::enable_if<
                traits::is_placeholder<TD>::value != 0
            >::type
        > : bind_eval_placeholder<
                (std::size_t)traits::is_placeholder<TD>::value - 1>
        {};

        template <typename T, typename TD>
        struct bind_eval<T, TD,
            typename std::enable_if<
                traits::is_bind_expression<TD>::value
            >::type
        >
        {
            template <typename Us>
            static HPX_CONSTEXPR HPX_HOST_DEVICE
            typename util::detail::invoke_fused_result<T, Us>::type
            call(T&& t, Us&& unbound)
            {
                return util::invoke_fused(
                    std::forward<T>(t), std::forward<Us>(unbound));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Us>
        struct invoke_bound_result;

        template <typename F, typename ...Ts, typename Us>
        struct invoke_bound_result<F, util::tuple<Ts...>, Us>
          : util::invoke_result<F, decltype(bind_eval<Ts>::call(
                std::declval<Ts>(), std::declval<Us>()))...>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename Is>
        struct bound_impl;

        template <typename F, typename ...Ts, std::size_t ...Is>
        struct bound_impl<F, util::tuple<Ts...>, pack_c<std::size_t, Is...>>
        {
            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                F&,
                util::tuple<Ts&...>,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) &
            {
                using invoke_impl = typename detail::dispatch_invoke<F&>::type;
                return invoke_impl(_f)(detail::bind_eval<Ts&>::call(
                    util::get<Is>(_args),
                    util::forward_as_tuple(std::forward<Us>(vs)...))...);
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                F const&,
                util::tuple<Ts const&...>,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) const&
            {
                using invoke_impl = typename detail::dispatch_invoke<F const&>::type;
                return invoke_impl(_f)(detail::bind_eval<Ts const&>::call(
                    util::get<Is>(_args),
                    util::forward_as_tuple(std::forward<Us>(vs)...))...);
            }

            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                F&&,
                util::tuple<Ts&&...>,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) &&
            {
                using invoke_impl = typename detail::dispatch_invoke<F>::type;
                return invoke_impl(std::move(_f))(detail::bind_eval<Ts>::call(
                    util::get<Is>(std::move(_args)),
                    util::forward_as_tuple(std::forward<Us>(vs)...))...);
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_result<
                F const&&,
                util::tuple<Ts const&&...>,
                util::tuple<Us&&...>
            >::type operator()(Us&&... vs) const&&
            {
                using invoke_impl = typename detail::dispatch_invoke<F const>::type;
                return invoke_impl(std::move(_f))(detail::bind_eval<Ts const>::call(
                    util::get<Is>(std::move(_args)),
                    util::forward_as_tuple(std::forward<Us>(vs)...))...);
            }

            F _f;
            util::tuple<Ts...> _args;
        };

        template <typename F, typename ...Ts>
        class bound
          : private bound_impl<
                F, util::tuple<typename util::decay_unwrap<Ts>::type...>,
                typename detail::make_index_pack<sizeof...(Ts)>::type
            >
        {
            using base_type = detail::bound_impl<
                F, util::tuple<typename util::decay_unwrap<Ts>::type...>,
                typename detail::make_index_pack<sizeof...(Ts)>::type
            >;

        public:
            bound() {} // needed for serialization

            template <typename F_, typename ...Ts_, typename =
                typename std::enable_if<
                    !std::is_same<typename std::decay<F_>::type, bound>::value
                >::type>
            HPX_CONSTEXPR explicit bound(F_&& f, Ts_&&... vs)
              : base_type{
                    std::forward<F_>(f),
                    util::forward_as_tuple(std::forward<Ts_>(vs)...)}
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound(bound const&) = default;
            bound(bound&&) = default;
#else
            HPX_HOST_DEVICE bound(bound const& other)
              : base_type{other}
            {}

            HPX_HOST_DEVICE bound(bound&& other)
              : base_type{std::move(other)}
            {}
#endif

            bound& operator=(bound const&) = delete;

            using base_type::operator();

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & _f;
                ar & _args;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<F>::call(_f);
#else
                return nullptr;
#endif
            }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<F>::call(_f);
#else
                static util::itt::string_handle sh("bound");
                return sh;
#endif
            }
#endif

        private:
            using base_type::_f;
            using base_type::_args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    HPX_CONSTEXPR typename std::enable_if<
        !traits::is_action<typename std::decay<F>::type>::value
      , detail::bound<
            typename std::decay<F>::type,
            typename std::decay<Ts>::type...>
    >::type bind(F&& f, Ts&&... vs)
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
