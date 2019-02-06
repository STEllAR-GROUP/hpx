//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_BIND_BACK_HPP
#define HPX_UTIL_BIND_BACK_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/one_shot.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename Ts, typename ...Us>
        struct invoke_bound_back_result;

        template <typename F, typename ...Ts, typename ...Us>
        struct invoke_bound_back_result<F&, util::tuple<Ts...>&, Us...>
          : util::invoke_result<F&, Us..., Ts&...>
        {};

        template <typename F, typename ...Ts, typename ...Us>
        struct invoke_bound_back_result<F const&, util::tuple<Ts...> const&, Us...>
          : util::invoke_result<F const&, Us..., Ts const&...>
        {};

        template <typename F, typename ...Ts, typename ...Us>
        struct invoke_bound_back_result<F&&, util::tuple<Ts...>&&, Us...>
          : util::invoke_result<F, Us..., Ts...>
        {};

        template <typename F, typename ...Ts, typename ...Us>
        struct invoke_bound_back_result<F const&&, util::tuple<Ts...> const&&, Us...>
          : util::invoke_result<F const, Us..., Ts const...>
        {};

        template <std::size_t ...Is, typename F, typename Ts, typename ...Us>
        HPX_CONSTEXPR HPX_HOST_DEVICE
        typename invoke_bound_back_result<F&&, Ts&&, Us...>::type
        bound_back_impl(pack_c<std::size_t, Is...>,
            F&& f, Ts&& bound, Us&&... unbound)
        {
            using invoke_impl = typename detail::dispatch_invoke<F>::type;
            return invoke_impl{std::forward<F>(f)}(
                std::forward<Us>(unbound)...,
                util::get<Is>(std::forward<Ts>(bound))...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename ...Ts>
        struct bound_back
        {
        public:
            bound_back() {} // needed for serialization

            template <typename F_, typename ...Ts_, typename =
                typename std::enable_if<
                    !std::is_same<typename std::decay<F_>::type, bound_back>::value
                >::type>
            HPX_CONSTEXPR explicit bound_back(F_&& f, Ts_&&... vs)
              : _f(std::forward<F_>(f))
              , _args(std::forward<Ts_>(vs)...)
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound_back(bound_back const&) = default;
            bound_back(bound_back&&) = default;
#else
            HPX_CONSTEXPR HPX_HOST_DEVICE bound_back(bound_back const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            HPX_CONSTEXPR HPX_HOST_DEVICE bound_back(bound_back&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}
#endif

            bound_back& operator=(bound_back const&) = delete;

            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_back_result<
                typename std::decay<F>::type&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&,
                Us...
            >::type operator()(Us&&... vs) &
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_back_impl(index_pack{},
                    _f, _args, std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_back_result<
                typename std::decay<F>::type const&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&,
                Us...
            >::type operator()(Us&&... vs) const&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_back_impl(index_pack{},
                    _f, _args, std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_back_result<
                typename std::decay<F>::type&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&&,
                Us...
            >::type operator()(Us&&... vs) &&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_back_impl(index_pack{},
                    std::move(_f), std::move(_args), std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_CONSTEXPR HPX_HOST_DEVICE
            typename invoke_bound_back_result<
                typename std::decay<F>::type const&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&&,
                Us...
            >::type operator()(Us&&... vs) const&&
            {
                using index_pack =
                    typename detail::make_index_pack<sizeof...(Ts)>::type;
                return detail::bound_back_impl(index_pack{},
                    std::move(_f), std::move(_args), std::forward<Us>(vs)...);
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
                static util::itt::string_handle sh("bound_back");
                return sh;
#endif
            }
#endif

        private:
            typename std::decay<F>::type _f;
            util::tuple<typename util::decay_unwrap<Ts>::type...> _args;
        };
    }

    template <typename F, typename ...Ts>
    HPX_CONSTEXPR detail::bound_back<
        typename std::decay<F>::type,
        typename std::decay<Ts>::type...>
    bind_back(F&& f, Ts&&... vs) {
        typedef detail::bound_back<
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
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename ...Ts>
    struct get_function_address<util::detail::bound_back<F, Ts...> >
    {
        static std::size_t
            call(util::detail::bound_back<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    struct get_function_annotation<util::detail::bound_back<F, Ts...> >
    {
        static char const*
            call(util::detail::bound_back<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename ...Ts>
    struct get_function_annotation_itt<util::detail::bound_back<F, Ts...> >
    {
        static util::itt::string_handle
            call(util::detail::bound_back<F, Ts...> const& f) noexcept
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
    // serialization of the bound_back object
    template <typename Archive, typename F, typename ...Ts>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::bound_back<F, Ts...>& bound
      , unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }
}}

#endif
