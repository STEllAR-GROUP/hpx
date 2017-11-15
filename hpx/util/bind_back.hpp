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
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
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

        template <typename F, std::size_t ...Is, typename Ts, typename ...Us>
        HPX_HOST_DEVICE
        typename invoke_bound_back_result<F&&, Ts&&, Us...>::type
        bound_back_impl(F&& f, pack_c<std::size_t, Is...>, Ts&& bound,
            Us&&... unbound)
        {
            return util::invoke(std::forward<F>(f),
                std::forward<Us>(unbound)...,
                util::get<Is>(std::forward<Ts>(bound))...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct bound_back;

        template <typename F, typename ...Ts>
        struct bound_back<F(Ts...)>
        {
        public:
            bound_back() {} // needed for serialization

            explicit bound_back(F&& f, Ts&&... vs)
              : _f(std::forward<F>(f))
              , _args(std::forward<Ts>(vs)...)
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound_back(bound_back const&) = default;
            bound_back(bound_back&&) = default;
#else
            HPX_HOST_DEVICE bound_back(bound_back const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            HPX_HOST_DEVICE bound_back(bound_back&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}
#endif

            bound_back& operator=(bound_back const&) = delete;

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename invoke_bound_back_result<
                typename std::decay<F>::type&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&,
                Us...
            >::type operator()(Us&&... vs) &
            {
                return detail::bound_back_impl(_f,
                    typename detail::make_index_pack<sizeof...(Ts)>::type(),
                    _args, std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename invoke_bound_back_result<
                typename std::decay<F>::type const&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&,
                Us...
            >::type operator()(Us&&... vs) const&
            {
                return detail::bound_back_impl(_f,
                    typename detail::make_index_pack<sizeof...(Ts)>::type(),
                    _args, std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename invoke_bound_back_result<
                typename std::decay<F>::type&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...>&&,
                Us...
            >::type operator()(Us&&... vs) &&
            {
                return detail::bound_back_impl(std::move(_f),
                    typename detail::make_index_pack<sizeof...(Ts)>::type(),
                    std::move(_args), std::forward<Us>(vs)...);
            }

            template <typename ...Us>
            HPX_HOST_DEVICE inline
            typename invoke_bound_back_result<
                typename std::decay<F>::type const&&,
                util::tuple<typename util::decay_unwrap<Ts>::type...> const&&,
                Us...
            >::type operator()(Us&&... vs) const&&
            {
                return detail::bound_back_impl(std::move(_f),
                    typename detail::make_index_pack<sizeof...(Ts)>::type(),
                    std::move(_args), std::forward<Us>(vs)...);
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
    detail::bound_back<F(Ts&&...)>
    bind_back(F&& f, Ts&&... vs) {
        return detail::bound_back<F(Ts&&...)>(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename Sig>
    struct get_function_address<util::detail::bound_back<Sig> >
    {
        static std::size_t
            call(util::detail::bound_back<Sig> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct get_function_annotation<util::detail::bound_back<Sig> >
    {
        static char const*
            call(util::detail::bound_back<Sig> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig>
    struct get_function_annotation_itt<util::detail::bound_back<Sig> >
    {
        static util::itt::string_handle
            call(util::detail::bound_back<Sig> const& f) noexcept
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
    template <typename Archive, typename T>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::bound_back<T>& bound
      , unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }
}}

#endif
