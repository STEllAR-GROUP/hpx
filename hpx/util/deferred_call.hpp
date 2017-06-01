//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DEFERRED_CALL_HPP
#define HPX_UTIL_DEFERRED_CALL_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits { namespace detail
{
    template <typename T>
    struct is_deferred_callable;

    template <typename F, typename ...Ts>
    struct is_deferred_callable<F(Ts...)>
      : is_callable<
            typename util::decay_unwrap<F>::type(
                typename util::decay_unwrap<Ts>::type...)
        >
    {};

    template <typename F, typename ...Ts>
    struct is_deferred_invocable
      : is_invocable<
            typename util::decay_unwrap<F>::type,
            typename util::decay_unwrap<Ts>::type...
        >
    {};

}}}

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        struct deferred_result_of;

        template <typename F, typename ...Ts>
        struct deferred_result_of<F(Ts...)>
          : util::result_of<
                typename util::decay_unwrap<F>::type(
                    typename util::decay_unwrap<Ts>::type...)
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class deferred;

        template <typename F, typename ...Ts>
        class deferred<F(Ts...)>
        {
        public:
            deferred() {} // needed for serialization

            explicit HPX_HOST_DEVICE deferred(F&& f, Ts&&... vs)
              : _f(std::forward<F>(f))
              , _args(std::forward<Ts>(vs)...)
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            deferred(deferred&&) = default;
#else
            HPX_HOST_DEVICE deferred(deferred&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}
#endif

            deferred& operator=(deferred const&) = delete;

            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename deferred_result_of<F(Ts...)>::type
            operator()()
            {
                return util::invoke_fused(std::move(_f), std::move(_args));
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
                        typename util::decay_unwrap<F>::type
                    >::call(_f);
            }

            char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<
                        typename util::decay_unwrap<F>::type
                    >::call(_f);
#else
                return nullptr;
#endif
            }

        private:
            typename util::decay_unwrap<F>::type _f;
            util::tuple<typename util::decay_unwrap<Ts>::type...> _args;
        };
    }

    template <typename F, typename ...Ts>
    inline detail::deferred<F(Ts&&...)>
    deferred_call(F&& f, Ts&&... vs)
    {
        static_assert(
            traits::detail::is_deferred_callable<F(Ts&&...)>::value
          , "F shall be Callable with decay_t<Ts> arguments");

        return detail::deferred<F(Ts&&...)>(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    inline typename std::decay<F>::type
    deferred_call(F&& f)
    {
        static_assert(
            traits::detail::is_deferred_callable<F()>::value
          , "F shall be Callable with no arguments");

        return std::forward<F>(f);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct get_function_address<util::detail::deferred<Sig> >
    {
        static std::size_t
            call(util::detail::deferred<Sig> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct get_function_annotation<util::detail::deferred<Sig> >
    {
        static char const*
            call(util::detail::deferred<Sig> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename T>
    HPX_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::util::detail::deferred<T>& d
      , unsigned int const version = 0
    )
    {
        d.serialize(ar, version);
    }
}}

#endif
