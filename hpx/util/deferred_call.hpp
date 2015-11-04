//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DEFERRED_CALL_HPP
#define HPX_UTIL_DEFERRED_CALL_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Args>
        class deferred_call_impl //-V690
        {
        public:
            // default constructor is needed for serialization
            deferred_call_impl()
            {}

            template <typename F_, typename Args_>
            explicit deferred_call_impl(
                F_ && f
              , Args_ && args
            ) : _f(std::forward<F_>(f))
              , _args(std::forward<Args_>(args))
            {}

            deferred_call_impl(deferred_call_impl const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            deferred_call_impl(deferred_call_impl && other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {}

            typedef
                typename util::detail::fused_result_of<F(Args)>::type
                result_type;

            BOOST_FORCEINLINE result_type operator()()
            {
                return util::invoke_fused(
                    std::move(_f), std::move(_args));
            }

        public: // exposition-only
            F _f;
            Args _args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct deferred_call_result_of
    {};

    template <typename F, typename ...Ts>
    struct deferred_call_result_of<F(Ts...)>
      : util::result_of<typename util::decay<F>::type(
            typename decay_unwrap<Ts>::type...)>
    {};

    template <typename F, typename ...Ts>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename decay_unwrap<Ts>::type...>
    >
    deferred_call(F && f, Ts&&... vs)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename decay_unwrap<Ts>::type...>
        > result_type;

        return result_type(std::forward<F>(f),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }

    // nullary functions do not need to be bound again
    template <typename F>
    typename util::decay<F>::type
    deferred_call(F && f)
    {
        return std::forward<F>(f);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    // serialization of the deferred_call_impl object
    template <typename F, typename Args>
    void serialize(
        ::hpx::serialization::input_archive& ar
      , ::hpx::util::detail::deferred_call_impl<F, Args>& deferred_call_impl
      , unsigned int const /*version*/)
    {
        ar >> deferred_call_impl._f;
        ar >> deferred_call_impl._args;
    }

    template <typename F, typename Args>
    void serialize(
        ::hpx::serialization::output_archive& ar
      , ::hpx::util::detail::deferred_call_impl<F, Args>& deferred_call_impl
      , unsigned int const /*version*/)
    {
        ar << deferred_call_impl._f;
        ar << deferred_call_impl._args;
    }
}}

#endif
