//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DEFERRED_CALL_HPP
#define HPX_UTIL_DEFERRED_CALL_HPP

#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
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
        template <typename F, typename Args>
        class deferred_call_impl
        {
            BOOST_COPYABLE_AND_MOVABLE(deferred_call_impl);

        public:
            // default constructor is needed for serialization
            deferred_call_impl()
            {}

            template <typename F_, typename Args_>
            explicit deferred_call_impl(
                BOOST_FWD_REF(F_) f
              , BOOST_FWD_REF(Args_) args
            ) : _f(boost::forward<F_>(f))
              , _args(boost::forward<Args_>(args))
            {}

            deferred_call_impl(deferred_call_impl const& other)
              : _f(other._f)
              , _args(other._args)
            {}

            deferred_call_impl(BOOST_RV_REF(deferred_call_impl) other)
              : _f(boost::move(other._f))
              , _args(boost::move(other._args))
            {}

            typedef 
                typename util::invoke_fused_result_of<F(Args)>::type
                result_type;

            BOOST_FORCEINLINE result_type operator()()
            {
                return util::invoke_fused(boost::move(_f), boost::move(_args));
            }

        public: // exposition-only
            F _f;
            Args _args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<>
    >
    deferred_call(BOOST_FWD_REF(F) f)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<>
        > result_type;

        return result_type(boost::forward<F>(f), util::forward_as_tuple());
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the deferred_call_impl object
    template <typename F, typename Args>
    void serialize(
        ::hpx::util::portable_binary_iarchive& ar
      , ::hpx::util::detail::deferred_call_impl<F, Args>& deferred_call_impl
      , unsigned int const /*version*/)
    {
        ar >> deferred_call_impl._f;
        ar >> deferred_call_impl._args;
    }

    template <typename F, typename Args>
    void serialize(
        ::hpx::util::portable_binary_oarchive& ar
      , ::hpx::util::detail::deferred_call_impl<F, Args>& deferred_call_impl
      , unsigned int const /*version*/)
    {
        ar << deferred_call_impl._f;
        ar << deferred_call_impl._args;
    }
}}

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/preprocessed/deferred_call.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/deferred_call_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_FUNCTION_ARGUMENT_LIMIT                                   \
              , <hpx/util/deferred_call.hpp>                                  \
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
#   define HPX_UTIL_DEFERRED_CALL_DECAY(Z, N, D)                              \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename T)>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_DEFERRED_CALL_DECAY, _)>
    >
    deferred_call(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, T, t))
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_DEFERRED_CALL_DECAY, _)>
        > result_type;

        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, T, t)));
    }
#   undef HPX_UTIL_DEFERRED_CALL_DECAY
}}

#undef N

#endif
