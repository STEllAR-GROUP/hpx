//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEM_FN_HPP
#define HPX_UTIL_MEM_FN_HPP

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>

#include <boost/move/move.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_member_pointer.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename MemPtr>
        struct mem_fn
        {
            explicit mem_fn(MemPtr mem_ptr)
              : f(mem_ptr)
            {}

            mem_fn(mem_fn const& other)
              : f(other.f)
            {}

            mem_fn& operator=(mem_fn const& other)
            {
                f = other.f;
                return *this;
            }
            
            template <typename>
            struct result;

#           define HPX_UTIL_MEM_FN_INVOKE(Z, N, D)                            \
            template <typename This                                           \
              , typename T BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>      \
            struct result<This(T BOOST_PP_ENUM_TRAILING_PARAMS(N, A))>        \
              : util::invoke_result_of<                                       \
                    MemPtr(T BOOST_PP_ENUM_TRAILING_PARAMS(N, A))>            \
            {};                                                               \
                                                                              \
            template <typename T BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>\
            BOOST_FORCEINLINE                                                 \
            typename util::invoke_result_of<                                  \
                MemPtr(T BOOST_PP_ENUM_TRAILING_PARAMS(N, A))                 \
            >::type                                                           \
            operator()(BOOST_FWD_REF(T) t                                     \
                BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, A, a)) const        \
            {                                                                 \
                return                                                        \
                    util::invoke(f, boost::forward<T>(t)                      \
                        BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, A, a)); \
            }                                                                 \
            /**/
            
            BOOST_PP_REPEAT(
                HPX_PP_ROUND_UP_ADD3(HPX_FUNCTION_ARGUMENT_LIMIT)
              , HPX_UTIL_MEM_FN_INVOKE, _
            )

#           undef HPX_UTIL_MEM_FN_INVOKE

            MemPtr f;
        };
    }

    template <typename MemPtr>
    detail::mem_fn<MemPtr> mem_fn(MemPtr pm)
    {
        BOOST_STATIC_ASSERT((boost::is_member_pointer<MemPtr>::value));

        return detail::mem_fn<MemPtr>(pm);
    }
}}

#endif
