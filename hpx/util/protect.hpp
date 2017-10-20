//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013-2016 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_PROTECT_HPP
#define HPX_UTIL_PROTECT_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_bind_expression.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F>
        class protected_bind : public F
        {
        public:
            explicit protected_bind(F const& f)
              : F(f)
            {}

            explicit protected_bind(F&& f)
              : F(std::move(f))
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            protected_bind(protected_bind const&) = default;
            protected_bind(protected_bind&&) = default;
#else
            HPX_HOST_DEVICE protected_bind(protected_bind const& other)
              : F(other)
            {}

            HPX_HOST_DEVICE protected_bind(protected_bind&& other)
              : F(std::move(other))
            {}
#endif

            protected_bind& operator=(protected_bind const&) = delete;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_HOST_DEVICE
    typename std::enable_if<
        traits::is_bind_expression<typename std::decay<T>::type>::value
      , detail::protected_bind<typename std::decay<T>::type>
    >::type
    protect(T&& f)
    {
        return detail::protected_bind<
            typename std::decay<T>::type
        >(std::forward<T>(f));
    }

    // leave everything that is not a bind expression as is
    template <typename T>
    HPX_HOST_DEVICE
    typename std::enable_if<
        !traits::is_bind_expression<typename std::decay<T>::type>::value
      , T
    >::type
    protect(T&& v) //-V659
    {
        return std::forward<T>(v);
    }
}}

#endif /*HPX_UTIL_PROTECT_HPP*/
