//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#ifndef HPX_UTIL_DECAY_HPP
#define HPX_UTIL_DECAY_HPP

#include <hpx/config.hpp>

#include <boost/ref.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    template <typename T>
    struct decay : std::decay<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename TD>
        struct decay_unwrap_impl
        {
            typedef TD type;
        };

        template <typename X>
        struct decay_unwrap_impl< ::boost::reference_wrapper<X> >
        {
            typedef X& type;
        };

        template <typename X>
        struct decay_unwrap_impl< ::std::reference_wrapper<X> >
        {
            typedef X& type;
        };
    }

    template <typename T>
    struct decay_unwrap
      : detail::decay_unwrap_impl<typename std::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        HPX_FORCEINLINE typename decay<T>::type
        decay_copy(T&& v)
        {
            return std::forward<T>(v);
        }
    }
}}

#endif
