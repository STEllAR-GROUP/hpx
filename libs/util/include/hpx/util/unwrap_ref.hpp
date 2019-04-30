//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#if !defined(HPX_UTIL_UNWRAP_REF_JAN_05_2017_0356PM)
#define HPX_UTIL_UNWRAP_REF_JAN_05_2017_0356PM

#include <hpx/config.hpp>

#include <boost/ref.hpp>

#include <functional>

namespace hpx { namespace util
{
    template <typename T>
    struct unwrap_reference
    {
        typedef T type;
    };

    template <typename T>
    struct unwrap_reference<boost::reference_wrapper<T> >
    {
        typedef T type;
    };

    template <typename T>
    struct unwrap_reference<boost::reference_wrapper<T> const>
    {
        typedef T type;
    };

    template <typename T>
    struct unwrap_reference<std::reference_wrapper<T> >
    {
        typedef T type;
    };

    template <typename T>
    struct unwrap_reference<std::reference_wrapper<T> const>
    {
        typedef T type;
    };

    template <typename T>
    HPX_FORCEINLINE typename unwrap_reference<T>::type&
    unwrap_ref(T& t)
    {
        return t;
    }
}}

#endif


