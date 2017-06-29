//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/coarray/detail/last_element.hpp

#ifndef HPX_LAST_ELEMENT_HPP
#define HPX_LAST_ELEMENT_HPP

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL

namespace hpx{ namespace detail{

    template <class... R>
    struct last_element;

    template <class T0>
    struct last_element<T0>
    {
        using type = T0;
    };

    template <class T0,class T1>
    struct last_element<T0,T1>
    {
       using type = T1;
    };

    template <class T0,class T1,class T2>
    struct last_element<T0,T1,T2>
    {
        using type = T2;
    };

    template <class T0,class T1,class T2,class T3>
    struct last_element<T0,T1,T2,T3>
    {
        using type = T3;
    };

    template <class T0,class T1,class T2,class T3,class T4>
    struct last_element<T0,T1,T2,T3,T4>
    {
        using type = T4;
    };

    template <class T0,class T1,class T2,class T3,class T4,class T5>
    struct last_element<T0,T1,T2,T3,T4,T5>
    {
        using type = T5;
    };

    template <class T0,class T1,class T2,class T3,class T4,class T5,class T6>
    struct last_element<T0,T1,T2,T3,T4,T5,T6>
    {
        using type = T6;
    };

    template <class T0,class T1,class T2,class T3,class T4,class T5,class T6,
        class T7>
    struct last_element<T0,T1,T2,T3,T4,T5,T6,T7>
    {
        using type = T7;
    };

    template <class T0,class T1,class T2,class T3,class T4,class T5,class T6,
        class T7,class... R>
    struct last_element<T0,T1,T2,T3,T4,T5,T6,T7,R...>
    {
        using type = typename last_element<R...>::type;
    };

}}

#endif //LAST_ELEMENT_HPP
