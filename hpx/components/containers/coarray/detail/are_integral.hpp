//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/coarray/detail/are_integral.hpp

#ifndef HPX_ARE_INTEGRAL_HPP
#define HPX_ARE_INTEGRAL_HPP

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL

namespace hpx {  namespace detail {

    template<bool ...>
    struct bools;

    template<typename ... T>
    struct are_integral
    : public std::integral_constant< bool,
        std::is_same<
            bools<true, std::is_integral<T>::value ...>,
            bools<std::is_integral<T>::value ...,true > >::value >
    {};

}}

#endif //HPX_ARE_INTEGRAL_HPP
