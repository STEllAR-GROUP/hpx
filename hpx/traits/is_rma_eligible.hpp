//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_RMA_ELIGIBLEE_HPP
#define HPX_TRAITS_IS_RMA_ELIGIBLEE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/parcelset/rma/rma_allocator.hpp>

#include <vector>
#include <type_traits>

// A type is eligible for RMA transfers if it is a simple bitwise_serializable
// type or array/vector of bitwise_serializable types

namespace hpx { namespace traits
{

    template <typename T>
    struct is_rma_elegible
        : is_bitwise_serializable<T>
    {};

    template <>
    template <typename T>
    struct is_rma_elegible<std::vector<T, parcelset::rma::rma_allocator<T>>>
        : is_bitwise_serializable<T>
    {};

}}

#define HPX_IS_RMA_ELIGIBLE(T)                                                \
namespace hpx { namespace traits {                                            \
    template <>                                                               \
    struct is_rma_eligible< T >                                               \
      : std::true_type                                                        \
    {};                                                                       \
}}                                                                            \

#endif /*HPX_TRAITS_IS_RMA_ELIGIBLEE_HPP*/
