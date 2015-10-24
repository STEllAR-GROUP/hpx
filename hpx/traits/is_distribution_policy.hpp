//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_DISTRIBUTION_POLICY_APR_07_2015_0412PM)
#define HPX_TRAITS_IS_DISTRIBUTION_POLICY_APR_07_2015_0412PM

#include <hpx/traits.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    template <typename T, typename Enable>
    struct is_distribution_policy
      : std::false_type
    {};
}}

#endif
