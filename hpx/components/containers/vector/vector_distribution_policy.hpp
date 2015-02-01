//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_VECTOR_DISTRIBUTION_POLICY_HPP
#define HPX_VECTOR_DISTRIBUTION_POLICY_HPP

#include <hpx/include/util.hpp>
#include <hpx/components/containers/distribution_policy.hpp>

#include <type_traits>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_vector_distribution_policy
          : std::false_type
        {};

        template <>
        struct is_vector_distribution_policy<distribution_policy>
          : std::true_type
        {};
        // \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_vector_distribution_policy
      : detail::is_vector_distribution_policy<typename hpx::util::decay<T>::type>
    {};
}

#endif
