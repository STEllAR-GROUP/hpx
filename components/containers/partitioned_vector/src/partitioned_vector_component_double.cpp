//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/components/containers/container_distribution_policy.hpp>
#include <hpx/components/containers/partitioned_vector/export_definitions.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_component.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector.hpp>

#include <vector>

HPX_REGISTER_PARTITIONED_VECTOR(double);

// an out-of-line definition of a member of a class template cannot have default
// arguments
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 5037)
#endif

template class HPX_PARTITIONED_VECTOR_EXPORT
    hpx::server::partitioned_vector<double, std::vector<double>>;
template class HPX_PARTITIONED_VECTOR_EXPORT
    hpx::partitioned_vector_partition<double, std::vector<double>>;
template class HPX_PARTITIONED_VECTOR_EXPORT
    hpx::partitioned_vector<double, std::vector<double>>;
template HPX_PARTITIONED_VECTOR_EXPORT
    hpx::partitioned_vector<double, std::vector<double>>::partitioned_vector(
        size_type, hpx::container_distribution_policy const&, void*);
template HPX_PARTITIONED_VECTOR_EXPORT
    hpx::partitioned_vector<double, std::vector<double>>::partitioned_vector(
        size_type, double const&, hpx::container_distribution_policy const&,
        void*);

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

