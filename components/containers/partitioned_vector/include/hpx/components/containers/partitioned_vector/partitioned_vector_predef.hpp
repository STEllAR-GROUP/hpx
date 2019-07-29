//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARTITIONED_VECTOR_PREDEF_HPP
#define HPX_PARTITIONED_VECTOR_PREDEF_HPP

#include <hpx/components/containers/partitioned_vector/export_definitions.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_decl.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_component_decl.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// declare explicitly instantiated templates

#if !defined(HPX_PARTITIONED_VECTOR_MODULE_EXPORTS)

// partitioned_vector<double>
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(double);

extern template class hpx::server::partitioned_vector<double,
    std::vector<double>>;
extern template class hpx::partitioned_vector_partition<double,
    std::vector<double>>;
extern template class hpx::partitioned_vector<double, std::vector<double>>;
extern template hpx::partitioned_vector<double,
    std::vector<double>>::partitioned_vector(size_type,
    hpx::container_distribution_policy const&, void*);
extern template hpx::partitioned_vector<double,
    std::vector<double>>::partitioned_vector(size_type, double const&,
    hpx::container_distribution_policy const&, void*);

// partitioned_vector<int>
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(int);

extern template class hpx::server::partitioned_vector<int, std::vector<int>>;
extern template class hpx::partitioned_vector_partition<int, std::vector<int>>;
extern template class hpx::partitioned_vector<int, std::vector<int>>;
extern template hpx::partitioned_vector<int,
    std::vector<int>>::partitioned_vector(size_type,
    hpx::container_distribution_policy const&, void*);
extern template hpx::partitioned_vector<int,
    std::vector<int>>::partitioned_vector(size_type, int const&,
    hpx::container_distribution_policy const&, void*);

// partitioned_vector<long long>
typedef long long long_long;
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(long_long);

extern template class hpx::server::partitioned_vector<long long, std::vector<long long>>;
extern template class hpx::partitioned_vector_partition<long long,
    std::vector<long long>>;
extern template class hpx::partitioned_vector<long long, std::vector<long long>>;
extern template hpx::partitioned_vector<long long,
    std::vector<long long>>::partitioned_vector(size_type,
    hpx::container_distribution_policy const&, void*);
extern template hpx::partitioned_vector<long long,
    std::vector<long long>>::partitioned_vector(size_type, long long const&,
    hpx::container_distribution_policy const&, void*);

// partitioned_vector<std::string>
using partitioned_vector_std_string_argument = std::string;

HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(
    partitioned_vector_std_string_argument);

extern template class hpx::server::partitioned_vector<std::string,
    std::vector<std::string>>;
extern template class hpx::partitioned_vector_partition<std::string,
    std::vector<std::string>>;
extern template class hpx::partitioned_vector<std::string,
    std::vector<std::string>>;
extern template hpx::partitioned_vector<std::string,
    std::vector<std::string>>::partitioned_vector(size_type,
    hpx::container_distribution_policy const&, void*);
extern template hpx::partitioned_vector<std::string,
    std::vector<std::string>>::partitioned_vector(size_type, std::string const&,
    hpx::container_distribution_policy const&, void*);

#endif

#endif
