//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARTITIONED_VECTOR_PREDEF_HPP
#define HPX_PARTITIONED_VECTOR_PREDEF_HPP

#include <hpx/components/containers/partitioned_vector/export_definitions.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_decl.hpp>

#include <string>
#include <vector>

using std_string = std::string;

HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(double);
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(std_string);
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(int);

#endif
