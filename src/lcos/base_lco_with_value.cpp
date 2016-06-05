//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/base_lco_with_value.hpp>

#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/unused.hpp>

#include <vector>

namespace hpx { namespace lcos
{
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::naming::gid_type, gid_type,
    hpx::actions::base_lco_with_value_gid_get,
    hpx::actions::base_lco_with_value_gid_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std::vector<hpx::naming::gid_type>, vector_gid_type,
    hpx::actions::base_lco_with_value_vector_gid_get,
    hpx::actions::base_lco_with_value_vector_gid_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::naming::id_type, id_type,
    hpx::actions::base_lco_with_value_id_get,
    hpx::actions::base_lco_with_value_id_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std::vector<hpx::naming::id_type>, vector_id_type,
    hpx::actions::base_lco_with_value_vector_id_get,
    hpx::actions::base_lco_with_value_vector_id_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::util::unused_type, unused_type,
    hpx::actions::base_lco_with_value_unused_get,
    hpx::actions::base_lco_with_value_unused_set)
