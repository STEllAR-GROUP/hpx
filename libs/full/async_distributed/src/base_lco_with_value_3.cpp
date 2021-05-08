//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/apply.hpp>
#include <hpx/async_distributed/base_lco_with_value.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/serialization/vector.hpp>

#include <cstdint>
#include <string>
#include <vector>

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(std::int64_t, int64_t,
    hpx::actions::base_lco_with_value_int64_get,
    hpx::actions::base_lco_with_value_int64_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(std::uint64_t, uint64_t,
    hpx::actions::base_lco_with_value_uint64_get,
    hpx::actions::base_lco_with_value_uint64_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(bool, bool,
    hpx::actions::base_lco_with_value_bool_get,
    hpx::actions::base_lco_with_value_bool_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(std::vector<bool>, vector_bool_type,
    hpx::actions::base_lco_with_value_vector_bool_get,
    hpx::actions::base_lco_with_value_vector_bool_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(std::vector<std::uint32_t>,
    vector_std_uint32_type,
    hpx::actions::base_lco_with_value_vector_std_uint32_get,
    hpx::actions::base_lco_with_value_vector_std_uint32_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(hpx::util::section, hpx_section,
    hpx::actions::base_lco_with_value_hpx_section_get,
    hpx::actions::base_lco_with_value_hpx_section_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(std::string, std_string,
    hpx::actions::base_lco_with_value_std_string_get,
    hpx::actions::base_lco_with_value_std_string_set)
