//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::int64_t, int64_t,
    hpx::actions::base_lco_with_value_int64_get,
    hpx::actions::base_lco_with_value_int64_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    boost::uint64_t, uint64_t,
    hpx::actions::base_lco_with_value_uint64_get,
    hpx::actions::base_lco_with_value_uint64_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    bool, bool,
    hpx::actions::base_lco_with_value_bool_get,
    hpx::actions::base_lco_with_value_bool_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::util::section, hpx_section,
    hpx::actions::base_lco_with_value_hpx_section_get,
    hpx::actions::base_lco_with_value_hpx_section_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std::string, std_string,
    hpx::actions::base_lco_with_value_std_string_get,
    hpx::actions::base_lco_with_value_std_string_set)
