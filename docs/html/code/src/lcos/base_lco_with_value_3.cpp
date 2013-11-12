
//  Copyright (c) 2007-2012 Hartmut Kaiser
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

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

HPX_REGISTER_BASE_LCO_WITH_VALUE(boost::int64_t, int64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE(boost::uint64_t, uint64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE(bool, bool)
HPX_REGISTER_BASE_LCO_WITH_VALUE(hpx::util::section, section)
HPX_REGISTER_BASE_LCO_WITH_VALUE(std::string, std_string)
