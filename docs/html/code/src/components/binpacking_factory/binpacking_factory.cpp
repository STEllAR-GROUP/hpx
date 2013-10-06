//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/binpacking_factory/server/binpacking_factory.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::binpacking_factory binpacking_factory_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<binpacking_factory_type>,
    binpacking_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(binpacking_factory_type)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the binpacking_factory actions
HPX_REGISTER_ACTION(
    binpacking_factory_type::create_components_action,
    binpacking_factory_create_components_action)
HPX_REGISTER_ACTION(
    binpacking_factory_type::create_components_counterbased_action,
    binpacking_factory_create_components_counterbased_action)

