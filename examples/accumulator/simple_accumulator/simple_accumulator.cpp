//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/components/simple_accumulator/server/simple_accumulator.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::simple_accumulator>, 
    simple_accumulator);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the simple_accumulator actions
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::init_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::add_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::query_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::print_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::simple_accumulator);

