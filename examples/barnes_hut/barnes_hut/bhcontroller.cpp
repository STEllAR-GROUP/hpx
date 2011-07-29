////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/bhcontroller.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::bhcontroller>,
    bhcontroller);

//Register the actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhcontroller::constructAction,hplConstructAction);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhcontroller::runAction,hplRunAction);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::bhcontroller);
