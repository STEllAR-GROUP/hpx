//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/particle.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    gtc::server::particle
> gtc_particle_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(gtc_particle_type, gtc_particle);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    gtc_particle_type::wrapped_type::init_action,
    gtc_particle_init_action);

HPX_REGISTER_ACTION(
    gtc_particle_type::wrapped_type::chargei_action,
    gtc_particle_chargei_action);

HPX_REGISTER_ACTION(
    gtc_particle_type::wrapped_type::distance_action,
    gtc_particle_distance_action);

HPX_REGISTER_ACTION(
    gtc_particle_type::wrapped_type::get_index_action,
    gtc_particle_get_index_action);

HPX_REGISTER_ACTION(
    gtc_particle_type::wrapped_type::get_densityi_action,
    gtc_particle_get_densityi_action);

