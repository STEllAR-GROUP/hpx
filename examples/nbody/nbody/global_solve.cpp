//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/global_solve.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    examples::server::global_solve> nbody_global_solve_type;

namespace boost { namespace serialization{
    ///////////////////////////////////////////////////////////////////////////
    // Implement the serialization functions.
    template <typename Archive>
    void serialize(Archive& ar, examples::server::particle& part,
        unsigned int const){
        ar & part.mass & part.p[0] & part.p[1] & part.p[2]
           & part.v[0] & part.v[1] & part.v[2];
    }

    ///////////////////////////////////////////////////////////////////////////
    // Explicit instantiation for the correct archive types.
    template HPX_COMPONENT_EXPORT void 
    serialize(hpx::util::portable_binary_iarchive&, examples::server::particle&, 
        unsigned int const);
    template HPX_COMPONENT_EXPORT void 
    serialize(hpx::util::portable_binary_oarchive&, examples::server::particle&, 
        unsigned int const);
}}

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(nbody_global_solve_type, global_solve_t);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for global_solve actions.
HPX_REGISTER_ACTION(
    nbody_global_solve_type::wrapped_type::init_action,
    global_solve_init_action);
HPX_REGISTER_ACTION(
    nbody_global_solve_type::wrapped_type::run_action,
    global_solve_run_action);
HPX_REGISTER_ACTION(
    nbody_global_solve_type::wrapped_type::report_action,
    global_solve_report_action);
HPX_REGISTER_ACTION(
    nbody_global_solve_type::wrapped_type::calc_action,
    global_solve_calculate_action);
//HPX_DEFINE_GET_COMPONENT_TYPE(nbody_global_solve_type::wrapped_type);
//]

