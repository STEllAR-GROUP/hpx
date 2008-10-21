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

#include <hpx/components/amr/server/functional_component_1.hpp>
#include <hpx/components/amr/server/functional_component_3.hpp>
#include <hpx/components/amr/server/functional_component_5.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef 
    hpx::components::amr::server::functional_component<1> 
functional_component_double_1_type;
typedef 
    hpx::components::amr::server::functional_component<3> 
functional_component_double_3_type;
typedef 
    hpx::components::amr::server::functional_component<5> 
functional_component_double_5_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION(functional_component_double_1_type::eval_action);
HPX_REGISTER_ACTION(functional_component_double_1_type::is_last_timestep_action);
HPX_DEFINE_GET_COMPONENT_TYPE(functional_component_double_1_type);

HPX_REGISTER_ACTION(functional_component_double_3_type::eval_action);
HPX_REGISTER_ACTION(functional_component_double_3_type::is_last_timestep_action);
HPX_DEFINE_GET_COMPONENT_TYPE(functional_component_double_3_type);

HPX_REGISTER_ACTION(functional_component_double_5_type::eval_action);
HPX_REGISTER_ACTION(functional_component_double_5_type::is_last_timestep_action);
HPX_DEFINE_GET_COMPONENT_TYPE(functional_component_double_5_type);

