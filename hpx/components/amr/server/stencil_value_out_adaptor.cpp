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

#include <hpx/components/amr/server/stencil_value_out_adaptor.hpp>

///////////////////////////////////////////////////////////////////////////////
/// Define types of stencil_value_out_adaptor components exposed by this module
typedef hpx::components::managed_component<
    hpx::components::amr::server::stencil_value_out_adaptor<double>
> stencil_value_out_adaptor_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    stencil_value_out_adaptor_type, stencil_value_out_adaptor);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(stencil_value_out_adaptor_type::wrapped_type::get_value_action);
HPX_DEFINE_GET_COMPONENT_TYPE(stencil_value_out_adaptor_type::wrapped_type);

