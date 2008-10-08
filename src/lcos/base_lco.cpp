//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the future actions
HPX_REGISTER_ACTION(hpx::lcos::base_lco::set_event_action);

HPX_REGISTER_ACTION(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_result_action);
HPX_REGISTER_ACTION(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_error_action);
HPX_REGISTER_ACTION(hpx::lcos::base_lco_with_value<double>::set_result_action);
HPX_REGISTER_ACTION(hpx::lcos::base_lco_with_value<double>::set_error_action);

