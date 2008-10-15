//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::memory);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION(hpx::components::server::memory::store8_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::store16_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::store32_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::store64_action);

HPX_REGISTER_ACTION(hpx::components::server::memory::load8_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::load16_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::load32_action);
HPX_REGISTER_ACTION(hpx::components::server::memory::load64_action);

///////////////////////////////////////////////////////////////////////////////
// make sure all needed action::get_action_name() functions get defined
HPX_DEFINE_GET_ACTION_NAME(hpx::lcos::base_lco_with_value<boost::uint8_t>::set_result_action);
HPX_DEFINE_GET_ACTION_NAME(hpx::lcos::base_lco_with_value<boost::uint16_t>::set_result_action);
HPX_DEFINE_GET_ACTION_NAME(hpx::lcos::base_lco_with_value<boost::uint32_t>::set_result_action);
HPX_DEFINE_GET_ACTION_NAME(hpx::lcos::base_lco_with_value<boost::uint64_t>::set_result_action);

HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<boost::uint8_t>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<boost::uint16_t>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<boost::uint32_t>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<boost::uint64_t>);

