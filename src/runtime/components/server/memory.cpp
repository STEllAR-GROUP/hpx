//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/assert.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(hpx::components::server::memory,
    hpx::components::component_memory)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION(hpx::components::server::memory::store8_action, store8_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::store16_action, store16_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::store32_action, store32_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::store64_action, store64_action)

HPX_REGISTER_ACTION(hpx::components::server::memory::load8_action, load8_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::load16_action, load16_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::load32_action, load32_action)
HPX_REGISTER_ACTION(hpx::components::server::memory::load64_action, load64_action)

