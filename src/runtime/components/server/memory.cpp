//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store8_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store16_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store32_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::store64_action);

HPX_SERIALIZE_ACTION(hpx::components::server::memory::load8_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::load16_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::load32_action);
HPX_SERIALIZE_ACTION(hpx::components::server::memory::load64_action);

