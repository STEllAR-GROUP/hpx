//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "reduce_max.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::detail::reduce_max reduce_max_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the reduce_max LCO actions

// These could go somewhere else ... but where?
HPX_DEFINE_GET_ACTION_NAME(reduce_max_type);
HPX_REGISTER_ACTION_EX(reduce_max_type::signal_action,
                       reduce_max_signal_action);
HPX_REGISTER_ACTION_EX(reduce_max_type::wait_action,
                       reduce_max_wait_action);
HPX_DEFINE_GET_COMPONENT_TYPE(reduce_max_type);
