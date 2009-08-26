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

#include <applications/graphs/ssca2/pbreak/pbreak.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::detail::pbreak pbreak_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the parallel break LCO actions

HPX_DEFINE_GET_ACTION_NAME(pbreak_type);
HPX_REGISTER_ACTION_EX(pbreak_type::signal_action,
                       pbreak_signal_action);
HPX_REGISTER_ACTION_EX(pbreak_type::wait_action,
                       pbreak_wait_action);
HPX_DEFINE_GET_COMPONENT_TYPE(pbreak_type);
