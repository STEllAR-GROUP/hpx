//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "../../applications/graphs/ssca2/ssca2/ssca2.hpp"

#include <hpx/components/distributed_set/server/local_set.hpp>
#include "local_set.hpp"

///////////////////////////////////////////////////////////////////////////////
// additional action definitions required for returning List

typedef hpx::components::server::ssca2::edge_set_type edge_set_type;

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<edge_set_type>::set_result_action,
    set_result_action_vector_edge_type);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<edge_set_type>);
