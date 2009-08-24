//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/distributed_map/server/distributed_map.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
// additional action definitions required for returning vector of GIDs to GIDs

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::set_result_action,
    set_result_action_vector_id_type);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >);
