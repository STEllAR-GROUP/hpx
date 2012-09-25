//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/partition.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    gtcx::server::partition
> gtcx_partition_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(gtcx_partition_type, gtcx_partition);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::loop_action,
    gtcx_partition_loop_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_data_action,
    gtcx_partition_set_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_tdata_action,
    gtcx_partition_set_tdata_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_params_action,
    gtcx_partition_set_params_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_comm_allreduce_data_action,
    gtcx_partition_set_comm_allreduce_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_int_comm_allreduce_data_action,
    gtcx_partition_set_int_comm_allreduce_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_sndrecv_data_action,
    gtcx_partition_set_sndrecv_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_ntoroidal_gather_data_action,
    gtcx_partition_set_ntoroidal_gather_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_ntoroidal_scatter_data_action,
    gtcx_partition_set_ntoroidal_scatter_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_complex_ntoroidal_gather_data_action,
    gtcx_partition_set_complex_ntoroidal_gather_data_action);

HPX_REGISTER_ACTION_EX(
    gtcx_partition_type::wrapped_type::set_toroidal_reduce_data_action,
    gtcx_partition_set_toroidal_reduce_data_action);
