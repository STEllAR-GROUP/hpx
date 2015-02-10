//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/vector.hpp>

#include <hpx/util/assert.hpp>

#include <hpx/components/vector/vector_configuration.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement the serialization functions.
    void serialize(input_archive& ar,
        hpx::server::vector_configuration::partition_data& pd, unsigned int const)
    {
        ar & pd.partition_ & pd.size_ & pd.locality_id_;
    }
    void serialize(output_archive& ar,
        hpx::server::vector_configuration::partition_data& pd, unsigned int const)
    {
        ar & pd.partition_ & pd.size_ & pd.locality_id_;
    }

    void serialize(input_archive& ar,
        hpx::server::vector_configuration::config_data& cfg, unsigned int const)
    {
        ar & cfg.size_ & cfg.block_size_ & cfg.partitions_ & cfg.policy_;
    }
    void serialize(output_archive& ar,
        hpx::server::vector_configuration::config_data& cfg, unsigned int const)
    {
        ar & cfg.size_ & cfg.block_size_ & cfg.partitions_ & cfg.policy_;
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::server::vector_configuration configuration_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions.
HPX_REGISTER_ACTION(
    hpx::server::vector_configuration::get_action,
    vector_configuration_get_action);

HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<
            hpx::server::vector_configuration::config_data
        >::set_value_action,
    set_value_action_vector_config_data);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<
        hpx::server::vector_configuration::config_data>,
    hpx::components::component_base_lco_with_value);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<configuration_type>,
    vector_configuration_type);


