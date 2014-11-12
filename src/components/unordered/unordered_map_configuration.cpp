//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/components/unordered/unordered_map_configuration.hpp>

#include <boost/serialization/unordered_map.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement the serialization functions.
    void serialize(hpx::util::portable_binary_iarchive& ar,
        hpx::server::unordered_map_configuration::partition_data& pd,
        unsigned int const)
    {
        ar & pd.partition_ & pd.size_ & pd.locality_id_;
    }
    void serialize(hpx::util::portable_binary_oarchive& ar,
        hpx::server::unordered_map_configuration::partition_data& pd,
        unsigned int const)
    {
        ar & pd.partition_ & pd.size_ & pd.locality_id_;
    }

    void serialize(hpx::util::portable_binary_iarchive& ar,
        hpx::server::unordered_map_configuration::config_data& cfg,
        unsigned int const)
    {
        ar & cfg.size_ & cfg.block_size_ & cfg.partitions_ & cfg.policy_;
    }
    void serialize(hpx::util::portable_binary_oarchive& ar,
        hpx::server::unordered_map_configuration::config_data& cfg,
        unsigned int const)
    {
        ar & cfg.size_ & cfg.block_size_ & cfg.partitions_ & cfg.policy_;
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::server::unordered_map_configuration configuration_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions.
HPX_REGISTER_ACTION(
    hpx::server::unordered_map_configuration::get_action,
    vector_configuration_get_action);

HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<
            hpx::server::unordered_map_configuration::config_data
        >::set_value_action,
    set_value_action_unordered_map_config_data);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<
        hpx::server::unordered_map_configuration::config_data>,
    hpx::components::component_base_lco_with_value);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<configuration_type>,
    unordered_map_configuration_type);


