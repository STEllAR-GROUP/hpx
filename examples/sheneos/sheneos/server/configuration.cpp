//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <hpx/util/assert.hpp>

#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos { namespace server
{
    void configuration::init(std::string const& datafilename,
        std::string const& symbolic_name, std::size_t num_instances)
    {
        data_.datafile_name_ = datafilename;
        data_.symbolic_name_ = symbolic_name;
        data_.num_instances_ = num_instances;
    }

    config_data configuration::get() const
    {
        return data_;
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement the serialization functions.
    void serialize(input_archive& ar,
        sheneos::config_data& cfg, unsigned int const)
    {
        ar & cfg.datafile_name_ & cfg.symbolic_name_& cfg.num_instances_;
    }

    void serialize(output_archive& ar,
        sheneos::config_data& cfg, unsigned int const)
    {
        ar & cfg.datafile_name_ & cfg.symbolic_name_& cfg.num_instances_;
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef sheneos::server::configuration configuration_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions.
HPX_REGISTER_ACTION(configuration_type::init_action,
    sheneos_configuration_init_action);
HPX_REGISTER_ACTION(configuration_type::get_action,
    sheneos_configuration_get_action);

HPX_REGISTER_COMPONENT(
    hpx::components::component<configuration_type>,
    sheneos_configuration_type);

HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<sheneos::config_data>::set_value_action,
    set_value_action_sheneos_config_data);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<sheneos::config_data>,
    hpx::components::component_base_lco_with_value);

