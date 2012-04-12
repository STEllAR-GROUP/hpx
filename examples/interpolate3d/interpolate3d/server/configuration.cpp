//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>

#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate3d { namespace server
{
    configuration::configuration()
    {
    }

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
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    template <typename Archive>
    void serialize(Archive& ar, interpolate3d::config_data& cfg, unsigned int const)
    {
        ar & cfg.datafile_name_ & cfg.symbolic_name_& cfg.num_instances_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive&, interpolate3d::config_data&,
        unsigned int const);
    template HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive&, interpolate3d::config_data&,
        unsigned int const);
}}

///////////////////////////////////////////////////////////////////////////////
typedef interpolate3d::server::configuration configuration_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(configuration_type::init_action,
    interpolate3d_configuration_init_action);
HPX_REGISTER_ACTION_EX(configuration_type::get_action,
    interpolate3d_configuration_get_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<configuration_type>,
    interpolate3d_configuration_type);
HPX_DEFINE_GET_COMPONENT_TYPE(configuration_type);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<interpolate3d::config_data>::set_value_action,
    set_value_action_interpolate3d_config_data);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<interpolate3d::config_data>,
    hpx::components::component_base_lco_with_value);
