//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace sheneos
{
    ///////////////////////////////////////////////////////////////////////////
    struct config_data
    {
        config_data() : num_instances_(0) {}

        std::string datafile_name_;     ///< Data file to load the data from.
        std::string symbolic_name_;     ///< Symbolic name this instance is registered.
        std::size_t num_instances_;     ///< Number of partition instances.
    };
}

namespace sheneos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT configuration
      : public hpx::components::component_base<configuration>
    {
    public:
        ///////////////////////////////////////////////////////////////////////
        configuration() {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality.

        /// Initialize the configuration.
        void init(std::string const& datafile, std::string const& symbolic_name,
            std::size_t num_instances);

        /// Retrieve the configuration data.
        config_data get() const;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(configuration, init);
        HPX_DEFINE_COMPONENT_ACTION(configuration, get);

    private:
        config_data data_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization.
namespace hpx { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(input_archive& ar,
        sheneos::config_data& cfg, unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(output_archive& ar,
        sheneos::config_data& cfg, unsigned int const);
}}

HPX_REGISTER_ACTION_DECLARATION(sheneos::server::configuration::init_action,
    sheneos_configuration_init_action);
HPX_REGISTER_ACTION_DECLARATION(sheneos::server::configuration::get_action,
    sheneos_configuration_get_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<sheneos::config_data>::set_value_action,
    set_value_action_sheneos_config_data);

#endif

