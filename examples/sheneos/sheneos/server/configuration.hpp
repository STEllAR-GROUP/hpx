//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include "../dimension.hpp"

namespace sheneos 
{
    ///////////////////////////////////////////////////////////////////////////
    struct config_data
    {
        config_data() 
          : num_instances_(0) 
        {
            std::memset(minval_, 0, sizeof(minval_));
            std::memset(maxval_, 0, sizeof(maxval_));
            std::memset(delta_, 0, sizeof(delta_));
            std::memset(num_values_, 0, sizeof(num_values_));
        }

        std::string datafile_name_;     // data file to load the data from
        std::string symbolic_name_;     // symbolic name this instance is registered
        std::size_t num_instances_;     // number of partition instances

        double minval_[dimension::dim]; // minimal possible values in queries
        double maxval_[dimension::dim]; // maximum possible values in queries
        double delta_[dimension::dim];  // distance between existing datapoints
        std::size_t num_values_[dimension::dim];    // number datapoints
    };
}

namespace sheneos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT configuration
      : public hpx::components::simple_component_base<configuration>
    {
    private:
        typedef hpx::components::simple_component_base<configuration> base_type;

    public:
        configuration();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef configuration wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            configuration_init = 0,
            configuration_get = 1
        };

        // exposed functionality
        void init(config_data const& data);
        config_data get() const;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            configuration, configuration_init, 
            config_data const&, &configuration::init
        > init_action;

        typedef hpx::actions::result_action0<
            configuration const, config_data, configuration_get, 
            &configuration::get
        > get_action;

    private:
        sheneos::config_data data_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
// non-intrusive serialization
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive&, sheneos::config_data&, unsigned int const);
}}

#endif
