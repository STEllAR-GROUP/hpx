//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1220PM

#include <hpx/hpx.hpp>

#include <cstddef>
#include <string>

namespace sheneos
{
    ///////////////////////////////////////////////////////////////////////////
    struct config_data
    {
        config_data()
          : num_instances_(0)
        {}

        config_data(std::string const& datafile_name,
                std::string const& symbolic_name, std::size_t num_instances)
          : datafile_name_(datafile_name),
            symbolic_name_(symbolic_name),
            num_instances_(num_instances)
        {}

        std::string datafile_name_;     // Data file to load the data from.
        std::string symbolic_name_;     // Symbolic name this instance is
                                        // registered with.
        std::size_t num_instances_;     // Number of partition instances.
    };
}

HPX_DISTRIBUTED_METADATA_DECLARATION(sheneos::config_data, sheneos_config_data);

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization.
namespace hpx { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(input_archive& ar, sheneos::config_data& cfg, unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(output_archive& ar, sheneos::config_data& cfg, unsigned int const);
}}

#endif

