//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/serialization.hpp>

#include <cstddef>
#include <string>

#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement the serialization functions.
    void serialize(input_archive& ar, sheneos::config_data& cfg, unsigned int const)
    {
        ar & cfg.datafile_name_ & cfg.symbolic_name_& cfg.num_instances_;
    }

    void serialize(output_archive& ar, sheneos::config_data& cfg, unsigned int const)
    {
        ar & cfg.datafile_name_ & cfg.symbolic_name_& cfg.num_instances_;
    }
}}

HPX_DISTRIBUTED_METADATA(sheneos::config_data, sheneos_config_data);
