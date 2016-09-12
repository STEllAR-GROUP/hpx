//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/assert.hpp>

#include <cstddef>
#include <string>

#include "partition.hpp"
#include "../read_values.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate1d { namespace server
{
    partition::partition()
      : min_value_(0), max_value_(0), delta_(0)
    {}

    void partition::init(std::string datafilename, dimension const& dim,
        std::size_t num_nodes)
    {
        // store all parameters
        dim_ = dim;

        // account for necessary overlap
        std::size_t ghost_width_left = 0, ghost_width_right = 0;
        if (dim_.offset_ + dim_.count_ < dim_.size_-1)
            ++ghost_width_right;

        if (dim_.offset_ > 0)
            ++ghost_width_left;

        // extract the full data range
        extract_data_range(datafilename, min_value_, max_value_, delta_,
            dim_.offset_, dim_.offset_+dim_.count_);

        // read the slice of our data
        values_.reset(new double[dim_.count_ + ghost_width_left + ghost_width_right]);
        extract_data(datafilename, values_.get(),
            dim_.offset_ - ghost_width_left,
            dim_.count_ + ghost_width_left + ghost_width_right);
    }

    // do the actual interpolation
    double partition::interpolate(double value)
    {
        if (value < min_value_ || value > max_value_) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "partition::interpolate",
                "argument out of range");
            return 0;
        }

        std::size_t index = static_cast<std::size_t>((value - min_value_) / delta_);
        HPX_ASSERT(0 <= index && index < dim_.count_);

        return values_[index];
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef interpolate1d::server::partition partition_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION(partition_type::init_action,
    partition_init_action);
HPX_REGISTER_ACTION(partition_type::interpolate_action,
    partition_interpolate_action);

HPX_REGISTER_COMPONENT(
    hpx::components::component<partition_type>,
    interpolate1d_partition_type);
