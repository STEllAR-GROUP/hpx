//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <cstddef>
#include <string>
#include <utility>

#include "read_values.hpp"
#include "partition.hpp"
#include "interpolate1d.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();    // create entry point for component factory

///////////////////////////////////////////////////////////////////////////////
typedef interpolate1d::partition partition_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(partition_client_type);

///////////////////////////////////////////////////////////////////////////////
// Interpolation client
namespace interpolate1d
{
    // create one partition on each of the localities, initialize the partitions
    interpolate1d::interpolate1d(std::string datafilename,
            std::size_t num_instances)
      : num_elements_(0), minval_(0), delta_(0)
    {
        // we want to create 'partition' instances
        hpx::future<std::vector<partition> > result = hpx::new_<partition[]>(
            hpx::default_layout(hpx::find_all_localities()), num_instances);

        // initialize the partitions and store the mappings
        fill_partitions(datafilename, std::move(result));
    }

    void interpolate1d::fill_partitions(std::string const& datafilename,
        hpx::future<std::vector<partition> > && future)
    {
        // read required data from file
        double maxval = 0;
        num_elements_ = extract_data_range(datafilename, minval_, maxval, delta_);

        // initialize the partitions
        partitions_ = future.get();

        std::size_t num_localities = partitions_.size();
        HPX_ASSERT(0 != num_localities);

        std::size_t partition_size = num_elements_ / num_localities;
        std::size_t last_partition_size =
            num_elements_ - partition_size * (num_localities-1);

        for (std::size_t i = 0; i != num_localities; ++i)
        {
            dimension dim;
            if (i == num_localities-1) {
                dim.offset_ = partition_size * i;
                dim.count_ = last_partition_size;
                dim.size_ = num_elements_;
            }
            else {
                dim.offset_ = partition_size * i;
                dim.count_ = partition_size;
                dim.size_ = num_elements_;
            }
            partitions_[i].init(datafilename, dim, num_localities);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    partition interpolate1d::get_partition(double value) const
    {
        std::size_t partition_size = num_elements_ / partitions_.size();
        std::size_t index = static_cast<std::size_t>(
            (value - minval_) / (delta_ * partition_size));

        if (index == partitions_.size())
            --index;
        HPX_ASSERT(index < partitions_.size());

        return partitions_[index];
    }
}

