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

#include "partition3d.hpp"
#include "../read_values.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate3d { namespace server
{
    partition3d::partition3d()
    {
        std::memset(ghost_left_, 0, sizeof(ghost_left_));
        std::memset(min_value_, 0, sizeof(min_value_));
        std::memset(max_value_, 0, sizeof(max_value_));
        std::memset(delta_, 0, sizeof(delta_));
    }

    inline void
    partition3d::init_dimension(std::string const& datafilename, int d,
        dimension const& dim, char const* name)
    {
        // store all parameters
        dim_[d] = dim;

        // account for necessary overlap
        if (dim.offset_ + dim.count_ < dim.size_-1)
            ++dim_[d].count_;

        if (dim.offset_ > 0) {
            ghost_left_[d] = 1;
            --dim_[d].offset_;
            ++dim_[d].count_;
        }

        // extract the full data range (without ghost zones)
        extract_data_range(datafilename, name, min_value_[d], max_value_[d],
            delta_[d], dim.offset_, dim.offset_ + dim.count_);
    }

    void partition3d::init(std::string const& datafilename,
        dimension const& dimx, dimension const& dimy, dimension const& dimz)
    {
        init_dimension(datafilename, dimension::x, dimx, "x");
        init_dimension(datafilename, dimension::y, dimy, "y");
        init_dimension(datafilename, dimension::z, dimz, "z");

        // read the slice of our data
        values_.reset(new double[dim_[dimension::x].count_ *
            dim_[dimension::y].count_ * dim_[dimension::z].count_]);

        extract_data(datafilename, "gauss", values_.get(),
            dim_[dimension::x], dim_[dimension::y], dim_[dimension::z]);
    }

    // do the actual interpolation
    inline std::size_t round(double d)
    {
        BOOST_ASSERT(d >= 0);
        return std::size_t(d + .5);
    }

    inline std::size_t partition3d::get_index(int d, double value)
    {
        if (value < min_value_[d] || value > max_value_[d]) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "verify_value",
                "argument out of range");
            return 0;
        }

        // FIXME: For now the index is rounded to the nearest existing sample
        //        point.
        std::size_t index = round((value - min_value_[d]) / delta_[d]) + ghost_left_[d];
        BOOST_ASSERT(index < dim_[d].count_);

        return index;
    }

    double partition3d::interpolate(double valuex, double valuey, double valuez)
    {
        std::size_t x = get_index(dimension::x, valuex);
        std::size_t y = get_index(dimension::y, valuey);
        std::size_t z = get_index(dimension::z, valuez);

        std::size_t index = x + (y + z * dim_[dimension::y].count_) * dim_[dimension::x].count_;
        BOOST_ASSERT(index < dim_[dimension::x].count_ *
            dim_[dimension::y].count_ * dim_[dimension::z].count_);

        return values_[index];
    }
}}

///////////////////////////////////////////////////////////////////////////////
typedef interpolate3d::server::partition3d partition3d_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION(partition3d_type::init_action,
    interpolate3d_partition3d_init_action);
HPX_REGISTER_ACTION(partition3d_type::interpolate_action,
    interpolate3d_partition3d_interpolate_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<partition3d_type>,
    interpolate3d_partition_type);
