//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>
#include <boost/assert.hpp>

#include "read_values.hpp"
#include "partition3d.hpp"
#include "interpolate3d.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();    // create entry point for component factory

///////////////////////////////////////////////////////////////////////////////
typedef interpolate3d::partition3d partition_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(partition_client_type);

typedef interpolate3d::configuration configuration_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(configuration_client_type);

///////////////////////////////////////////////////////////////////////////////
// Interpolation client
namespace interpolate3d
{
    ///////////////////////////////////////////////////////////////////////////
    // AGAS helpers
    inline void
    register_name(hpx::naming::gid_type const& gid, std::string const& name)
    {
        hpx::get_runtime().get_agas_client().registerid(name, gid);
    }

    inline void unregister_name(std::string const& name)
    {
        hpx::get_runtime().get_agas_client().unregisterid(name);
    }

    inline hpx::naming::id_type query_name(std::string const& name)
    {
        hpx::naming::gid_type gid;
        if (hpx::get_runtime().get_agas_client().queryid(name, gid))
          return hpx::naming::id_type(gid, hpx::naming::id_type::unmanaged);

        return hpx::naming::invalid_id;
    }

    // create one partition on each of the localities, initialize the partitions
    interpolate3d::interpolate3d()
      : num_partitions_per_dim_(0),
        was_created_(false)
    {
        std::memset(minval_, 0, sizeof(minval_));
        std::memset(delta_, 0, sizeof(delta_));
        std::memset(num_values_, 0, sizeof(num_values_));
    }

    interpolate3d::~interpolate3d()
    {
        if (was_created_) {
            // FIXME: This is currently a fully synchronous operation. AGAS V2
            //        needs to be extended to expose async functions before this
            //        can be fixed.

            // unregister all symbolic names
            config_data data = cfg_.get();
            unregister_name(data.symbolic_name_);   // unregister config data

            for (std::size_t i = 0; i < partitions_.size();)
            {
                unregister_name(data.symbolic_name_ +
                    boost::lexical_cast<std::string>(i++));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance
    void interpolate3d::create(std::string const& datafilename,
        std::string const& symbolic_name_base, std::size_t num_instances)
    {
        // we want to create 'partition' instances
        hpx::components::component_type type =
            hpx::components::get_component_type<server::partition3d>();

        // create distributing factory and let it create the required amount
        // of 'partition' objects
        typedef hpx::components::distributing_factory distributing_factory;

        distributing_factory factory;
        factory.create(hpx::find_here());

        distributing_factory::async_create_result_type result =
            factory.create_components_async(type, num_instances);

        // initialize the partitions and store the mappings
        partitions_.reserve(num_instances);
        fill_partitions(datafilename, symbolic_name_base, result);

        was_created_ = true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // create one partition on each of the localities, initialize the partitions
    void interpolate3d::connect(std::string symbolic_name_base)
    {
        if (symbolic_name_base[symbolic_name_base.size()-1] != '/')
            symbolic_name_base += "/";

        // FIXME: This is currently a fully synchronous operation. AGAS V2
        //        needs to be extended to expose async functions before this
        //        can be fixed.

        // connect to the config object
        cfg_ = configuration(query_name(symbolic_name_base));
        config_data data = cfg_.get();

        // reconnect to the partitions
        partitions_.reserve(data.num_instances_);
        for (std::size_t i = 0; i < data.num_instances_; ++i)
        {
            using boost::lexical_cast;
            partitions_.push_back(query_name(
                data.symbolic_name_ + lexical_cast<std::string>(i)));
        }

        // read required data from given file
        double maxval = 0;
        num_values_[dimension::x] = extract_data_range(data.datafile_name_, "x",
            minval_[dimension::x], maxval, delta_[dimension::x]);
        num_values_[dimension::y] = extract_data_range(data.datafile_name_, "y",
            minval_[dimension::y], maxval, delta_[dimension::y]);
        num_values_[dimension::z] = extract_data_range(data.datafile_name_, "z",
            minval_[dimension::z], maxval, delta_[dimension::z]);

        num_partitions_per_dim_ = static_cast<std::size_t>(
            std::exp(std::log(double(data.num_instances_)) / 3));
    }

    ///////////////////////////////////////////////////////////////////////////
    void interpolate3d::fill_partitions(std::string const& datafilename,
        std::string symbolic_name_base, async_create_result_type future)
    {
        // read required data from file
        double maxval = 0;
        num_values_[dimension::x] = extract_data_range(datafilename, "x",
            minval_[dimension::x], maxval, delta_[dimension::x]);
        num_values_[dimension::y] = extract_data_range(datafilename, "y",
            minval_[dimension::y], maxval, delta_[dimension::y]);
        num_values_[dimension::z] = extract_data_range(datafilename, "z",
            minval_[dimension::z], maxval, delta_[dimension::z]);

        // wait for the partitions to be created
        distributing_factory::result_type results = future.get();
        distributing_factory::iterator_range_type parts =
            hpx::util::locality_results(results);

        BOOST_FOREACH(hpx::naming::id_type id, parts)
            partitions_.push_back(id);

        // initialize all attached partition objects
        std::size_t num_localities = partitions_.size();
        BOOST_ASSERT(0 != num_localities);

        // cubic root
        num_partitions_per_dim_ = static_cast<std::size_t>(
            std::exp(std::log(double(num_localities)) / 3));

        std::size_t partition_size_x =
            num_values_[dimension::x] / num_partitions_per_dim_;
        std::size_t last_partition_size_x =
            num_values_[dimension::x] - partition_size_x * (num_partitions_per_dim_-1);

        std::size_t partition_size_y =
            num_values_[dimension::y] / num_partitions_per_dim_;
        std::size_t last_partition_size_y =
            num_values_[dimension::y] - partition_size_y * (num_partitions_per_dim_-1);

        std::size_t partition_size_z =
            num_values_[dimension::z] / num_partitions_per_dim_;
        std::size_t last_partition_size_z =
            num_values_[dimension::z] - partition_size_z * (num_partitions_per_dim_-1);

        dimension dim_x(num_values_[dimension::x]);
        dimension dim_y(num_values_[dimension::y]);
        dimension dim_z(num_values_[dimension::z]);

        std::vector<hpx::lcos::future<void> > lazy_sync;
        for (std::size_t x = 0; x != num_partitions_per_dim_; ++x)
        {
            dim_x.offset_ = partition_size_x * x;
            if (x == num_partitions_per_dim_-1)
                dim_x.count_ = last_partition_size_x;
            else
                dim_x.count_ = partition_size_x;

            for (std::size_t y = 0; y != num_partitions_per_dim_; ++y)
            {
                dim_y.offset_ = partition_size_y * y;
                if (y == num_partitions_per_dim_-1)
                    dim_y.count_ = last_partition_size_y;
                else
                    dim_y.count_ = partition_size_y;

                for (std::size_t z = 0; z != num_partitions_per_dim_; ++z)
                {
                    dim_z.offset_ = partition_size_z * z;
                    if (z == num_partitions_per_dim_-1)
                        dim_z.count_ = last_partition_size_z;
                    else
                        dim_z.count_ = partition_size_z;

                    std::size_t index =
                        x + (y + z * num_partitions_per_dim_) * num_partitions_per_dim_;
                    BOOST_ASSERT(index < partitions_.size());

                    lazy_sync.push_back(stubs::partition3d::init_async(
                        partitions_[index], datafilename, dim_x, dim_y, dim_z));
                }
            }
        }

        // Register symbolic names of all involved components.

        if (symbolic_name_base[symbolic_name_base.size()-1] != '/')
            symbolic_name_base += "/";

        // FIXME: This is currently a fully synchronous operation. AGAS V2
        //        needs to be extended to expose async functions before this
        //        can be fixed.
        // create the config object locally
        cfg_ = configuration(datafilename, symbolic_name_base, num_localities);
        register_name(cfg_.get_raw_gid(), symbolic_name_base);

        int i = 0;
        BOOST_FOREACH(hpx::naming::id_type const& id, partitions_)
        {
            using boost::lexical_cast;
            register_name(id.get_gid(),
                symbolic_name_base + lexical_cast<std::string>(i++));
        }

        // wait for initialization to finish
        hpx::lcos::wait(lazy_sync);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t interpolate3d::get_index(int d, double value)
    {
        std::size_t partition_size = num_values_[d] / num_partitions_per_dim_;
        std::size_t index = static_cast<std::size_t>(
            (value - minval_[d]) / (delta_[d] * partition_size));
        if (index == num_partitions_per_dim_)
            --index;
        BOOST_ASSERT(index < num_partitions_per_dim_);
        return index;
    }

    hpx::naming::id_type
    interpolate3d::get_gid(double value_x, double value_y, double value_z)
    {
        std::size_t x = get_index(dimension::x, value_x);
        std::size_t y = get_index(dimension::y, value_y);
        std::size_t z = get_index(dimension::z, value_z);

        std::size_t index =
            x + (y + z * num_partitions_per_dim_) * num_partitions_per_dim_;
        BOOST_ASSERT(index < partitions_.size());

        return partitions_[index];
    }
}

