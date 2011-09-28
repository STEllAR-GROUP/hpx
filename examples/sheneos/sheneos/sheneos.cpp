//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include "sheneos.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();    // create entry point for component factory

///////////////////////////////////////////////////////////////////////////////
typedef sheneos::partition3d partition_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(partition_client_type);

typedef sheneos::configuration configuration_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(configuration_client_type);

///////////////////////////////////////////////////////////////////////////////
// Interpolation client
namespace sheneos
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
    sheneos::sheneos()
      : num_partitions_per_dim_(0), 
        was_created_(false)
    {
        std::memset(minval_, 0, sizeof(minval_));
        std::memset(maxval_, 0, sizeof(maxval_));
        std::memset(delta_, 0, sizeof(delta_));
        std::memset(num_values_, 0, sizeof(num_values_));
    }

    sheneos::~sheneos()
    {
        if (was_created_) {
            // FIXME: This is currently a fully synchronous operation. AGAS V2 
            //        needs to be extended to expose async functions before this 
            //        can be fixed.

            // unregister all symbolic names
            config_data data = cfg_.get();
            unregister_name(data.symbolic_name_);   // unregister config data

            for (std::size_t i = 0; i < partitions_.size(); ++i)
            {
                unregister_name(data.symbolic_name_ + 
                    boost::lexical_cast<std::string>(i++));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance
    void sheneos::create(std::string const& datafilename,
        std::string const& symbolic_name_base, std::size_t num_instances)
    {
        // we want to create 'partition' instances
        hpx::components::component_type type = 
            hpx::components::get_component_type<server::partition3d>();

        // create distributing factory and let it create the required amount
        // of 'partition' objects
        typedef hpx::components::distributing_factory distributing_factory;

        distributing_factory factory(
            distributing_factory::create_sync(hpx::find_here()));
        distributing_factory::async_create_result_type result = 
            factory.create_components_async(type, num_instances);

        // initialize the partitions and store the mappings
        partitions_.reserve(num_instances);
        fill_partitions(datafilename, symbolic_name_base, result);

        was_created_ = true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // create one partition on each of the localities, initialize the partitions
    void sheneos::connect(std::string symbolic_name_base)
    {
        // FIXME: This is currently a fully synchronous operation. AGAS V2 
        //        needs to be extended to expose async functions before this 
        //        can be fixed.

        // connect to the config object 
        cfg_ = configuration(query_name(symbolic_name_base));
        config_data data = cfg_.get();

        if (data.symbolic_name_[data.symbolic_name_.size()-1] != '/')
            data.symbolic_name_ += "/";

        // reconnect to the partitions
        partitions_.reserve(data.num_instances_);
        for (int i = 0; i < data.num_instances_; ++i)
        {
            using boost::lexical_cast;
            partitions_.push_back(query_name(
                data.symbolic_name_ + lexical_cast<std::string>(i)));
        }

        // read required data from given file
        num_values_[dimension::ye] = extract_data_range(data.datafile_name_, 
            "ye", minval_[dimension::ye], maxval_[dimension::ye], delta_[dimension::ye]);
        num_values_[dimension::temp] = extract_data_range(data.datafile_name_, 
            "logtemp", minval_[dimension::temp], maxval_[dimension::temp], delta_[dimension::temp]);
        num_values_[dimension::rho] = extract_data_range(data.datafile_name_, 
            "logrho", minval_[dimension::rho], maxval_[dimension::rho], delta_[dimension::rho]);

        num_partitions_per_dim_ = std::exp(std::log(double(data.num_instances_)) / 3);
    }

    ///////////////////////////////////////////////////////////////////////////
    void sheneos::fill_partitions(std::string const& datafilename,
        std::string symbolic_name_base, async_create_result_type future)
    {
        // read required data from file
        num_values_[dimension::ye] = extract_data_range(datafilename, 
            "ye", minval_[dimension::ye], maxval_[dimension::ye], delta_[dimension::ye]);
        num_values_[dimension::temp] = extract_data_range(datafilename, 
            "logtemp", minval_[dimension::temp], maxval_[dimension::temp], delta_[dimension::temp]);
        num_values_[dimension::rho] = extract_data_range(datafilename, 
            "logrho", minval_[dimension::rho], maxval_[dimension::rho], delta_[dimension::rho]);

        // wait for the partitions to be created
        distributing_factory::result_type results = future.get();
        distributing_factory::iterator_range_type parts = 
            hpx::components::server::locality_results(results);

        BOOST_FOREACH(hpx::naming::id_type id, parts) 
            partitions_.push_back(id);

        // initialize all attached partition objects
        std::size_t num_localities = partitions_.size();
        BOOST_ASSERT(0 != num_localities);

        // cubic root
        num_partitions_per_dim_ = std::exp(std::log(double(num_localities)) / 3);

        std::size_t partition_size_x = 
            num_values_[dimension::ye] / num_partitions_per_dim_;
        std::size_t last_partition_size_x = 
            num_values_[dimension::ye] - partition_size_x * (num_partitions_per_dim_-1);

        std::size_t partition_size_y = 
            num_values_[dimension::temp] / num_partitions_per_dim_;
        std::size_t last_partition_size_y = 
            num_values_[dimension::temp] - partition_size_y * (num_partitions_per_dim_-1);

        std::size_t partition_size_z = 
            num_values_[dimension::rho] / num_partitions_per_dim_;
        std::size_t last_partition_size_z = 
            num_values_[dimension::rho] - partition_size_z * (num_partitions_per_dim_-1);

        dimension dim_x(num_values_[dimension::ye]);
        dimension dim_y(num_values_[dimension::temp]);
        dimension dim_z(num_values_[dimension::rho]);

        std::vector<hpx::lcos::promise<void> > lazy_sync;
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

        // FIXME: This is currently a fully synchronous operation. AGAS V2 
        //        needs to be extended to expose async functions before this 
        //        can be fixed.
        // create the config object locally
        cfg_ = configuration(datafilename, symbolic_name_base, num_localities);
        register_name(cfg_.get_raw_gid(), symbolic_name_base);

        if (symbolic_name_base[symbolic_name_base.size()-1] != '/')
            symbolic_name_base += "/";

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
    std::size_t sheneos::get_partition_index(int d, double value)
    {
        std::size_t partition_size = num_values_[d] / num_partitions_per_dim_;
        std::size_t partition_index = (value - minval_[d]) / (delta_[d] * partition_size);
        if (partition_index == num_partitions_per_dim_) 
            --partition_index;
        BOOST_ASSERT(partition_index < num_partitions_per_dim_);
        return partition_index;
    }

    hpx::naming::id_type 
    sheneos::get_gid(double ye, double temp, double rho)
    {
        std::size_t x = get_partition_index(dimension::ye, ye);
        std::size_t y = get_partition_index(dimension::temp, std::log10(temp));
        std::size_t z = get_partition_index(dimension::rho, std::log10(rho));

        std::size_t index = 
            x + (y + z * num_partitions_per_dim_) * num_partitions_per_dim_;
        BOOST_ASSERT(index < partitions_.size());

        return partitions_[index];
    }

    // return the description for the given dimension 
    void sheneos::get_dimension(int what, double& min, double& max)
    {
        if (what < 0 || what > dimension::dim) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, 
                "sheneos::get_dimension",
                "value of parameter 'what' is not valid");
        }

        switch (what) {
        case dimension::ye:
            min = minval_[dimension::ye];
            max = maxval_[dimension::ye];
            break;

        case dimension::temp:
        case dimension::rho:
            min = std::pow(10., minval_[what]);
            max = std::pow(10., maxval_[what]);
            break;
        }
    }
}

