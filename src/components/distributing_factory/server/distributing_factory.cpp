//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/move/move.hpp>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    struct lazy_result
    {
        lazy_result(naming::gid_type const& locality_id)
          : locality_(locality_id)
        {}

        naming::gid_type locality_;
        lcos::future<std::vector<naming::gid_type > > gids_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a component
    distributing_factory::remote_result_type
    distributing_factory::create_components(
        components::component_type type, std::size_t count) const
    {
        // make sure we get localities for derived component type, if any
        components::component_type prefix_type = type;
        if (type != components::get_base_type(type))
            prefix_type = components::get_derived_type(type);

        // get list of locality prefixes
        std::vector<naming::id_type> localities =
            hpx::find_all_localities(prefix_type);

        if (localities.empty())
        {
            HPX_THROW_EXCEPTION(bad_component_type,
                "distributing_factory::create_components",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        if (count == std::size_t(-1))
            count = localities.size();

        std::size_t created_count = 0;
        std::size_t count_on_locality = count / localities.size();
        std::size_t excess = count - count_on_locality*localities.size();

        // distribute the number of components to create evenly over all
        // available localities
        typedef std::vector<lazy_result> future_values_type;
        typedef server::runtime_support::bulk_create_components_action
            action_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;

        BOOST_FOREACH(naming::id_type const& fact, localities)
        {
            std::size_t numcreate = count_on_locality;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }

            if (created_count + numcreate > count)
                numcreate = count - created_count;

            if (numcreate == 0)
                break;

            // create components for each locality in one go
            v.push_back(future_values_type::value_type(fact.get_gid()));
            lcos::packaged_action<action_type, std::vector<naming::gid_type> > p;
            p.apply(fact, type, numcreate);
            v.back().gids_ = p.get_future();

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        remote_result_type results;

        BOOST_FOREACH(lazy_result& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.locality_, type));
            results.back().gids_ = boost::move(lr.gids_.move());
        }

        return results;
    }

    ///////////////////////////////////////////////////////////////////////////
    distributing_factory::remote_result_type
    distributing_factory::create_partitioned(components::component_type type,
        std::size_t count, std::size_t parts, partition_info const& info) const
    {
        // make sure we get prefixes for derived component type, if any
        components::component_type prefix_type = type;
        if (type != components::get_base_type(type))
            prefix_type = components::get_derived_type(type);

        // get list of locality prefixes
        std::vector<naming::id_type> localities =
            hpx::find_all_localities(prefix_type);

        if (localities.empty())
        {
            // no locality supports creating the requested component type
            HPX_THROW_EXCEPTION(bad_component_type,
                "distributing_factory::create_partitioned",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        std::size_t part_size = info.size();
        if (part_size < localities.size())
        {
            // we have less localities than required by one partition
            HPX_THROW_EXCEPTION(bad_parameter,
                "distributing_factory::create_partitioned",
                "partition size is larger than number of localities");
        }

        // a new partition starts every parts_delta localities
        std::size_t parts_delta = 0;
        if (localities.size() > part_size)
            parts_delta = localities.size() / part_size;

        // distribute the number of components to create evenly over all
        // available localities
        typedef std::vector<lazy_result> future_values_type;
        typedef server::runtime_support::bulk_create_components_action
            action_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;

        for (std::size_t i = 0, j = 0;
             i < localities.size() && j < parts;
             i += parts_delta, ++j)
        {
            // create components for each locality in one go, overall, 'count'
            // components for each partition
            v.push_back(future_values_type::value_type(localities[i].get_gid()));
            lcos::packaged_action<action_type, std::vector<naming::gid_type> > p;
            p.apply(localities[i], type, count);
            v.back().gids_ = p.get_future();
        }

        // now wait for the results
        remote_result_type results;

        BOOST_FOREACH(lazy_result& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.locality_, type));
            results.back().gids_ = boost::move(lr.gids_.move());
        }

        return results;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive& ar,
        hpx::components::server::partition_info& info, unsigned int const)
    {
        ar & info.dims_ & info.dim_sizes_;
    }

    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive& ar,
        hpx::components::server::partition_info& info, unsigned int const)
    {
        ar & info.dims_ & info.dim_sizes_;
    }
}}

