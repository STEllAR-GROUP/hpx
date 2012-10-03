//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/components/binpacking_factory/server/binpacking_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>
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
        lcos::future<std::vector<naming::gid_type> > gids_;
    };

    ///////////////////////////////////////////////////////////////////////////
    binpacking_factory::remote_result_type
    binpacking_factory::create_components(components::component_type type,
        std::size_t count) const
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
                "binpacking_factory::create_components_binpacked",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        if (count == std::size_t(-1))
            count = localities.size();

        // retrieve the current number of instances of the given component
        std::vector<lcos::future<long> > lazy_counts;
        BOOST_FOREACH(naming::id_type const& id, localities)
        {
            lazy_counts.push_back(
                stubs::runtime_support::get_instance_count_async(id, type));
        }

        // wait for counts to get back, collect statistics
        long maxcount = 0;
        long existing = 0;

        std::vector<long> counts;
        counts.reserve(lazy_counts.size());
        BOOST_FOREACH(lcos::future<long> const& f, lazy_counts)
        {
            counts.push_back(f.get());
            maxcount = (std::max)(maxcount, counts.back());
            existing += counts.back();
        }

        // distribute the number of components to create in a way, so that the
        // overall number of component instances on all localities is
        // approximately the same
        BOOST_ASSERT(maxcount * counts.size() >= std::size_t(existing));
        std::size_t missing = maxcount * counts.size() - existing;
        if (missing == 0) missing = 1;

        double hole_ratio = (std::min)(count, missing) / double(missing);
        BOOST_ASSERT(hole_ratio <= 1.);

        std::size_t overflow_count =
            (count > missing) ? (count - missing) / counts.size() : 0;
        std::size_t excess = count - overflow_count * counts.size();

        typedef std::vector<lazy_result> future_values_type;
        typedef server::runtime_support::bulk_create_components_action
            action_type;

        std::size_t created_count = 0;
        future_values_type v;

        // start an asynchronous operation for each of the localities
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            std::size_t numcreate =
                std::size_t((maxcount - counts[i]) * hole_ratio) + overflow_count;

            if (excess != 0) {
                --excess;
                ++numcreate;
            }

            if (i == counts.size()-1) {
                // last bin gets all the rest
                if (created_count + numcreate < count)
                    numcreate = count - created_count;
            }

            if (created_count + numcreate > count)
                numcreate = count - created_count;

            if (numcreate == 0)
                break;

            // create one component at a time
            v.push_back(future_values_type::value_type(localities[i].get_gid()));
            lcos::packaged_action<action_type, std::vector<naming::gid_type> > p;
            p.apply(localities[i], type, numcreate);
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
    binpacking_factory::remote_result_type
    binpacking_factory::create_components_counterbased(
        components::component_type type, std::size_t count,
        std::string const& countername) const
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
                "binpacking_factory::create_components_binpacked",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        if (count == std::size_t(-1))
            count = localities.size();

        // create performance counters on all localities
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_type_path_elements(countername, p);

        // FIXME: make loop asynchronous
        typedef lcos::future<naming::id_type, naming::id_type> future_type;

        std::vector<future_type> lazy_counts;
        BOOST_FOREACH(naming::id_type const& id, localities)
        {
            std::string name;
            p.parentinstanceindex_ = naming::get_locality_id_from_id(id);
            performance_counters::get_counter_name(p, name);
            lazy_counts.push_back(performance_counters::get_counter(name));
        }

        // wait for counts to get back, collect statistics
        long maxcount = 0;
        long existing = 0;

        // FIXME: make loop asynchronous
        std::vector<long> counts;
        counts.reserve(lazy_counts.size());
        BOOST_FOREACH(future_type const& f, lazy_counts)
        {
            performance_counters::counter_value value =
                performance_counters::stubs::performance_counter::get_value(f.get());
            counts.push_back(value.get_value<long>());
            maxcount = (std::max)(maxcount, counts.back());
            existing += counts.back();
        }

        // distribute the number of components to create in a way, so that the
        // overall number of component instances on all localities is
        // approximately the same
        BOOST_ASSERT(maxcount * counts.size() >= std::size_t(existing));
        std::size_t missing = maxcount * counts.size() - existing;
        if (missing == 0) missing = 1;

        double hole_ratio = (std::min)(count, missing) / double(missing);
        BOOST_ASSERT(hole_ratio <= 1.);

        std::size_t overflow_count =
            (count > missing) ? (count - missing) / counts.size() : 0;
        std::size_t excess = count - overflow_count * counts.size();

        typedef std::vector<lazy_result> future_values_type;
        typedef server::runtime_support::bulk_create_components_action
            action_type;

        std::size_t created_count = 0;
        future_values_type v;

        // start an asynchronous operation for each of the localities
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            std::size_t numcreate =
                std::size_t((maxcount - counts[i]) * hole_ratio) + overflow_count;

            if (excess != 0) {
                --excess;
                ++numcreate;
            }

            if (i == counts.size()-1) {
                // last bin gets all the rest
                if (created_count + numcreate < count)
                    numcreate = count - created_count;
            }

            if (created_count + numcreate > count)
                numcreate = count - created_count;

            if (numcreate == 0)
                break;

            // create all components  for each locality at a time
            v.push_back(future_values_type::value_type(localities[i].get_gid()));
            lcos::packaged_action<action_type, std::vector<naming::gid_type> > p;
            p.apply(localities[i], type, numcreate);
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
}}}

