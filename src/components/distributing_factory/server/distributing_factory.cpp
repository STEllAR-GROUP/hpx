//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

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
        std::vector<lcos::future<naming::gid_type> > gids_;
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
        typedef server::runtime_support::create_component_action action_type;

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

            // create one component at a time
            v.push_back(future_values_type::value_type(fact.get_gid()));
            for (std::size_t i = 0; i < numcreate; ++i)
            {
                lcos::packaged_action<action_type, naming::gid_type> p;
                p.apply(fact, type, 1);
                v.back().gids_.push_back(p.get_future());
            }

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        remote_result_type results;

        BOOST_FOREACH(lazy_result const& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.locality_, type));
            lcos::wait(lr.gids_, results.back().gids_);
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
        typedef server::runtime_support::create_component_action action_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;

        for (std::size_t i = 0, j = 0;
             i < localities.size() && j < parts;
             i += parts_delta, ++j)
        {
            // create one component at a time, overall, 'count' components
            // for each partition
            naming::id_type fact(localities[i]);

            v.push_back(future_values_type::value_type(fact.get_gid()));
            for (std::size_t k = 0; k < count; ++k)
            {
                lcos::packaged_action<action_type, naming::gid_type> p;
                p.apply(fact, type, 1);
                v.back().gids_.push_back(p.get_future());
            }
        }

        // now wait for the results
        remote_result_type results;

        BOOST_FOREACH(lazy_result const& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.locality_, type));
            lcos::wait(lr.gids_, results.back().gids_);
        }

        return results;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///
    locality_result_iterator::data::data(result_type::const_iterator begin,
            result_type::const_iterator end)
      : current_(begin), end_(end), is_at_end_(begin == end)
    {
        if (!is_at_end_)
            current_gid_ = (*current_).begin();
    }

    /// construct end iterator
    locality_result_iterator::data::data()
      : is_at_end_(true)
    {}

    void locality_result_iterator::data::increment()
    {
        if (!is_at_end_) {
            if (++current_gid_ == (*current_).end()) {
                if (++current_ != end_) {
                    current_gid_ = (*current_).begin();
                }
                else {
                    is_at_end_ = true;
                }
            }
        }
    }

    bool locality_result_iterator::data::equal(data const& rhs) const
    {
        if (is_at_end_ != rhs.is_at_end_)
            return false;

        return (is_at_end_ && rhs.is_at_end_) ||
               (current_ == rhs.current_ && current_gid_ == rhs.current_gid_);
    }

    naming::id_type const& locality_result_iterator::data::dereference() const
    {
        BOOST_ASSERT(!is_at_end_);
        return *current_gid_;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// return an iterator range for the given locality_result's
    std::pair<locality_result_iterator, locality_result_iterator>
    locality_results(distributing_factory::result_type const& v)
    {
        typedef std::pair<locality_result_iterator, locality_result_iterator>
            result_type;
        return result_type(locality_result_iterator(v), locality_result_iterator());
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    template <typename Archive>
    void serialize(Archive& ar, hpx::components::server::partition_info& info,
        unsigned int const)
    {
        ar & info.dims_ & info.dim_sizes_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive&,
        hpx::components::server::partition_info&, unsigned int const);
    template HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive&,
        hpx::components::server::partition_info&, unsigned int const);
}}

