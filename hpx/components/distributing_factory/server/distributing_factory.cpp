//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    struct lazy_result
    {
        lazy_result(naming::id_type const& prefix, 
                lcos::future_value<naming::id_type> gids,
                std::size_t count)
          : prefix_(prefix), gids_(gids), count_(count)
        {}

        naming::id_type prefix_;
        lcos::future_value<naming::id_type> gids_;
        std::size_t count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a component
    distributing_factory::result_type distributing_factory::create_components(
        components::component_type type, std::size_t count)
    {
        // make sure we get prefixes for derived component type, if any
        components::component_type prefix_type = type;
        if (type != components::get_base_type(type))
            prefix_type = components::get_derived_type(type);

        // get list of locality prefixes
        std::vector<naming::id_type> prefixes;
        hpx::applier::get_applier().get_agas_client().get_prefixes(prefixes, prefix_type);

        if (prefixes.empty())
        {
            HPX_THROW_EXCEPTION(bad_component_type, 
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        std::size_t created_count = 0;
        std::size_t count_on_locality = count / prefixes.size();
        if (0 == count_on_locality)
            count_on_locality = 1;

        // distribute the number of components to create evenly over all 
        // available localities
        typedef std::vector<lazy_result> future_values_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;

        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            std::size_t numcreate = count_on_locality;
            if (created_count + numcreate > count)
                numcreate = count - created_count;

            // figure out, whether we can create more than one instance of the 
            // component at once
            int factory_props = factory_none;
            if (1 != numcreate) {
                factory_props = components::stubs::runtime_support::
                    get_factory_properties(*it, type);
            }

            if (factory_props & factory_is_multi_instance) {
                // create all component instances at once
                lcos::future_value<naming::id_type> f (
                    components::stubs::runtime_support::create_component_async(
                        *it, type, numcreate));
                v.push_back(future_values_type::value_type(*it, f, numcreate));
            }
            else {
                // create one component at a time
                for (std::size_t i = 0; i < numcreate; ++i) {
                    lcos::future_value<naming::id_type> f (
                        components::stubs::runtime_support::
                            create_component_async(*it, type));
                    v.push_back(future_values_type::value_type(*it, f, 1));
                }
            }

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        result_type gids;
        future_values_type::iterator vend = v.end();
        for (future_values_type::iterator vit = v.begin(); vit != vend; ++vit)
        {
            gids.push_back(result_type::value_type(
                (*vit).prefix_, (*vit).gids_.get(), (*vit).count_, type));
        }
        return gids;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Action to delete existing components
    void distributing_factory::free_components(result_type const& gids, bool sync)
    {
        result_type::const_iterator end = gids.end();
        for (result_type::const_iterator it = gids.begin(); it != end; ++it) 
        {
            for (std::size_t i = 0; i < (*it).count_; ++i) 
            {
                // We need to free every components separately because it may
                // have been moved to a different locality than it was 
                // initially created on.
                if (sync) {
                    components::stubs::runtime_support::free_component_sync(
                        (*it).type_, (*it).first_gid_ + i);
                }
                else {
                    components::stubs::runtime_support::free_component(
                        (*it).type_, (*it).first_gid_ + i);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// 
    locality_result_iterator::data::data(result_type::const_iterator begin, 
            result_type::const_iterator end)
      : current_(begin), end_(end), count_(0), is_at_end_(begin == end)
    {
        if (!is_at_end_)
            value_ = (*current_).first_gid_;
    }

    /// construct end iterator
    locality_result_iterator::data::data()
      : is_at_end_(true)
    {}

    void locality_result_iterator::data::increment()
    {
        if (!is_at_end_) {
            if (++count_ != (*current_).count_) {
                value_ = (*current_).first_gid_ + count_;
            }
            else if (++current_ != end_) {
                count_ = 0;
                value_ = (*current_).first_gid_;
            }
            else {
                is_at_end_ = true;
            }
        }
    }

    bool locality_result_iterator::data::equal(data const& rhs) const
    {
        if (is_at_end_ != rhs.is_at_end_)
            return false;

        return (is_at_end_ && rhs.is_at_end_) ||
               (current_ == rhs.current_ && count_ == rhs.count_);
    }

    naming::id_type const& locality_result_iterator::data::dereference() const
    {
        BOOST_ASSERT(!is_at_end_);
        return value_;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// return an iterator range for the given locality_result's
    std::pair<locality_result_iterator, locality_result_iterator>
    locality_results(distributing_factory::result_type const& v)
    {
        typedef 
            std::pair<locality_result_iterator, locality_result_iterator>
        result_type;
        return result_type(locality_result_iterator(v), locality_result_iterator());
    }

}}}

