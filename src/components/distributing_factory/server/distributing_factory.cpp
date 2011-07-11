//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    struct lazy_result
    {
        lazy_result(naming::gid_type const& prefix)
          : prefix_(prefix)
        {}

        naming::gid_type prefix_;
        std::vector<lcos::future_value<naming::gid_type> > gids_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a component
    distributing_factory::remote_result_type 
    distributing_factory::create_components(
        components::component_type type, std::size_t count)
    {
        // make sure we get prefixes for derived component type, if any
        components::component_type prefix_type = type;
        if (type != components::get_base_type(type))
            prefix_type = components::get_derived_type(type);

        // get list of locality prefixes
        std::vector<naming::gid_type> prefixes;
        hpx::applier::get_applier().get_agas_client().get_prefixes(prefixes, prefix_type);

        if (prefixes.empty())
        {
            HPX_THROW_EXCEPTION(bad_component_type, 
                "distributing_factory::create_components",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        std::size_t created_count = 0;
        std::size_t count_on_locality = count / prefixes.size();
        std::size_t excess = count - count_on_locality*prefixes.size();

        // distribute the number of components to create evenly over all 
        // available localities
        typedef std::vector<lazy_result> future_values_type;
        typedef server::runtime_support::create_component_action action_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;

        BOOST_FOREACH(naming::gid_type const& gid, prefixes)
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
            naming::id_type fact(gid, naming::id_type::unmanaged);

            v.push_back(future_values_type::value_type(gid));
            for (std::size_t i = 0; i < numcreate; ++i) {
                lcos::eager_future<action_type, naming::gid_type> f(fact, type, 1);
                v.back().gids_.push_back(f);
            }

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        remote_result_type results;

        BOOST_FOREACH(lazy_result const& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.prefix_, type));
            components::wait(lr.gids_, results.back().gids_);
        }

        return results;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Action to delete existing components
//     void distributing_factory::free_components(result_type const& gids, bool sync)
//     {
//         result_type::const_iterator end = gids.end();
//         for (result_type::const_iterator it = gids.begin(); it != end; ++it) 
//         {
//             for (std::size_t i = 0; i < (*it).count_; ++i) 
//             {
//                 // We need to free every components separately because it may
//                 // have been moved to a different locality than it was 
//                 // initially created on.
//                 if (sync) {
//                     components::stubs::runtime_support::free_component_sync(
//                         (*it).type_, (*it).first_gid_ + i);
//                 }
//                 else {
//                     components::stubs::runtime_support::free_component(
//                         (*it).type_, (*it).first_gid_ + i);
//                 }
//             }
//         }
//     }

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

