//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

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
    threads::thread_state distributing_factory::create_components(
        threads::thread_self& self, applier::applier& appl,
        result_type* gids, components::component_type type, 
        std::size_t count)
    {
        // get list of locality prefixes
        std::vector<naming::id_type> prefixes;
        appl.get_agas_client().get_prefixes(prefixes);

        std::size_t created_count = 0;
        std::size_t count_on_locality = count / prefixes.size();
        if (0 == count_on_locality)
            count_on_locality = 1;

        // distribute the number of components to create evenly on all 
        // available localities
        typedef std::vector<lazy_result> future_values_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;
        components::stubs::runtime_support rts(appl);

        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            std::size_t numcreate = count_on_locality;
            if (created_count + numcreate > count)
                numcreate = count - created_count;

            // figure out, whether we can create more than one instance of the 
            // component at once
            bool has_multi_instance_factory = false;
            if (1 != numcreate) {
                has_multi_instance_factory = 
                    rts.has_multi_instance_factory(self, *it, type);
            }

            if (has_multi_instance_factory) {
                // create all component instances at once
                lcos::future_value<naming::id_type> f (
                    rts.create_component_async(*it, type, numcreate));
                v.push_back(future_values_type::value_type(*it, f, numcreate));
            }
            else {
                // create one component at a time
                for (std::size_t i = 0; i < numcreate; ++i) {
                    lcos::future_value<naming::id_type> f (
                        rts.create_component_async(*it, type));
                    v.push_back(future_values_type::value_type(*it, f, 1));
                }
            }

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        future_values_type::iterator vend = v.end();
        for (future_values_type::iterator vit = v.begin(); vit != vend; ++vit)
        {
            gids->push_back(result_type::value_type(
                (*vit).prefix_, (*vit).gids_.get(self), (*vit).count_, type));
        }

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Action to delete existing components
    threads::thread_state distributing_factory::free_components(
        threads::thread_self& self, applier::applier& appl,
        result_type const& gids)
    {
        components::stubs::runtime_support rts(appl);

        result_type::const_iterator end = gids.end();
        for (result_type::const_iterator it = gids.begin(); it != end; ++it) 
        {
            for (std::size_t i = 0; i < (*it).count_; ++i) 
            {
                // We need to free every components separately because it may
                // have been moved to a different locality than it was 
                // initially created on.
                rts.free_component((*it).type_, (*it).first_gid_ + i);
            }
        }

        return threads::terminated;
    }


}}}

