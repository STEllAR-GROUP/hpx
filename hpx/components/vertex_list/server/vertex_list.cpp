//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/vertex/vertex.hpp>

#include "vertex_list.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::vertex_list vertex_list_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the vertex_list actions
HPX_REGISTER_ACTION_EX(vertex_list_type::init_action, vertex_list_init_action);

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<vertex_list_type>, vertex_list);
HPX_DEFINE_GET_COMPONENT_TYPE(vertex_list_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    vertex_list::vertex_list()
      : num_items_(0),
        blocks_(0)
    {}
    
    int vertex_list::init(components::component_type item_type, std::size_t num_items)
    {                    
        std::size_t block_size_;

        std::cout << "Initializng vertex_list of length " << num_items << "\n";

        // Get vertex_list of all known localities
        std::vector<naming::id_type> locales;
        naming::id_type locale;
        applier::applier& appl = applier::get_applier();
        if (appl.get_remote_prefixes(locales))
        {
            locale = locales[0];
        }
        else
        {
            locale = appl.get_runtime_support_gid();
        }
        locales.push_back(appl.get_runtime_support_gid());
    
        // Calculate block distribution
        block_size_ = num_items / (locales.size());
        std::cout << "Block size is " << block_size_ << "\n";

        typedef components::distributing_factory::result_type result_type;

        std::vector<result_type> sub_lists;

        // Create factories on each locality
        for (std::size_t i = 0; i < locales.size(); i++)
        {
            std::cout << "Creating factory on locality " << i << "\n";

            // Create distributing factories across the system
            components::distributing_factory factory(
                components::distributing_factory::create(
                    locales[i], true));

            sub_lists.push_back(factory.create_components(item_type, block_size_));
        }

        // Test: run through and initialize each vertex in order
        components::distributing_factory::iterator_range_type range;
        components::distributing_factory::iterator_type iter;

        for (int i=0; i < sub_lists.size(); ++i)
        {
            range = locality_results(sub_lists[i]);
            iter = range.first;
            for(int label = 0; iter != range.second; ++iter, ++label)
            {
                using components::vertex;
                vertex v(*iter);
                v.init(label);
            }
        }

        return 0;
    }
    
}}}
