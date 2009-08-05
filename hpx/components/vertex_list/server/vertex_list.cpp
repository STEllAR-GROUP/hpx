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
#include "../stubs/vertex_list.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::vertex_list vertex_list_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the vertex_list actions
HPX_REGISTER_ACTION_EX(vertex_list_type::init_action, vertex_list_init_action);
HPX_REGISTER_ACTION_EX(vertex_list_type::size_action, vertex_list_size_action);
HPX_REGISTER_ACTION_EX(vertex_list_type::at_index_action, vertex_list_at_index_action);
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
        num_items_ = num_items;

        std::cout << "Initializng vertex_list of length " << num_items << std::endl;

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
        std::cout << "Using block size of " << block_size_ << std::endl;

        // Create factories on each locality
        for (std::size_t i = 0; i < locales.size(); i++)
        {
            // Create distributing factories across the system
            components::distributing_factory factory(
                components::distributing_factory::create(
                    locales[i], true));

            sub_lists_.push_back(factory.create_components(item_type, block_size_));
        }

        // Run through and initialize each vertex in order
        components::distributing_factory::iterator_range_type range;
        components::distributing_factory::iterator_type iter;

        for (int i=0; i < sub_lists_.size(); ++i)
        {
            range = locality_results(sub_lists_[i]);
            iter = range.first;
            for(int label = 0; iter != range.second; ++iter, ++label)
            {
                components::stubs::vertex::init(*iter, label);
            }
        }

        return 0;
    }
    
    int vertex_list::size(void)
    {
        return num_items_;
    }

    naming::id_type vertex_list::at_index(const int index)
    {
        int block = index / block_size_;
        int item = index % block_size_;

        // Run through and initialize each vertex in order
        components::distributing_factory::iterator_range_type range;
        components::distributing_factory::iterator_type iter;

        range = locality_results(sub_lists_[block]);
        iter = range.first;
        for(int j = 0; iter != range.second; ++iter, ++j)
        {
            if (j == item)
                return *iter;
        }

        // This is what I want to do, but it only ever returns the first item
    	return (locality_results(sub_lists_[block]).first[item]);
    }

}}}
