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
HPX_REGISTER_ACTION_EX(vertex_list_type::list_action, vertex_list_list);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<vertex_list_type>, vertex_list);
HPX_DEFINE_GET_COMPONENT_TYPE(vertex_list_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    vertex_list::vertex_list()
      : num_items_(0)
    {}
    
    int vertex_list::init(components::component_type item_type, std::size_t num_items)
    {                    
        num_items_ = num_items;

        std::cout << "Initializing vertex_list of length " << num_items << std::endl;

        naming::id_type here = applier::get_applier().get_runtime_support_gid();

        using components::distributing_factory;
        distributing_factory factory(distributing_factory::create(here, true));

        list_ = factory.create_components(item_type, num_items);

        std::cout << "Number of sub lists: " << list_.size() << std::endl;

        // Run through and initialize each vertex in order
        components::distributing_factory::iterator_range_type range;
        components::distributing_factory::iterator_type iter;

        range = locality_results(list_);
        iter = range.first;
        for (int label = 0; iter != range.second; ++iter, ++label)
        {
            components::stubs::vertex::init(*iter, label);
        }

        return 0;
    }
    
    int vertex_list::size(void)
    {
        return num_items_;
    }

    naming::id_type vertex_list::at_index(const int index)
    {
        // Run through and initialize each vertex in order
        components::distributing_factory::iterator_range_type range;
        components::distributing_factory::iterator_type iter;

        range = locality_results(list_);
        iter = range.first;
        for(int j = 0; iter != range.second; ++iter, ++j)
        {
            if (j == index)
                return *iter;
        }

        // This is what I want to do, but it only ever returns the first item
    	return (locality_results(list_).first[index]);
    }

    typedef hpx::components::distributing_factory::result_type result_type;
    result_type vertex_list::list(void)
    {
        return list_;
    }

}}}
