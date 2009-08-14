//  Copyright (c) 2009-2010 Dylan Stark
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

#include "local_list.hpp"
#include "../stubs/local_list.hpp"

// Needs this to define edge_list_type
#include "../../../../applications/graphs/ssca2/kernel2/kernel2.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::local_list<
    hpx::components::server::kernel2::edge_list_type
> local_edge_list_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the local_list actions

HPX_REGISTER_ACTION_EX(
    local_edge_list_type::append_action,
    local_list_append_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_edge_list_type>, local_list);
HPX_DEFINE_GET_COMPONENT_TYPE(local_edge_list_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    local_list<List>::local_list()
      : local_list_(0)
    {}
    
    template <typename List>
    int local_list<List>::append(List list)
    {
        std::cout << "Appending to local list at locale " << std::endl;

        // Probably should do some locking ... somewhere ... maybe here

        typedef typename List::iterator list_iter;
        list_iter end = list.end();
        for (list_iter it = list.begin(); it != end; ++it)
        {
            local_list_.push_back(*it);
        }

        return local_list_.size();
    }

}}}
