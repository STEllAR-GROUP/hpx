//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/performance_counters/registry.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters 
{
    ///////////////////////////////////////////////////////////////////////////
    registry::registry(naming::resolver_client& agas_client)
      : agas_client_(agas_client)
    {}

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::add_counter_type(counter_info const& info)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name);
        if (status_valid_data != status) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it != countertypes_.end()) 
            return status_already_defined;

        std::pair<counter_type_map_type::iterator, bool> p = 
            countertypes_.insert(counter_type_map_type::value_type(type_name, info));

        return p.second ? status_valid_data : status_invalid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::add_counter(counter_info const& info, 
        boost::int64_t* countervalue, naming::id_type& id)
    {
        id = naming::invalid_id;
        return status_valid_data;
    }

    counter_status registry::add_counter(counter_info const& info, 
        boost::function<boost::int64_t()> f, naming::id_type& id)
    {
        id = naming::invalid_id;
        return status_valid_data;
    }

}}


