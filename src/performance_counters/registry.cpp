//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters 
{
    ///////////////////////////////////////////////////////////////////////////
    registry::registry(naming::resolver_client& agas_client)
      : agas_client_(agas_client)
    {}

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::add_counter_type(counter_info const& info, 
        error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (status_valid_data != status) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it != countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter_type", 
                "counter type already defined");
            return status_already_defined;
        }

        std::pair<counter_type_map_type::iterator, bool> p = 
            countertypes_.insert(counter_type_map_type::value_type(type_name, info));

        return p.second ? status_valid_data : status_invalid_data;
    }

    boost::int64_t wrap_counter(boost::int64_t* p)
    {
        return *p;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::add_counter(counter_info const& info, 
        boost::int64_t* countervalue, naming::id_type& id, error_code& ec)
    {
        return add_counter(info, boost::bind(wrap_counter, countervalue), id, ec);
    }

    counter_status registry::add_counter(counter_info const& info, 
        boost::function<boost::int64_t()> f, naming::id_type& id, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (status_valid_data != status) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter", 
                "unknown counter type");
            return status_counter_type_unknown;
        }

        // make sure the counter type requested is supported
        if (counter_raw != (*it).second.type_ || counter_raw != info.type_) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter", 
                "invalid counter type requested (only counter_raw is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        naming::id_type newid = naming::invalid_id;
        try {
            typedef components::managed_component<server::raw_counter> counter_type;
            newid = components::server::create_one<counter_type>(complemented_info, f);
        }
        catch (hpx::exception const& e) {
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            return status_invalid_data;
        }

        // register the conical name with AGAS
        agas_client_.registerid(complemented_info.fullname_, id, ec);
        if (ec) return status_invalid_data;

        id = newid;
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::remove_counter(counter_info const& info, 
        naming::id_type const& id, error_code& ec)
    {
        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, ec);
        if (ec) return status_invalid_data;

        // create canonical name for the counter
        std::string name;
        counter_status status = get_counter_name(complemented_info.fullname_, 
            name, ec);
        if (status_valid_data != status) return status;

        // unregister this counter from AGAS
        agas_client_.unregisterid(name, ec);
        if (ec) return status_invalid_data;

        // delete the counter
        switch (info.type_) {
        case counter_raw:
            {
                typedef 
                    components::managed_component<server::raw_counter> 
                counter_type;
                components::server::destroy<counter_type>(id, ec);
                if (ec) return status_invalid_data;
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "registry::remove_counter", 
                "invalid counter type requested (only counter_raw is supported)");
            return status_counter_type_unknown;
        }
        return status_valid_data;
    }

}}


