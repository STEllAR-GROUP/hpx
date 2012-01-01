//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>
#include <hpx/performance_counters/server/average_count_counter.hpp>
#include <hpx/util/logging.hpp>

#include <boost/format.hpp>

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
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it != countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter_type",
                "counter type already defined");
            return status_already_defined;
        }

        std::pair<counter_type_map_type::iterator, bool> p =
            countertypes_.insert(counter_type_map_type::value_type(type_name, info));

        if (p.second) {
            LPCS_(info)
                << (boost::format("counter type %s created") % type_name);

            if (&ec != &throws)
                ec = make_success_code();
            return status_valid_data;
        }

        LPCS_(warning)
            << (boost::format("failed to create counter type %s") % type_name);
        return status_invalid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::remove_counter_type(counter_info const& info,
        error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::remove_counter_type",
                "counter type is not defined");
            return status_counter_type_unknown;
        }

        LPCS_(info)
            << (boost::format("counter type %s destroyed") % type_name);

        countertypes_.erase(it);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::int64_t wrap_counter(boost::int64_t* p)
    {
        return *p;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::create_raw_counter(counter_info const& info,
        boost::int64_t* countervalue, naming::id_type& id, error_code& ec)
    {
        return create_raw_counter(info, boost::bind(wrap_counter, countervalue), id, ec);
    }

    counter_status registry::create_raw_counter(counter_info const& info,
        HPX_STD_FUNCTION<boost::int64_t()> f, naming::id_type& id, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_raw_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the counter type requested is supported
        if (counter_raw != (*it).second.type_ || counter_raw != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_raw_counter",
                "invalid counter type requested (only counter_raw is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            typedef components::managed_component<server::raw_counter> counter_type;
            naming::gid_type const newid =
                components::server::create_one<counter_type>(complemented_info, f);

            std::string name(complemented_info.fullname_);
            ensure_counter_prefix(name);      // prepend prefix, if necessary

            // register the canonical name with AGAS
            id = naming::id_type(newid, naming::id_type::managed);
            agas::register_name(name, id);
        }
        catch (hpx::exception const& e) {
            id = naming::invalid_id;        // reset result
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (boost::format("failed to create counter %s (%s)")
                % complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("counter %s created at %s")
            % complemented_info.fullname_ % id);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::create_counter(counter_info const& info,
        naming::id_type& id, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            naming::gid_type newid = naming::invalid_gid;

            switch (complemented_info.type_) {
            case counter_elapsed_time:
                {
                    typedef components::managed_component<server::elapsed_time_counter> counter_type;
                    newid = components::server::create_one<counter_type>(complemented_info);
                }
                break;

            case counter_raw:
                HPX_THROWS_IF(ec, bad_parameter, "registry::create_counter",
                    "need function parameter for raw_counter");
                return status_counter_type_unknown;

            default:
                HPX_THROWS_IF(ec, bad_parameter, "registry::create_counter",
                    "invalid counter type requested");
                return status_counter_type_unknown;
            }

            std::string name(complemented_info.fullname_);
            ensure_counter_prefix(name);      // prepend prefix, if necessary

            // register the canonical name with AGAS
            id = naming::id_type(newid, naming::id_type::managed);
            agas::register_name(name, id);
        }
        catch (hpx::exception const& e) {
            id = naming::invalid_id;        // reset result
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (boost::format("failed to create counter %s (%s)")
                % complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("counter %s created at %s")
            % complemented_info.fullname_ % id);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a new performance counter instance of type
    ///        counter_average_count based on given base counter name and
    ///        given base time interval (milliseconds)
    counter_status registry::create_average_count_counter(
        counter_info const& info, std::string const& base_counter_name,
        std::size_t base_time_interval, naming::id_type& id, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_average_count_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the counter type requested is supported
        if (counter_average_count != (*it).second.type_ ||
            counter_average_count != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_average_count_counter",
                "invalid counter type requested (only counter_raw is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            typedef components::managed_component<server::average_count_counter>
                counter_type;
            naming::gid_type const newid = components::server::create_one<counter_type>(
                complemented_info, base_counter_name, base_time_interval);

            std::string name(complemented_info.fullname_);
            ensure_counter_prefix(name);      // prepend prefix, if necessary

            // register the canonical name with AGAS
            id = naming::id_type(newid, naming::id_type::managed);
            agas::register_name(name, id);
        }
        catch (hpx::exception const& e) {
            id = naming::invalid_id;        // reset result
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (boost::format("failed to create counter %s (%s)")
                % complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("counter %s created at %s")
            % complemented_info.fullname_ % id);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add an existing performance counter instance to the registry
    counter_status registry::add_counter(naming::id_type const& id,
        counter_info const& info, error_code& ec)
    {
        // complement counter info data
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, ec);
        if (ec) return status_invalid_data;

        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(
            complemented_info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        // make sure the type of the new counter is known to the registry
        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // register the canonical name with AGAS
        std::string name(complemented_info.fullname_);
        ensure_counter_prefix(name);      // prepend prefix, if necessary
        agas::register_name(name, id, ec);
        if (ec) return status_invalid_data;

        if (&ec != &throws)
            ec = make_success_code();
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
        if (!status_is_valid(status)) return status;

        // unregister this counter from AGAS
        ensure_counter_prefix(name);      // prepend prefix, if necessary
        agas::unregister_name(name, ec);
        if (ec) {
            LPCS_(warning) << ( boost::format("failed to destroy counter %s")
                % complemented_info.fullname_);
            return status_invalid_data;
        }

        // delete the counter
        switch (info.type_) {
        case counter_elapsed_time:
        case counter_raw:
//             {
//                 typedef
//                     components::managed_component<server::raw_counter>
//                 counter_type;
//                 components::server::destroy<counter_type>(id.get_gid(), ec);
//                 if (ec) return status_invalid_data;
//             }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "registry::remove_counter",
                "invalid counter type requested");
            return status_counter_type_unknown;
        }
        return status_valid_data;
    }
}}


