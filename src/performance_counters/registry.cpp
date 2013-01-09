//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/create_component_with_args.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>
#include <hpx/performance_counters/server/aggregating_counter.hpp>
#include <hpx/util/logging.hpp>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    registry::counter_type_map_type::iterator
        registry::locate_counter_type(std::string const& type_name)
    {
        counter_type_map_type::iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            // if the full type is not available, try to locate the object name
            // as a type only
            error_code ec(lightweight);
            counter_path_elements p;
            get_counter_type_path_elements(type_name, p, ec);
            if (!ec)
                it = countertypes_.find("/" + p.objectname_);
        }
        return it;
    }

    registry::counter_type_map_type::const_iterator
        registry::locate_counter_type(std::string const& type_name) const
    {
        counter_type_map_type::const_iterator it = countertypes_.find(type_name);
        if (it == countertypes_.end()) {
            // if the full type is not available, try to locate the object name
            // as a type only
            error_code ec(lightweight);
            counter_path_elements p;
            get_counter_type_path_elements(type_name, p, ec);
            if (!ec)
                it = countertypes_.find("/" + p.objectname_);
        }
        return it;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::add_counter_type(counter_info const& info,
        HPX_STD_FUNCTION<create_counter_func> const& create_counter_,
        HPX_STD_FUNCTION<discover_counters_func> const& discover_counters_,
        error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it != countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter_type",
                boost::str(boost::format(
                    "counter type already defined: %s") % type_name));
            return status_already_defined;
        }

        std::pair<counter_type_map_type::iterator, bool> p =
            countertypes_.insert(counter_type_map_type::value_type(
            type_name, counter_data(info, create_counter_, discover_counters_)));

        if (!p.second) {
            LPCS_(warning) << (
                boost::format("failed to register counter type %s") % type_name);
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("counter type %s registered") %
            type_name);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Call the supplied function for the given registered counter type.
    counter_status registry::discover_counter_type(
        std::string const& fullname,
        HPX_STD_FUNCTION<discover_counter_func> discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(fullname, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            // compose a list of known counter types
            std::string types;
            counter_type_map_type::const_iterator end = countertypes_.end();
            for (counter_type_map_type::const_iterator it = countertypes_.begin();
                 it != end; ++it)
            {
                types += "  " + (*it).first + "\n";
            }

            HPX_THROWS_IF(ec, bad_parameter, "registry::discover_counter_type",
                boost::str(boost::format("unknown counter type: %s, known counter types: %s") % 
                    type_name % types));
            return status_counter_type_unknown;
        }

        if (mode == discover_counters_full) 
        {
            using HPX_STD_PLACEHOLDERS::_1;
            discover_counter = HPX_STD_BIND(&expand_counter_info, _1, 
                discover_counter, boost::ref(ec));
        }

        counter_info info = (*it).second.info_;
        info.fullname_ = fullname;

        if (!(*it).second.discover_counters_.empty() &&
            !(*it).second.discover_counters_(info, discover_counter, mode, ec))
        {
            return status_invalid_data;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return status_valid_data;
    }

    /// \brief Call the supplied function for all registered counter types.
    counter_status registry::discover_counter_types(
        HPX_STD_FUNCTION<discover_counter_func> discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        if (mode == discover_counters_full) 
        {
            using HPX_STD_PLACEHOLDERS::_1;
            discover_counter = HPX_STD_BIND(&expand_counter_info, _1, 
                discover_counter, boost::ref(ec));
        }

        BOOST_FOREACH(counter_type_map_type::value_type const& d, countertypes_)
        {
            if (!d.second.discover_counters_.empty() &&
                !d.second.discover_counters_(
                      d.second.info_, discover_counter, mode, ec))
            {
                return status_invalid_data;
            }
        }

        if (&ec != &throws)
            ec = make_success_code();

        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::get_counter_create_function(
        counter_info const& info, HPX_STD_FUNCTION<create_counter_func>& func,
        error_code& ec) const
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::const_iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::get_counter_create_function",
                "counter type is not defined");
            return status_counter_type_unknown;
        }

        if ((*it).second.create_counter_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::get_counter_create_function",
                "counter type has no associated create function");
            return status_invalid_data;
        }

        func = (*it).second.create_counter_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::get_counter_discovery_function(
        counter_info const& info, HPX_STD_FUNCTION<discover_counters_func>& func,
        error_code& ec) const
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::const_iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::get_counter_discovery_function",
                "counter type is not defined");
            return status_counter_type_unknown;
        }

        if ((*it).second.discover_counters_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::get_counter_discovery_function",
                "counter type has no associated discovery function");
            return status_invalid_data;
        }

        func = (*it).second.discover_counters_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::remove_counter_type(counter_info const& info,
        error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::remove_counter_type",
                "counter type is not defined");
            return status_counter_type_unknown;
        }

        LPCS_(info) << (
            boost::format("counter type %s unregistered") % type_name);

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
    counter_status registry::create_raw_counter_value(counter_info const& info,
        boost::int64_t* countervalue, naming::gid_type& id, error_code& ec)
    {
        return create_raw_counter(info, boost::bind(wrap_counter, countervalue), id, ec);
    }

    counter_status registry::create_raw_counter(counter_info const& info,
        HPX_STD_FUNCTION<boost::int64_t()> const& f, naming::gid_type& id,
        error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_raw_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the counter type requested is supported
        if (counter_raw != (*it).second.info_.type_ || counter_raw != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_raw_counter",
                "invalid counter type requested (only counter_raw is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second.info_, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            typedef components::managed_component<server::raw_counter> counter_t;
            id = components::server::create_with_args<counter_t>(complemented_info, f);

            std::string name(complemented_info.fullname_);
            ensure_counter_prefix(name);      // pre-pend prefix, if necessary
        }
        catch (hpx::exception const& e) {
            id = naming::invalid_gid;        // reset result
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (
                boost::format("failed to create raw counter %s (%s)") %
                    complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("raw counter %s created at %s")
            % complemented_info.fullname_ % id);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::create_counter(counter_info const& info,
        naming::gid_type& id, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second.info_, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            switch (complemented_info.type_) {
            case counter_elapsed_time:
                {
                    typedef components::managed_component<server::elapsed_time_counter> counter_t;
                    id = components::server::create_with_args<counter_t>(complemented_info);
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
        }
        catch (hpx::exception const& e) {
            id = naming::invalid_gid;        // reset result
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

    /// \brief Create a new aggregating performance counter instance based
    ///        on given base counter name and given base time interval
    ///        (milliseconds).
    counter_status registry::create_aggregating_counter(
        counter_info const& info, std::string const& base_counter_name,
        boost::int64_t base_time_interval, naming::gid_type& gid, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_aggregating_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the counter type requested is supported
        if (counter_aggregating != (*it).second.info_.type_ ||
            counter_aggregating != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_aggregating_counter",
                "invalid counter type requested (only counter_aggregating is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second.info_, ec);
        if (ec) return status_invalid_data;

        // split name
        counter_path_elements p;
        get_counter_path_elements(complemented_info.fullname_, p, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            // create base counter only if it does not exist yet
            if (p.countername_ == "average") {
                typedef hpx::components::managed_component<
                    hpx::performance_counters::server::aggregating_counter<
                        boost::accumulators::tag::mean>
                > counter_t;
                gid = components::server::create_with_args<counter_t>(
                    complemented_info, base_counter_name, base_time_interval);
            }
            else if (p.countername_ == "max") {
                typedef hpx::components::managed_component<
                    hpx::performance_counters::server::aggregating_counter<
                        boost::accumulators::tag::max>
                > counter_t;
                gid = components::server::create_with_args<counter_t>(
                    complemented_info, base_counter_name, base_time_interval);
            }
            else if (p.countername_ == "min") {
                typedef hpx::components::managed_component<
                    hpx::performance_counters::server::aggregating_counter<
                        boost::accumulators::tag::min>
                > counter_t;
                gid = components::server::create_with_args<counter_t>(
                    complemented_info, base_counter_name, base_time_interval);
            }
            else {
                HPX_THROWS_IF(ec, bad_parameter,
                    "registry::create_aggregating_counter",
                    "invalid counter type requested");
                return status_counter_type_unknown;
            }
        }
        catch (hpx::exception const& e) {
            gid = naming::invalid_gid;        // reset result
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (
                boost::format("failed to create aggregating counter %s (%s)") %
                    complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("aggregating counter %s created at %s") %
            complemented_info.fullname_ % gid);

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
        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::add_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // register the canonical name with AGAS
        std::string name(complemented_info.fullname_);
        ensure_counter_prefix(name);      // pre-pend prefix, if necessary
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
        ensure_counter_prefix(name);      // pre-pend prefix, if necessary
        agas::unregister_name(name, ec);
        if (ec) {
            LPCS_(warning) << ( boost::format("failed to remove counter %s")
                % complemented_info.fullname_);
            return status_invalid_data;
        }

//         // delete the counter
//         switch (info.type_) {
//         case counter_elapsed_time:
//         case counter_raw:
// //             {
// //                 typedef
// //                     components::managed_component<server::raw_counter>
// //                 counter_type;
// //                 components::server::destroy<counter_type>(id.get_gid(), ec);
// //                 if (ec) return status_invalid_data;
// //             }
//             break;
//
//         default:
//             HPX_THROWS_IF(ec, bad_parameter, "registry::remove_counter",
//                 "invalid counter type requested");
//             return status_counter_type_unknown;
//         }
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Retrieve counter type information for given counter name
    counter_status registry::get_counter_type(std::string const& name,
        counter_info& info, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(name, type_name, ec);
        if (!status_is_valid(status)) return status;

        // make sure the type of the counter is known to the registry
        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::get_counter_type",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        info = (*it).second.info_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }
}}


