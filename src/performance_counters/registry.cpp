//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>
#include <hpx/performance_counters/server/raw_values_counter.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>
#include <hpx/performance_counters/server/statistics_counter.hpp>
#include <hpx/performance_counters/server/arithmetics_counter.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/logging.hpp>

#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/accumulators/statistics_fwd.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

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
        create_counter_func const& create_counter_,
        discover_counters_func const& discover_counters_,
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

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        inline std::string
        regex_from_character_set(std::string::const_iterator& it,
            std::string::const_iterator end, error_code& ec)
        {
            std::string::const_iterator start = it;
            std::string result(1, *it);  // copy '['
            if (*++it == '!') {
                result.append(1, '^');   // negated character set
            }
            else if (*it == ']') {
                HPX_THROWS_IF(ec, bad_parameter, "regex_from_character_set",
                    "Invalid pattern (empty character set) at: " +
                        std::string(start, end));
                return "";
            }
            else {
                result.append(1, *it);   // append this character
            }

            // copy while in character set
            while (++it != end) {
                result.append(1, *it);
                if (*it == ']')
                    break;
            }

            if (it == end || *it != ']') {
                HPX_THROWS_IF(ec, bad_parameter, "regex_from_character_set",
                    "Invalid pattern (missing closing ']') at: " +
                        std::string(start, end));
                return "";
            }

            return result;
        }

        std::string regex_from_pattern(std::string const& pattern, error_code& ec)
        {
            std::string result;
            std::string::const_iterator end = pattern.end();
            for (std::string::const_iterator it = pattern.begin(); it != end; ++it)
            {
                char c = *it;
                switch (c) {
                case '*':
                    result.append(".*");
                    break;

                case '?':
                    result.append(1, '.');
                    break;

                case '[':
                    {
                        std::string r = regex_from_character_set(it, end, ec);
                        if (ec) return "";
                        result.append(r);
                    }
                    break;

                case '\\':
                    if (++it == end) {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "regex_from_pattern",
                            "Invalid escape sequence at: " + pattern);
                        return "";
                    }
                    result.append(1, *it);
                    break;

                default:
                    result.append(1, c);
                    break;
                }
            }
            return result;
        }
    }

    /// \brief Call the supplied function for the given registered counter type.
    counter_status registry::discover_counter_type(
        std::string const& fullname,
        discover_counter_func discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(fullname, type_name, ec);
        if (!status_is_valid(status)) return status;

        if (type_name.find_first_of("*?[]") == std::string::npos)
        {
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

                HPX_THROWS_IF(ec, bad_parameter,
                    "registry::discover_counter_type",
                    boost::str(boost::format(
                        "unknown counter type: %s, known counter "
                        "types: \n%s") % type_name % types));
                return status_counter_type_unknown;
            }

            if (mode == discover_counters_full)
            {
                using hpx::util::placeholders::_1;
                discover_counter = hpx::util::bind(&expand_counter_info, _1,
                    discover_counter, std::ref(ec));
            }

            counter_info info = (*it).second.info_;
            info.fullname_ = fullname;

            if (!(*it).second.discover_counters_.empty() &&
                !(*it).second.discover_counters_(info, discover_counter, mode, ec))
            {
                return status_invalid_data;
            }
        }
        else
        {
            std::string str_rx(detail::regex_from_pattern(type_name, ec));
            if (ec) return status_invalid_data;

            if (mode == discover_counters_full)
            {
                using hpx::util::placeholders::_1;
                discover_counter = hpx::util::bind(&expand_counter_info, _1,
                    discover_counter, std::ref(ec));
            }

            // split name
            counter_path_elements p;
            get_counter_path_elements(fullname, p, ec);
            if (ec) return status_invalid_data;

            bool found_one = false;
            boost::regex rx(str_rx, boost::regex::perl);

            counter_type_map_type::const_iterator end = countertypes_.end();
            for (counter_type_map_type::const_iterator it = countertypes_.begin();
                 it != end; ++it)
            {
                if (!boost::regex_match((*it).first, rx))
                    continue;
                found_one = true;

                // propagate parameters
                counter_info info = (*it).second.info_;
                if (!p.parameters_.empty())
                    info.fullname_ += "@" + p.parameters_;

                if (!(*it).second.discover_counters_.empty() &&
                    !(*it).second.discover_counters_(info,
                        discover_counter, mode, ec))
                {
                    return status_invalid_data;
                }
            }

            if (!found_one) {
                // compose a list of known counter types
                std::string types;
                counter_type_map_type::const_iterator end = countertypes_.end();
                for (counter_type_map_type::const_iterator it = countertypes_.begin();
                     it != end; ++it)
                {
                    types += "  " + (*it).first + "\n";
                }

                HPX_THROWS_IF(ec, bad_parameter, "registry::discover_counter_type",
                    boost::str(boost::format(
                        "counter type %s does not match any known type, "
                        "known counter types: \n%s") % type_name % types));
                return status_counter_type_unknown;
            }
        }

        if (&ec != &throws)
            ec = make_success_code();

        return status_valid_data;
    }

    /// \brief Call the supplied function for all registered counter types.
    counter_status registry::discover_counter_types(
        discover_counter_func discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        // Introducing this temporary silence a report about a potential memory
        // from clang's static analyzer
        discover_counter_func discover_counter_;
        if (mode == discover_counters_full)
        {
            using hpx::util::placeholders::_1;
            discover_counter_ = hpx::util::bind(&expand_counter_info, _1,
                std::move(discover_counter), std::ref(ec));
        }
        else
        {
            discover_counter_ = std::move(discover_counter);
        }

        for (counter_type_map_type::value_type const& d : countertypes_)
        {
            if (!d.second.discover_counters_.empty() &&
                !d.second.discover_counters_(
                      d.second.info_, discover_counter_, mode, ec))
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
        counter_info const& info, create_counter_func& func,
        error_code& ec) const
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::const_iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            // compose a list of known counter types
            std::string types;
            counter_type_map_type::const_iterator end = countertypes_.end();
            for (counter_type_map_type::const_iterator it = countertypes_.begin();
                    it != end; ++it)
            {
                types += "  " + (*it).first + "\n";
            }

            HPX_THROWS_IF(ec, bad_parameter,
                "registry::get_counter_create_function",
                boost::str(boost::format(
                    "counter type %s is not defined, known counter "
                    "types: \n%s") % type_name % types));
            return status_counter_type_unknown;
        }

        if ((*it).second.create_counter_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter,
                "registry::get_counter_create_function",
                boost::str(boost::format(
                    "counter type %s has no associated create "
                    "function") % type_name));
            return status_invalid_data;
        }

        func = (*it).second.create_counter_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::get_counter_discovery_function(
        counter_info const& info, discover_counters_func& func,
        error_code& ec) const
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::const_iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter,
                "registry::get_counter_discovery_function",
                boost::str(boost::format(
                    "counter type %s is not defined") % type_name));
            return status_counter_type_unknown;
        }

        if ((*it).second.discover_counters_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter,
                "registry::get_counter_discovery_function",
                boost::str(boost::format(
                    "counter type %s has no associated discovery "
                    "function") % type_name));
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
    boost::int64_t wrap_counter(boost::int64_t* p, bool reset)
    {
        boost::int64_t result = *p;
        *p = 0;
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status registry::create_raw_counter_value(counter_info const& info,
        boost::int64_t* countervalue, naming::gid_type& id, error_code& ec)
    {
        using util::placeholders::_1;
        hpx::util::function_nonser<boost::int64_t(bool)> func(
            util::bind(wrap_counter, countervalue, _1));
        return create_raw_counter(info, func, id, ec);
    }

    static boost::int64_t
    wrap_raw_counter(hpx::util::function_nonser<boost::int64_t()> const& f, bool)
    {
        return f();
    }

    static std::vector<boost::int64_t>
    wrap_raw_values_counter(
        hpx::util::function_nonser<std::vector<boost::int64_t>()> const& f, bool)
    {
        return f();
    }

    counter_status registry::create_raw_counter(counter_info const& info,
        hpx::util::function_nonser<boost::int64_t()> const& f, naming::gid_type& id,
        error_code& ec)
    {
        using util::placeholders::_1;
        hpx::util::function_nonser<boost::int64_t(bool)> func(
            util::bind(&wrap_raw_counter, f, _1));
        return create_raw_counter(info, func, id, ec);
    }

    counter_status registry::create_raw_counter(counter_info const& info,
        hpx::util::function_nonser<boost::int64_t(bool)> const& f, naming::gid_type& id,
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
            typedef components::component<server::raw_counter> counter_t;
            id = components::server::construct<counter_t>(complemented_info, f);

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

    counter_status registry::create_raw_counter(counter_info const& info,
        hpx::util::function_nonser<std::vector<boost::int64_t>()> const& f,
        naming::gid_type& id, error_code& ec)
    {
        using util::placeholders::_1;
        hpx::util::function_nonser<std::vector<boost::int64_t>(bool)> func(
            util::bind(&wrap_raw_values_counter, f, _1));
        return create_raw_counter(info, func, id, ec);
    }

    counter_status registry::create_raw_counter(counter_info const& info,
        hpx::util::function_nonser<std::vector<boost::int64_t>(bool)> const& f,
        naming::gid_type& id, error_code& ec)
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
        if (counter_histogram != (*it).second.info_.type_ ||
            counter_histogram != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_raw_counter",
                "invalid counter type requested (only counter_histogram "
                "is supported)");
            return status_counter_type_unknown;
        }

        // make sure parent instance name is set properly
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, (*it).second.info_, ec);
        if (ec) return status_invalid_data;

        // create the counter as requested
        try {
            typedef components::component<server::raw_values_counter> counter_t;
            id = components::server::construct<counter_t>(complemented_info, f);

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
                    typedef components::component<server::elapsed_time_counter>
                        counter_t;
                    id = components::server::construct<counter_t>(complemented_info);
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

    /// \brief Create a new statistics performance counter instance based
    ///        on given base counter name and given base time interval
    ///        (milliseconds).
    counter_status registry::create_statistics_counter(
        counter_info const& info, std::string const& base_counter_name,
        std::vector<boost::int64_t> const& parameters,
        naming::gid_type& gid, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_statistics_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the requested counter type is supported
        if (counter_aggregating != (*it).second.info_.type_ ||
            counter_aggregating != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_statistics_counter",
                "invalid counter type requested \
                 (only counter_aggregating is supported)");
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
            // extract parameters
            boost::uint64_t sample_interval = 1000;   // default sampling interval
            if (!parameters.empty())
                sample_interval = parameters[0];

            // create base counter only if it does not exist yet
            if (p.countername_ == "average") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::mean>
                > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, 0);
            }
            else if (p.countername_ == "stddev") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::variance>
                > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, 0);
            }
            else if (p.countername_ == "rolling_average") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::rolling_mean>
                > counter_t;

                boost::uint64_t window_size = 10;   // default rolling window size
                if (parameters.size() > 1)
                    window_size = parameters[1];

                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, window_size);
            }
            else if (p.countername_ == "median") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::median>
                > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, 0);
            }
            else if (p.countername_ == "max") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::max>
                > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, 0);
            }
            else if (p.countername_ == "min") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::statistics_counter<
                        boost::accumulators::tag::min>
                > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_name, sample_interval, 0);
            }
            else {
                HPX_THROWS_IF(ec, bad_parameter,
                    "registry::create_statistics_counter",
                    "invalid counter type requested: " + p.countername_);
                return status_counter_type_unknown;
            }
        }
        catch (hpx::exception const& e) {
            gid = naming::invalid_gid;        // reset result
            if (&ec == &throws)
                throw;

            ec = make_error_code(e.get_error(), e.what());
            LPCS_(warning) << (
                boost::format("failed to create statistics counter %s (%s)") %
                    complemented_info.fullname_ % e.what());
            return status_invalid_data;
        }

        LPCS_(info) << (boost::format("statistics counter %s created at %s") %
            complemented_info.fullname_ % gid);

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a new arithmetics performance counter instance based
    ///        on given base counter name and given base time interval
    ///        (milliseconds).
    counter_status registry::create_arithmetics_counter(
        counter_info const& info, std::vector<std::string> const& base_counter_names,
        naming::gid_type& gid, error_code& ec)
    {
        // create canonical type name
        std::string type_name;
        counter_status status = get_counter_type_name(info.fullname_, type_name, ec);
        if (!status_is_valid(status)) return status;

        counter_type_map_type::iterator it = locate_counter_type(type_name);
        if (it == countertypes_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_arithmetics_counter",
                boost::str(boost::format("unknown counter type %s") % type_name));
            return status_counter_type_unknown;
        }

        // make sure the requested counter type is supported
        if (counter_aggregating != (*it).second.info_.type_ ||
            counter_aggregating != info.type_)
        {
            HPX_THROWS_IF(ec, bad_parameter, "registry::create_arithmetics_counter",
                "invalid counter type requested \
                 (only counter_aggregating is supported)");
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
            if (p.countername_ == "add") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::arithmetics_counter<
                        std::plus<double> > > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_names);
            }
            else if (p.countername_ == "subtract") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::arithmetics_counter<
                        std::minus<double> > > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_names);
            }
            else if (p.countername_ == "multiply") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::arithmetics_counter<
                        std::multiplies<double> > > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_names);
            }
            else if (p.countername_ == "divide") {
                typedef hpx::components::component<
                    hpx::performance_counters::server::arithmetics_counter<
                        std::divides<double> > > counter_t;
                gid = components::server::construct<counter_t>(
                    complemented_info, base_counter_names);
            }
            else {
                HPX_THROWS_IF(ec, bad_parameter,
                    "registry::create_arithmetics_counter",
                    "invalid counter type requested: " + p.countername_);
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
        agas::register_name(launch::sync, name, id, ec);
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
        agas::unregister_name(launch::sync, name, ec);
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
// //                     components::component<server::raw_counter>
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


