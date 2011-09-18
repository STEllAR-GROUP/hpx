//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/format.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Initialization support for the performance counter actions
HPX_REGISTER_ACTION_EX(
    hpx::performance_counters::server::base_performance_counter::get_counter_info_action, 
    performance_counter_get_counter_info_action);
HPX_REGISTER_ACTION_EX(
    hpx::performance_counters::server::base_performance_counter::get_counter_value_action, 
    performance_counter_get_counter_value_action);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_info>::set_result_action, 
    set_result_action_counter_info);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_value>::set_result_action, 
    set_result_action_counter_value);

HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::base_performance_counter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    template hpx::performance_counters::counter_info const& 
    continuation::trigger(hpx::performance_counters::counter_info const& arg0);

    template hpx::performance_counters::counter_value const& 
    continuation::trigger(hpx::performance_counters::counter_value const& arg0);
}}

///////////////////////////////////////////////////////////////////////////////
// HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
//     hpx::performance_counters::server::base_performance_counter);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_info>,
    hpx::components::component_base_lco_with_value);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_value>,
    hpx::components::component_base_lco_with_value);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters 
{
    /// \brief Create a full name of a counter from the contents of the given 
    ///        \a counter_path_elements instance.
    counter_status get_counter_name(counter_path_elements const& path, 
        std::string& result, error_code& ec)
    {
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_name", 
                    "empty object name");
            return status_invalid_data;
        }

        result = "/";
        if (!path.objectname_.empty())
            result += path.objectname_;

        if (!path.parentinstancename_.empty() || !path.instancename_.empty())
        {
            result += "(";
            if (!path.parentinstancename_.empty())
            {
                result += path.parentinstancename_;
                result += "/";
            }
            if (path.instancename_.empty()) {
                HPX_THROWS_IF(ec, bad_parameter, "get_counter_name", 
                        "empty instance name");
                return status_invalid_data;
            }

            result += path.instancename_;
            if (-1 != path.instanceindex_)
            {
                result += "#";
                result += boost::lexical_cast<std::string>(path.instanceindex_);
            }
            result += ")";
        }
        if (path.countername_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_name", 
                    "empty counter name");
            return status_invalid_data;
        }

        result += "/";
        result += path.countername_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a full name of a counter from the contents of the given 
    ///        \a counter_path_elements instance.
    counter_status get_counter_name(counter_type_path_elements const& path, 
        std::string& result, error_code& ec)
    {
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_type_name", 
                    "empty object name");
            return status_invalid_data;
        }

        result = "/";
        if (!path.objectname_.empty())
            result += path.objectname_;

        if (path.countername_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_type_name", 
                    "empty counter name");
            return status_invalid_data;
        }

        result += "/";
        result += path.countername_;

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Fill the given \a counter_type_path_elements instance from the 
    ///        given full name of a counter
    ///
    ///    /objectname(parentinstancename/instancename#instanceindex)/countername
    counter_status get_counter_path_elements(std::string const& name, 
        counter_type_path_elements& path, error_code& ec)
    {
        if (name.empty() || name[0] != '/') {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty name parameter");
            return status_invalid_data;
        }

        std::string::size_type p = name.find_first_of("(/", 1);
        if (p == std::string::npos) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "expected delimiter: '(' or '/'");
            return status_invalid_data;
        }

        // object name is the first part of the full name
        path.objectname_ = name.substr(1, p-1);
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty object name");
            return status_invalid_data;
        }

        if (name[p] == '(') {
            std::string::size_type p1 = name.find_first_of(")", p);
            if (p1 == std::string::npos || p1 >= name.size()) {
                HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                        "mismatched parenthesis, expected: ')'");
                return status_invalid_data;
            }
            p = p1+1;
        }

        // counter name is always the last part of the full name
        path.countername_ = name.substr(p+1);
        if (path.countername_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty counter name");
            return status_invalid_data;
        }

        // we now allow the countername to be hierarchical 
//         if (path.countername_.find_first_of("/") != std::string::npos) {
//             HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
//                     "more than one path element in counter name");
//             return status_invalid_data;
//         }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Fill the given \a counter_path_elements instance from the given 
    ///        full name of a counter
    ///
    ///    /objectname(parentinstancename/instancename#instanceindex)/countername
    counter_status get_counter_path_elements(std::string const& name, 
        counter_path_elements& path, error_code& ec)
    {
        if (name.empty() || name[0] != '/') {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty name parameter");
            return status_invalid_data;
        }

        std::string::size_type p = name.find_first_of("(/", 1);
        if (p == std::string::npos) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "expected delimiter: '(' or '/'");
            return status_invalid_data;
        }

        // object name is the first part of the full name
        path.objectname_ = name.substr(1, p-1);
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty object name");
            return status_invalid_data;
        }

        if (name[p] == '(') {
            std::string::size_type p1 = name.find_first_of(")", p);
            if (p1 == std::string::npos || p1 >= name.size()) {
                HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                        "mismatched parenthesis, expected: ')'");
                return status_invalid_data;
            }

            // instance name is in between p and p1
            std::string::size_type p2 = name.find_last_of("/#", p1);
            if (p2 == std::string::npos) {
                HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                        "expected delimiter: '/' or '#'");
                return status_invalid_data;
            }

            if (name[p2] == '#') {
                // instance index
                try {
                    path.instanceindex_ = 
                        boost::lexical_cast<boost::uint32_t>(name.substr(p2+1, p1-p2-1));
                }
                catch (boost::bad_lexical_cast const& /*e*/) {
                    HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                            "bogus instance index, expected: integer");
                    return status_invalid_data;
                }

                std::string::size_type p3 = name.find_last_of("/", p2);
                if (p3 != std::string::npos && p3 > p) {
                    // instance name and parent instance name
                    path.parentinstancename_ = name.substr(p+1, p3-p-1);
                    if (path.parentinstancename_.empty()) {
                        HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                                "empty parent instance name");
                        return status_invalid_data;
                    }
                    path.instancename_ = name.substr(p3+1, p2-p3-1);
                    if (path.instancename_.empty()) {
                        HPX_THROWS_IF(ec, bad_parameter, 
                            "get_counter_path_elements", "empty instance name");
                        return status_invalid_data;
                    }
                }
                else {
                    // instance name only
                    path.instancename_ = name.substr(p+1, p2-p-1);
                    if (path.instancename_.empty()) {
                        HPX_THROWS_IF(ec, bad_parameter, 
                            "get_counter_path_elements", "empty instance name");
                        return status_invalid_data;
                    }
                }
            }
            else if (p2 > p) {
                // instance name and parent instance name
                path.parentinstancename_ = name.substr(p+1, p2-p-1);
                if (path.parentinstancename_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter, 
                        "get_counter_path_elements", "empty parent instance name");
                    return status_invalid_data;
                }
                path.instancename_ = name.substr(p2+1, p1-p2-1);
                if (path.instancename_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter, 
                        "get_counter_path_elements", "empty instance name");
                    return status_invalid_data;
                }
            }
            else {
                // just instance name
                path.instancename_ = name.substr(p+1, p1-p-1);
                if (path.instancename_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter, 
                        "get_counter_path_elements", "empty instance name");
                    return status_invalid_data;
                }
            }
            p = p1+1;
        }

        // counter name is always the last part of the full name
        path.countername_ = name.substr(p+1);
        if (path.countername_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
                    "empty counter name");
            return status_invalid_data;
        }

        // we now allow the countername to be hierarchical 
//         if (path.countername_.find_first_of("/") != std::string::npos) {
//             HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements", 
//                     "more than one path element in counter name");
//             return status_invalid_data;
//         }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Return the counter type name from a given full instance name
    counter_status get_counter_type_name(std::string const& name, 
        std::string& type_name, error_code& ec)
    {
        counter_type_path_elements p;

        counter_status status = get_counter_path_elements(name, p, ec);
        if (status_valid_data != status) return status;

        return get_counter_name(p, type_name, ec);
    }

    /// \brief Return the counter type name from a given full instance name
    counter_status get_counter_name(std::string const& name, 
        std::string& countername, error_code& ec)
    {
        counter_path_elements p;

        counter_status status = get_counter_path_elements(name, p, ec);
        if (status_valid_data != status) return status;

        return get_counter_name(p, countername, ec);
    }

    /// \brief Complement the counter info if parent instance name is missing
    counter_status complement_counter_info(counter_info& info, 
        counter_info const& type_info, error_code& ec)
    {
        info.type_ = type_info.type_;
        if (info.helptext_.empty())
            info.helptext_ = type_info.helptext_;
        return complement_counter_info(info, ec);
    }

    counter_status complement_counter_info(counter_info& info, 
        error_code& ec)
    {
        counter_path_elements p;

        counter_status status = get_counter_path_elements(info.fullname_, p, ec);
        if (status_valid_data != status) return status;

        if (p.parentinstancename_.empty()) {
            p.parentinstancename_ = boost::str(boost::format("[locality#%d]") % 
                applier::get_applier().get_prefix_id());
        }

        return get_counter_name(p, info.fullname_, ec);
    }
}}

