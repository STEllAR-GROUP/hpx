//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/server/reply.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::server::reply, HPX_REPLY_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::server::reply, boost::serialization::track_never)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    namespace status_strings 
    {
        char const* const success = 
            "hpx: dgas: success";
        char const* const repeated_request = 
            "hpx: dgas: success (repeated request)";
        char const* const no_success = 
            "hpx: dgas: no success";
        char const* const internal_server_error = 
            "hpx: dgas: internal server error";
        char const* const service_unavailable = 
            "hpx: dgas: service unavailable";
        char const* const invalid_status = 
            "hpx: dgas: corrupted internal status";
        char const* const bad_parameter = 
            "hpx: dgas: bad parameter";
        char const* const bad_request = 
            "hpx: dgas: ill formatted request or unknown command";
        char const* const out_of_memory = 
            "hpx: dgas: internal server error (out of memory)";
        char const* const unknown_version = 
            "hpx: dgas: ill formatted request (unknown version)";
    }

    char const* const get_error_text(error status)
    {
        switch (status) {
        case hpx::success:               return status_strings::success;
        case hpx::no_success:            return status_strings::no_success;
        case hpx::service_unavailable:   return status_strings::service_unavailable;
        case hpx::invalid_status:        return status_strings::invalid_status;
        case hpx::bad_parameter:         return status_strings::bad_parameter;
        case hpx::bad_request:           return status_strings::bad_request;
        case hpx::out_of_memory:         return status_strings::out_of_memory;
        case hpx::version_unknown:       return status_strings::unknown_version;
        case hpx::repeated_request:      return status_strings::repeated_request;
        case hpx::internal_server_error: return status_strings::internal_server_error;
        default:
            break;
        }
        return status_strings::internal_server_error;
    }

    // Streaming operator, used by logging
    std::ostream& operator<< (std::ostream& os, reply const& rep)
    {
        os << get_command_name(rep.command_) << ": ";

        if (rep.status_ != success && rep.status_ != repeated_request) {
            os << "status(" << rep.error_ << ") ";
        }
        else {
            switch (rep.command_) {
            case command_resolve:
            case command_unbind_range:
                os << "addr" << rep.address_ << " ";
                break;

            case command_getidrange:
                os << "range(" << rep.lower_bound_ << ", "
                   << rep.upper_bound_ << ") ";
                break;

            case command_getprefix:
                os << "prefix" << rep.lower_bound_;
                break;

            case command_queryid:
                os << "id" << rep.id_ << " ";
                break;

            case command_get_component_id:
                os << "type(" << (int)rep.type_ << ") ";
                break;

            case command_statistics_count:
            case command_statistics_mean:
            case command_statistics_moment2:
                break;

            case command_getprefixes:
                {
                    os << "prefixes(";
                    typedef std::vector<boost::uint32_t>::const_iterator iterator;
                    iterator end = rep.get_prefixes().end(); 
                    for (iterator it = rep.get_prefixes().begin(); it != end; )
                    {
                        os << std::hex << get_id_from_prefix(*it);
                        if (++it != end)
                            os << ", ";
                    }
                    os << ")";
                }
                break;

            case command_bind_range:
            case command_registerid: 
            case command_unregisterid: 
            default:
                break;  // nothing additional to be received
            }
            if (rep.status_ == repeated_request)
                os << " (" << error_names[repeated_request] << ")";
        }
        return os;
    }

}}}
