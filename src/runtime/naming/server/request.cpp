//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx_fwd.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/config.hpp>
#include <hpx/runtime/naming/server/request.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/io/ios_state.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    namespace strings
    {
        const char* const command_names[] = 
        {
            "command_getprefix",
            "command_getconsoleprefix",
            "command_getprefixes",
            "command_getprefix_for_site",
            "command_get_component_id",
            "command_register_factory",
            "command_getidrange",
            "command_bind_range",
            "command_incref",
            "command_decref",
            "command_unbind_range",
            "command_resolve",
            "command_queryid",
            "command_registerid",
            "command_unregisterid",
            "command_statistics_count",
            "command_statistics_mean",
            "command_statistics_moment2",
            ""
        };
    }

    char const* get_command_name(int cmd)
    {
        if (cmd >= command_firstcommand && cmd < command_lastcommand)
            return strings::command_names[cmd];
        return "<unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    // debug support for a request class
    std::ostream& operator<< (std::ostream& os, request const& req)
    {
        boost::io::ios_flags_saver ifs(os); 
        os << get_command_name(req.command_) << ": ";

        switch (req.command_) {
        case command_resolve:
            os << "id" << req.id_ << " ";
            break;

        case command_bind_range:
            os << "id" << req.id_ << " ";
            if (req.count_ != 1)
                os << "count(" << req.count_ << ") ";
            os << "addr(" << req.addr_ << ") ";
            if (req.offset_ != 0)
                os << "offset(" << req.offset_ << ") ";
            break;

        case command_incref:
        case command_decref:
        case command_unbind_range:
            os << "id" << req.id_ << " ";
            if (req.count_ != 1)
                os << "count(" << req.count_ << ") ";
            break;

        case command_get_component_id:
        case command_queryid:
        case command_unregisterid: 
            os << "name(\"" << req.name_ << "\") ";
            break;

        case command_register_factory:
        case command_registerid:
            os << "id" << req.id_ << " ";
            os << "name(\"" << req.name_ << "\") ";
            break;

        case command_getprefix:
            os << "site(" << req.site_ << "), ";
            os << "isconsole(" << std::boolalpha << req.isconsole_ << ") ";
            break;

        case command_getprefix_for_site:
            os << "site(" << req.site_ << "), ";
            break;

        case command_getidrange:
            os << "site(" << req.site_ << ") ";
            os << "count(" << req.count_ << ") ";
            break;

        case command_getprefixes:
            os << "type(" << components::get_component_type_name(req.type_) << ") ";
            break;

        case command_getconsoleprefix:
        case command_statistics_count:
        case command_statistics_mean:
        case command_statistics_moment2:
        default:
            break;
        }
        return os;
    }

}}}

#endif
