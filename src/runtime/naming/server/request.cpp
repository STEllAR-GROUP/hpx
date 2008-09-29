//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/config.hpp>
#include <hpx/runtime/naming/server/request.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    namespace strings
    {
        const char* const command_names[] = 
        {
            "command_getprefix",
            "command_getprefixes",
            "command_get_component_id",
            "command_getidrange",
            "command_bind_range",
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

    char const* const get_command_name(int cmd)
    {
        if (cmd >= command_firstcommand && cmd < command_lastcommand)
            return strings::command_names[cmd];
        return "<unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    // debug support for a request class
    std::ostream& operator<< (std::ostream& os, request const& req)
    {
        os << get_command_name(req.command_) << ": ";

        switch (req.command_) {
        case command_resolve:
            os << "id" << req.id_ << " ";
            break;

        case command_bind_range:
            os << "id" << req.id_ << " ";
            if (req.count_ != 1)
                os << "count:" << std::dec << req.count_ << " ";
            os << "addr(" << req.addr_ << ") ";
            if (req.offset_ != 0)
                os << "offset:" << std::dec << req.offset_ << " ";
            break;

        case command_unbind_range:
            os << "id" << req.id_ << " ";
            if (req.count_ != 1)
                os << "count:" << std::dec << req.count_ << " ";
            break;

        case command_get_component_id:
        case command_queryid:
        case command_unregisterid: 
            os << "name(\"" << req.name_ << "\") ";
            break;

        case command_registerid:
            os << "id" << req.id_ << " ";
            os << "name(\"" << req.name_ << "\") ";
            break;

        case command_getprefix:
            os << "site(" << req.site_ << ") ";
            break;

        case command_getidrange:
            os << "site(" << req.site_ << ") ";
            os << "count:" << std::dec << req.count_ << " ";
            break;

        case command_getprefixes:
        case command_statistics_count:
        case command_statistics_mean:
        case command_statistics_moment2:
        default:
            break;
        }
        return os;
    }

}}}

