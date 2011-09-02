//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_MAP_HOSTNAMES_AUG_29_2011_1257PM)
#define HPX_UTIL_MAP_HOSTNAMES_AUG_29_2011_1257PM

#include <hpx/hpx_fwd.hpp>

#include <map>
#include <iostream>
#include <fstream>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // Try to map a given host name based on the list of mappings read from a 
    // file
    struct map_hostnames
    {
        map_hostnames(bool debug = false) 
          : debug_(debug) 
        {}

        void use_suffix(std::string const& suffix)
        {
            suffix_ = suffix;
        }

        std::string map(std::string const& host_name, boost::uint16_t port) const
        {
            if (host_name == "localhost") {
                // map local host to loopback ip address (that's a quick hack 
                // which will be removed as soon as we figure out why name 
                // resolution does not handle this anymore)
                if (debug_) {
                    std::cerr << "resolved: 'localhost' to: 127.0.0.1" 
                              << std::endl;
                }
                return "127.0.0.1";
            }

            // do full host name resolution
            boost::asio::io_service io_service;
            boost::asio::ip::tcp::endpoint ep =
                util::resolve_hostname(host_name, port, io_service);

            std::string resolved_addr(util::get_endpoint_name(ep));
            if (debug_) {
                std::cerr << "resolved: '" << host_name << "' to: " 
                          << resolved_addr << std::endl;
            }
            return resolved_addr;
        }

        std::string suffix_;
        bool debug_;
    };
}}

#endif
