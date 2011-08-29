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
        {
            // map localhost to loopback ip address (that's a quick hack 
            // which will be removed as soon as we figure out why name 
            // resolution does not handle this anymore)
            mappings_["localhost"] = "127.0.0.1";
            if (debug_) {
                std::cerr << "inserting mapping: localhost:127.0.0.1" 
                          << std::endl;
            }
        }

        void init_from_file(std::string mappingsfile)
        {
            if (!mappingsfile.empty()) {
                if (debug_)
                    std::cerr << "opened: " << mappingsfile << std::endl;

                std::ifstream ifs(mappingsfile.c_str());
                if (ifs.is_open()) {
                    std::string line;
                    while (std::getline(ifs, line)) {
                        if (!line.empty()) {
                            if (debug_)
                                std::cerr << "read: " << line << std::endl;

                            std::string::size_type p = line.find_first_of(" \t,:");
                            if (p != std::string::npos) {
                                if (debug_) {
                                    std::cerr << "inserting mapping: " 
                                              << line.substr(0, p) << ":"
                                              << line.substr(p+1) << std::endl;
                                }
                                mappings_[line.substr(0, p)] = line.substr(p+1);
                            }
                            else if (debug_) {
                                std::cerr << "failed to insert mapping: " 
                                          << line << std::endl;
                            }
                        }
                    }
                }
                else if (debug_) {
                    std::cerr << "failed opening: " << mappingsfile << std::endl;
                }
            }
        }

        std::string map(std::string const& host_name) const
        {
            std::map<std::string, std::string>::const_iterator it = 
                mappings_.find(host_name);

            if (it != mappings_.end()) {
                if (debug_) {
                    std::cerr << "found mapping: " << host_name 
                              << ":" << (*it).second << std::endl;
                }
                return (*it).second;
            }

            if (debug_) 
                std::cerr << "no mapping found : " << host_name << std::endl;
            return host_name;
        }

        std::map<std::string, std::string> mappings_;
        bool debug_;
    };
}}

#endif
