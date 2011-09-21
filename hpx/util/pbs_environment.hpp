//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM)
#define HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM

#include <hpx/hpx_fwd.hpp>

#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/asio/ip/host_name.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////
    // Try to retrieve PBS related settings from the environment
    struct pbs_environment
    {
        typedef std::map<std::string, std::size_t> node_map_type;

        // the constructor tries to read from a PBS node-file, filling our
        // map of nodes and thread counts
        pbs_environment(bool debug = false) : debug_(debug) {}

        // this function tries to read from a PBS node-file, filling our
        // map of nodes and thread counts
        std::string init_from_file(std::string nodefile)
        {
            // read node file
            if (nodefile.empty()) {
                char* pbs_nodefile_env = std::getenv("PBS_NODEFILE");
                if (pbs_nodefile_env) 
                    nodefile = pbs_nodefile_env;
            }

            if (!nodefile.empty()) {
                std::ifstream ifs(nodefile.c_str());
                if (ifs.is_open()) {
                    if (debug_)
                        std::cerr << "opened: " << nodefile << std::endl;
                    std::string line;
                    while (std::getline(ifs, line)) {
                        if (!line.empty()) {
                            if (debug_)
                                std::cerr << "read: " << line << std::endl;
                            if (nodes_.empty())
                                agas_node_ = line;
                            ++nodes_[line];
                        }
                    }
                }
                else if (debug_) {
                    std::cerr << "failed opening: " << nodefile << std::endl;
                }
            }

            return nodefile;
        }

        // this function initializes the map of nodes from the given (space 
        // separated) list of nodes
        std::string init_from_nodelist(std::vector<std::string> const& nodes)
        {
            if (debug_)
                std::cerr << "got node list" << std::endl;

            std::string nodes_list;
            BOOST_FOREACH(std::string const& s, nodes) 
            {
                if (!s.empty()) {
                    if (debug_)
                        std::cerr << "extracted: " << s << std::endl;
                    if (nodes_.empty())
                        agas_node_ = s;
                    ++nodes_[s];
                    nodes_list += s + ' ';
                }
            }

            return nodes_list;
        }

        // The number of threads is either one (if no PBS information was 
        // found), or it is the same as the number of times this node has 
        // been listed in the node file.
        std::size_t retrieve_number_of_threads() const
        {
            char* pbs_num_ppn = std::getenv("PBS_NUM_PPN");
            if (pbs_num_ppn) {
                try {
                    std::string value(pbs_num_ppn);
                    std::size_t result = boost::lexical_cast<std::size_t>(value);
                    if (debug_) {
                        std::cerr << "retrieve_number_of_threads: " << result 
                                  << std::endl;
                    }
                    return result;
                }
                catch (boost::bad_lexical_cast const&) {
                    ; // just ignore the error
                }
            }

            // fall back to counting the number of occurrences of this node 
            // in the node-file
            node_map_type::const_iterator it = nodes_.find(host_name());
            std::size_t result = it != nodes_.end() ? (*it).second : 1;
            if (debug_) {
                std::cerr << "retrieve_number_of_threads: " << result 
                          << std::endl;
            }
            return result;
        }

        // The number of localities is either one (if no PBS information 
        // was found), or it is the same as the number of distinct node 
        // names listed in the node file.
        std::size_t retrieve_number_of_localities() const
        {
            std::size_t result = nodes_.empty() ? 1 : nodes_.size();
            if (debug_) {
                std::cerr << "retrieve_number_of_localities: " << result 
                          << std::endl;
            }
            return result;
        }

        // Try to retrieve the node number from the PBS environment
        std::size_t retrieve_node_number() const
        {
            char* pbs_nodenum = std::getenv("PBS_NODENUM");
            if (pbs_nodenum) {
                try {
                    std::string value(pbs_nodenum);
                    std::size_t result = boost::lexical_cast<std::size_t>(value);
                    if (debug_) {
                        std::cerr << "retrieve_node_number: " << result 
                                  << std::endl;
                    }
                    return result;
                }
                catch (boost::bad_lexical_cast const&) {
                    ; // just ignore the error
                }
            }
            if (debug_)
                std::cerr << "retrieve_node_number: -1" << std::endl;
            return std::size_t(-1);
        }

        // This helper function returns the host name of this node.
        static std::string strip_local(std::string const& name)
        {
            std::string::size_type pos = name.find(".local");
            if (pos != std::string::npos)
                return name.substr(0, pos);
            return name;
        }

        static std::string host_name() 
        {
            return strip_local(boost::asio::ip::host_name());
        }

        std::string host_name(std::string const& def_hpx_name) const
        {
            std::string host = nodes_.empty() ? def_hpx_name : host_name();
            if (debug_) 
                std::cerr << "host_name: " << host << std::endl;
            return host;
        }

        // We arbitrarily select the first host listed in the node file to
        // host the AGAS server.
        std::string agas_host_name(std::string const& def_agas) const
        {
            std::string host = agas_node_.empty() ? def_agas : agas_node_; 
            if (debug_) 
                std::cerr << "agas host_name: " << host << std::endl;
            return host;
        }
        
        std::string agas_node_;
        std::map<std::string, std::size_t> nodes_;
        bool debug_;
    };
}}

#endif
