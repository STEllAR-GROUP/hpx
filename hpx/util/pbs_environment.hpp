//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM)
#define HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM

#include <hpx/hpx_fwd.hpp>

#include <map>
#include <cstdlib>

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
        pbs_environment(std::string pbs_nodefile)
        {
            // read node file
            if (pbs_nodefile.empty()) {
                char* pbs_nodefile_env = std::getenv("PBS_NODEFILE");
                if (pbs_nodefile_env) 
                    pbs_nodefile = pbs_nodefile_env;
            }

            if (!pbs_nodefile.empty()) {
                std::ifstream ifs(pbs_nodefile.c_str());
                if (ifs.is_open()) {
                    std::string line;
                    while (std::getline(ifs, line))
                        if (!line.empty())
                            ++nodes_[line];
                }
            }
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
                    return boost::lexical_cast<std::size_t>(value);
                }
                catch (boost::bad_lexical_cast const&) {
                    ; // just ignore the error
                }
            }
            // fall back to counting the number of occurrences of this node 
            // in the node-file
            node_map_type::const_iterator it = nodes_.find(host_name());
            return it != nodes_.end() ? (*it).second : 1;
        }

        // The number of localities is either one (if no PBS information 
        // was found), or it is the same as the number of distinct node 
        // names listed in the node file.
        std::size_t retrieve_number_of_localities() const
        {
            return nodes_.empty() ? 1 : nodes_.size();
        }

        // Try to retrieve the node number from the PBS environment
        std::size_t retrieve_node_number() const
        {
            char* pbs_nodenum = std::getenv("PBS_NODENUM");
            if (pbs_nodenum) {
                try {
                    std::string value(pbs_nodenum);
                    return boost::lexical_cast<std::size_t>(value);
                }
                catch (boost::bad_lexical_cast const&) {
                    ; // just ignore the error
                }
            }
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
            return nodes_.empty() ? 
                def_hpx_name : strip_local(boost::asio::ip::host_name());
        }

        // We arbitrarily select the first host listed in the node file to
        // host the AGAS server.
        std::string agas_host_name(std::string const& def_agas_name) const
        {
            return nodes_.empty() ? 
                def_agas_name : strip_local((*nodes_.begin()).first);
        }

        std::map<std::string, std::size_t> nodes_;
    };
}}

#endif
