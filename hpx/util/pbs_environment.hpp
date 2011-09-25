//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM)
#define HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM

#include <hpx/hpx_fwd.hpp>

#include <map>
#include <cstdlib>
#include <string>
#include <vector>

#include <boost/asio/ip/host_name.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////
    // Try to retrieve PBS related settings from the environment
    struct HPX_EXPORT pbs_environment
    {
        typedef std::map<std::string, std::size_t> node_map_type;

        // the constructor tries to read from a PBS node-file, filling our
        // map of nodes and thread counts
        pbs_environment(bool debug = false) 
          : agas_node_num_(0), debug_(debug) 
        {}

        // this function tries to read from a PBS node-file, filling our
        // map of nodes and thread counts
        std::string init_from_file(std::string nodefile, 
            std::string const& agas_host);

        // this function initializes the map of nodes from the given (space 
        // separated) list of nodes
        std::string init_from_nodelist(std::vector<std::string> const& nodes, 
            std::string const& agas_host);

        // The number of threads is either one (if no PBS information was 
        // found), or it is the same as the number of times this node has 
        // been listed in the node file.
        std::size_t retrieve_number_of_threads() const;

        // The number of localities is either one (if no PBS information 
        // was found), or it is the same as the number of distinct node 
        // names listed in the node file.
        std::size_t retrieve_number_of_localities() const;

        // Try to retrieve the node number from the PBS environment
        std::size_t retrieve_node_number() const;

        // This helper function returns the host name of this node.
        static std::string strip_local(std::string const& name);

        static std::string host_name() 
        {
            return strip_local(boost::asio::ip::host_name());
        }

        std::string host_name(std::string const& def_hpx_name) const;

        // We either select the first host listed in the node file or a given 
        // host name to host the AGAS server.
        std::string agas_host_name(std::string const& def_agas) const;
        
        // The AGAS node number represents the number of the node which has 
        // been selected as the AGAS host.
        std::size_t agas_node() const { return agas_node_num_; }

        std::string agas_node_;
        std::size_t agas_node_num_;
        std::map<std::string, std::size_t> nodes_;
        bool debug_;
    };
}}

#endif
