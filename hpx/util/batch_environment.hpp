//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM)
#define HPX_UTIL_PBS_ENVIRONMENT_AUG_26_2011_0901AM

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/asio/ip/tcp.hpp>

#include <map>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////
    // Try to retrieve default values from a batch environment
    struct HPX_EXPORT batch_environment
    {
        // the constructor tries to read initial values from a batch environment,
        // filling our map of nodes and thread counts
        batch_environment(std::vector<std::string> & nodelist,
            util::runtime_configuration const& cfg,
            bool debug = false, bool enable = true);

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

        std::string host_name() const;

        std::string host_name(std::string const& def_hpx_name) const;

        // We either select the first host listed in the node file or a given
        // host name to host the AGAS server.
        std::string agas_host_name(std::string const& def_agas) const;

        // The AGAS node number represents the number of the node which has
        // been selected as the AGAS host.
        std::size_t agas_node() const { return agas_node_num_; }

        // The function will analyze the current environment and return true
        // if it finds sufficient information to deduce its running as a batch
        // job.
        bool found_batch_environment() const;

        // Return a string containing the name of the batch system
        std::string get_batch_name() const;

        typedef std::map<
            boost::asio::ip::tcp::endpoint, std::pair<std::string, std::size_t>
        > node_map_type;

        std::string agas_node_;
        std::size_t agas_node_num_;
        std::size_t node_num_;
        std::size_t num_threads_;
        node_map_type nodes_;
        std::size_t num_localities_;
        std::string batch_name_;
        bool debug_;
    };
}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif
