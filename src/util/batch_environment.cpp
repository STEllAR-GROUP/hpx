//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/config/asio.hpp>

#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/batch_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <hpx/util/batch_environments/alps_environment.hpp>
#include <hpx/util/batch_environments/slurm_environment.hpp>
#include <hpx/util/batch_environments/pbs_environment.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/host_name.hpp>

namespace hpx { namespace util
{
    batch_environment::batch_environment(std::vector<std::string> & nodelist,
            util::runtime_configuration const& cfg, bool debug, bool enable)
      : agas_node_num_(0)
      , node_num_(-1)
      , num_threads_(-1)
      , num_localities_(-1)
      , debug_(debug)
    {
        if (!enable)
            return;

        batch_environments::alps_environment alps_env(nodelist, debug);
        if(alps_env.valid())
        {
            batch_name_ = "ALPS";
            num_threads_ = alps_env.num_threads();
            node_num_ = alps_env.node_num();
            return;
        }
        batch_environments::slurm_environment slurm_env(nodelist, debug);
        if(slurm_env.valid())
        {
            batch_name_ = "SLURM";
            num_threads_ = slurm_env.num_threads();
            num_localities_ = slurm_env.num_localities();
            node_num_ = slurm_env.node_num();
            return;
        }
        batch_environments::pbs_environment pbs_env(nodelist, debug, cfg);
        if(pbs_env.valid())
        {
            batch_name_ = "PBS";
            num_threads_ = pbs_env.num_threads();
            num_localities_ = pbs_env.num_localities();
            node_num_ = pbs_env.node_num();
            return;
        }
    }

    // This function returns true if a batch environment was found.
    bool batch_environment::found_batch_environment() const
    {
        return !batch_name_.empty();
    }

    // this function initializes the map of nodes from the given a list of nodes
    std::string batch_environment::init_from_nodelist(
        std::vector<std::string> const& nodes,
        std::string const& agas_host)
    {
        if (debug_)
            std::cerr << "got node list" << std::endl;

        boost::asio::io_service io_service;

        bool found_agas_host = false;
        std::size_t agas_node_num = 0;
        std::string nodes_list;
        for (std::string s : nodes)
        {
            if (!s.empty()) {
                if (debug_)
                    std::cerr << "extracted: '" << s << "'" << std::endl;

                boost::asio::ip::tcp::endpoint ep =
                    util::resolve_hostname(s, 0, io_service);

                if (!found_agas_host &&
                    ((agas_host.empty() && nodes_.empty()) || s == agas_host))
                {
                    agas_node_ = s;
                    found_agas_host = true;
                    agas_node_num_ = agas_node_num;
                }

                if (0 == nodes_.count(ep)) {
                    if (debug_)
                        std::cerr << "incrementing agas_node_num" << std::endl;
                    ++agas_node_num;
                }

                std::pair<std::string, std::size_t>& data = nodes_[ep];
                if (data.first.empty())
                    data.first = s;
                ++data.second;

                nodes_list += s + ' ';
            }
        }

        // if an AGAS host is specified, it needs to be in the list
        // of nodes participating in this run
        if (!agas_host.empty() && !found_agas_host) {
            throw hpx::detail::command_line_error("Requested AGAS host (" + agas_host +
                ") not found in node list");
        }

        if (debug_) {
            if (!agas_node_.empty()) {
                std::cerr << "using AGAS host: '" << agas_node_
                    << "' (node number " << agas_node_num_ << ")" << std::endl;
            }

            std::cerr << "Nodes from nodelist:" << std::endl;
            node_map_type::const_iterator end = nodes_.end();
            for (node_map_type::const_iterator it = nodes_.begin();
                 it != end; ++it)
            {
                std::cerr << (*it).second.first << ": "
                    << (*it).second.second << " (" << (*it).first << ")"
                    << std::endl;
            }
        }
        return nodes_list;
    }

    // The number of threads is either one (if no PBS/SLURM information was
    // found), or it is the same as the number of times this node has
    // been listed in the node file. Additionally this takes into account
    // the number of tasks run on this node.
    std::size_t batch_environment::retrieve_number_of_threads() const
    {
        return num_threads_;
    }

    // The number of localities is either one (if no PBS information
    // was found), or it is the same as the number of distinct node
    // names listed in the node file. In case of SLURM we can extract
    // the number of localities from the job environment.
    std::size_t batch_environment::retrieve_number_of_localities() const
    {
        return num_localities_;
    }

    // Try to retrieve the node number from the PBS/SLURM environment
    std::size_t batch_environment::retrieve_node_number() const
    {
        return node_num_;
    }

    std::string batch_environment::host_name() const
    {
        std::string hostname = boost::asio::ip::host_name();
        if (debug_)
            std::cerr << "asio host_name: " << hostname << std::endl;
        return hostname;
    }

    std::string batch_environment::host_name(std::string const& def_hpx_name) const
    {
        std::string host = nodes_.empty() ? def_hpx_name : host_name();
        if (debug_)
            std::cerr << "host_name: " << host << std::endl;
        return host;
    }

    // We either select the first host listed in the node file or a given
    // host name to host the AGAS server.
    std::string batch_environment::agas_host_name(std::string const& def_agas) const
    {
        std::string host = agas_node_.empty() ? def_agas : agas_node_;
        if (debug_)
            std::cerr << "agas host_name: " << host << std::endl;
        return host;
    }

    // Return a string containing the name of the batch system
    std::string batch_environment::get_batch_name() const
    {
        return batch_name_;
    }
}}

