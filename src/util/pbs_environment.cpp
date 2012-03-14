//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/pbs_environment.hpp>

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/asio/ip/host_name.hpp>

namespace hpx { namespace util
{
    // this function tries to read from a PBS node-file, filling our
    // map of nodes and thread counts
    std::string pbs_environment::init_from_file(std::string nodefile,
        std::string const& agas_host)
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

                bool found_agas_host = false;
                std::size_t agas_node = 0;
                std::string line;
                while (std::getline(ifs, line)) {
                    if (!line.empty()) {
                        if (debug_)
                            std::cerr << "read: '" << line << "'" << std::endl;

                        if (!found_agas_host) {
                            if ((agas_host.empty() && nodes_.empty()) ||
                                line == agas_host)
                            {
                                agas_node_ = line;
                                found_agas_host = true;
                                agas_node_num_ = agas_node;
                            }

                            if (0 == nodes_.count(line)) {
                                if (debug_)
                                    std::cerr << "incrementing agas_node"
                                              << std::endl;
                                ++agas_node;
                            }
                        }

                        ++nodes_[line];
                    }
                }

                // if an AGAS host is specified, it needs to be in the list
                // of nodes participating in this run
                if (!agas_host.empty() && !found_agas_host) {
                    throw std::logic_error("Requested AGAS host (" +
                        agas_host + ") not found in node list");
                }

                if (debug_) {
                    if (!agas_node_.empty()) {
                        std::cerr << "using AGAS host: '" << agas_node_
                                  << "' (node number " << agas_node_num_ << ")"
                                  << std::endl;
                    }
                    std::cerr << "Nodes from file:" << std::endl;
                    std::map<std::string, std::size_t>::const_iterator end = nodes_.end();
                    for (std::map<std::string, std::size_t>::const_iterator it = nodes_.begin();
                         it != end; ++it)
                    {
                        std::cerr << (*it).first << ":" << (*it).second
                            << std::endl;
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
    std::string pbs_environment::init_from_nodelist(
        std::vector<std::string> const& nodes,
        std::string const& agas_host)
    {
        if (debug_)
            std::cerr << "got node list" << std::endl;

        bool found_agas_host = false;
        std::size_t agas_node = 0;
        std::string nodes_list;
        BOOST_FOREACH(std::string s, nodes)
        {
            if (!s.empty()) {
                if (debug_)
                    std::cerr << "extracted: '" << s << "'" << std::endl;

                if (!found_agas_host &&
                    ((agas_host.empty() && nodes_.empty()) || s == agas_host))
                {
                    agas_node_ = s;
                    found_agas_host = true;
                    agas_node_num_ = agas_node;
                }

                if (!!transform_) {    // If the transform is not empty
                    s = transform_(s);
                    if (debug_) {
                        std::cerr << "extracted(transformed): '" << s << "'"
                                  << std::endl;
                    }
                }

                if (0 == nodes_.count(s)) {
                    if (debug_)
                        std::cerr << "incrementing agas_node" << std::endl;
                    ++agas_node;
                }

                ++nodes_[s];
                nodes_list += s + ' ';
            }
        }

        // if an AGAS host is specified, it needs to be in the list
        // of nodes participating in this run
        if (!agas_host.empty() && !found_agas_host) {
            throw std::logic_error("Requested AGAS host (" + agas_host +
                ") not found in node list");
        }

        if (debug_) {
            if (!agas_node_.empty()) {
                std::cerr << "using AGAS host: '" << agas_node_
                    << "' (node number " << agas_node_num_ << ")" << std::endl;
            }

            std::cerr << "Nodes from nodelist:" << std::endl;
            std::map<std::string, std::size_t>::const_iterator end = nodes_.end();
            for (std::map<std::string, std::size_t>::const_iterator it = nodes_.begin();
                  it != end; ++it)
            {
                std::cerr << (*it).first << ":" << (*it).second
                    << std::endl;
            }
        }
        return nodes_list;
    }

    // The number of threads is either one (if no PBS information was
    // found), or it is the same as the number of times this node has
    // been listed in the node file.
    std::size_t pbs_environment::retrieve_number_of_threads() const
    {
        // WARNING: PBS_NUM_PPN appears to be plain unreliable
        /*
        char* pbs_num_ppn = std::getenv("PBS_NUM_PPN");
        if (pbs_num_ppn) {
            try {
                std::string value(pbs_num_ppn);
                std::size_t result = boost::lexical_cast<std::size_t>(value);
                if (debug_) {
                    std::cerr << "retrieve_number_of_threads (PBS_NUM_PPN): " << result
                              << std::endl;
                }
                return result;
            }
            catch (boost::bad_lexical_cast const&) {
                ; // just ignore the error
            }
        }
        */

        // fall back to counting the number of occurrences of this node
        // in the node-file
        node_map_type::const_iterator it = nodes_.find(host_name());
        std::size_t result = it != nodes_.end() ? (*it).second : 1;
        if (debug_)
            std::cerr << "retrieve_number_of_threads: " << result << std::endl;
        return result;
    }

    // The number of localities is either one (if no PBS information
    // was found), or it is the same as the number of distinct node
    // names listed in the node file.
    std::size_t pbs_environment::retrieve_number_of_localities() const
    {
        std::size_t result = nodes_.empty() ? 1 : nodes_.size();
        if (debug_) {
            std::cerr << "retrieve_number_of_localities: " << result
                      << std::endl;
        }
        return result;
    }

    // Try to retrieve the node number from the PBS environment
    std::size_t pbs_environment::retrieve_node_number() const
    {
        char* pbs_nodenum = std::getenv("PBS_NODENUM");
        if (pbs_nodenum) {
            try {
                std::string value(pbs_nodenum);
                std::size_t result = boost::lexical_cast<std::size_t>(value);
                if (debug_) {
                    std::cerr << "retrieve_node_number (PBS_NODENUM): " << result
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

    std::string pbs_environment::host_name() const
    {
        std::string hostname = boost::asio::ip::host_name();
        if (debug_)
            std::cerr << "asio host_name: " << hostname << std::endl;
        if (!!transform_) {   // If the transform is not empty
            hostname = transform_(hostname);
            if (debug_) {
                std::cerr << "asio host_name(transformed): " << hostname
                          << std::endl;
            }
        }
        return hostname;
    }

    std::string pbs_environment::host_name(std::string const& def_hpx_name) const
    {
        std::string host = nodes_.empty() ? def_hpx_name : host_name();
        if (debug_)
            std::cerr << "host_name: " << host << std::endl;
        if (!!transform_) {   // If the transform is not empty
            host = transform_(host);
            if (debug_)
                std::cerr << "host_name(transformed): " << host << std::endl;
        }
        return host;
    }

    // We either select the first host listed in the node file or a given
    // host name to host the AGAS server.
    std::string pbs_environment::agas_host_name(std::string const& def_agas) const
    {
        std::string host = agas_node_.empty() ? def_agas : agas_node_;
        if (debug_)
            std::cerr << "agas host_name: " << host << std::endl;
        if (!!transform_) {   // If the transform is not empty
            host = transform_(host);
            if (debug_) {
                std::cerr << "agas host_name(transformed): " << host
                          << std::endl;
            }
        }
        return host;
    }

    bool pbs_environment::run_with_pbs() const
    {
        return std::getenv("PBS_NODENUM") != 0 ? true : false;
    }
}}

