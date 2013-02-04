//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/batch_environment.hpp>

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/tokenizer.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>

namespace hpx { namespace util
{
    // The function will analyze the current environment and return true
    // if it finds sufficient information to deduce its running as a batch job.
    bool batch_environment::found_batch_environment() const
    {
        // All we have to do for now is to identify SLURM as PBS does not
        // provide sufficient environment variable to deduce all required
        // information.
        // (https://computing.llnl.gov/linux/slurm/srun.html)
        return std::getenv("SLURM_NODELIST") &&
               std::getenv("SLURM_PROCID");
    }

    // this function tries to read from a PBS node-file, filling our
    // map of nodes and thread counts
    std::string batch_environment::init_from_file(std::string nodefile,
        std::string const& agas_host)
    {
        if (!nodefile.empty()) {
            boost::asio::io_service io_service;
            std::ifstream ifs(nodefile.c_str());
            if (ifs.is_open()) {
                if (debug_)
                    std::cerr << "opened: " << nodefile << std::endl;

                bool found_agas_host = false;
                std::size_t agas_node_num = 0;
                std::string line;
                while (std::getline(ifs, line)) {
                    if (!line.empty()) {
                        if (debug_)
                            std::cerr << "read: '" << line << "'" << std::endl;

                        boost::asio::ip::tcp::endpoint ep =
                            util::resolve_hostname(line, 0, io_service);

                        if (!found_agas_host) {
                            if ((agas_host.empty() && nodes_.empty()) ||
                                line == agas_host)
                            {
                                agas_node_ = line;
                                found_agas_host = true;
                                agas_node_num_ = agas_node_num;
                            }

                            if (0 == nodes_.count(ep)) {
                                if (debug_)
                                    std::cerr << "incrementing agas_node_num"
                                              << std::endl;
                                ++agas_node_num;
                            }
                        }

                        std::pair<std::string, std::size_t>& data = nodes_[ep];
                        if (data.first.empty())
                            data.first = line;
                        ++data.second;
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
                    node_map_type::const_iterator end = nodes_.end();
                    for (node_map_type::const_iterator it = nodes_.begin();
                         it != end; ++it)
                    {
                        std::cerr << (*it).second.first << ": "
                            << (*it).second.second << " (" << (*it).first
                            << ")" << std::endl;
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
        BOOST_FOREACH(std::string s, nodes)
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
            throw std::logic_error("Requested AGAS host (" + agas_host +
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

    // this function initializes the map of nodes from the environment
    std::string batch_environment::init_from_environment(
        std::string const& agas_host)
    {
        char* slurm_nodelist_env = std::getenv("SLURM_NODELIST");
        char* tasks_per_node_env = std::getenv("SLURM_TASKS_PER_NODE");

        std::vector<std::size_t> tasks_per_node;
        if(tasks_per_node_env)
        {
            if (debug_) {
                std::cerr << "SLURM tasks per node found: " << slurm_nodelist_env
                          << std::endl;
            }

            std::string tasks_per_node_str(tasks_per_node_env);
            std::string::iterator begin = tasks_per_node_str.begin();
            std::string::iterator end = tasks_per_node_str.end();
            int i = 0;
            boost::spirit::qi::parse(
                begin
              , end
              , (
                  (   boost::spirit::qi::int_
                    >> boost::spirit::qi::lit('(')
                    >> boost::spirit::qi::lit('x')
                    >> boost::spirit::qi::int_
                    >> boost::spirit::qi::lit(')'))[
                           boost::phoenix::for_(
                               boost::phoenix::ref(i) = 0
                             , boost::phoenix::ref(i) != boost::spirit::qi::_2
                             , ++boost::phoenix::ref(i)
                           )
                           [
                               boost::phoenix::push_back(
                                   boost::phoenix::ref(tasks_per_node)
                                 , boost::spirit::qi::_1
                               )
                           ]
                       ]
                  | boost::spirit::qi::int_[
                      boost::phoenix::push_back(
                          boost::phoenix::ref(tasks_per_node)
                        , boost::spirit::qi::_1
                      )
                    ]
                ) % ','
            );
        }

        if (slurm_nodelist_env) {

            if (debug_) {
                std::cerr << "SLURM nodelist found: " << slurm_nodelist_env
                          << std::endl;
            }

            typedef boost::tokenizer<boost::char_separator<char> > tokenizer_type;

            std::vector<std::string> nodes;
            boost::char_separator<char> sep (" \t:");
            tokenizer_type tok(std::string(slurm_nodelist_env), sep);
            tokenizer_type::iterator end = tok.end();
            std::size_t i = 0;
            for (tokenizer_type::iterator it = tok.begin (); it != end; ++it)
            {
                if(!tasks_per_node.empty())
                {
                    for(std::size_t j = 0; j != tasks_per_node[j]; ++j)
                    {
                        nodes.push_back (*it);
                    }
                }
                else
                {
                    nodes.push_back (*it);
                }
                ++i;
            }

            return init_from_nodelist(nodes, agas_host);
        }
        return std::string();
    }

    // The number of threads is either one (if no PBS information was
    // found), or it is the same as the number of times this node has
    // been listed in the node file.
    std::size_t batch_environment::retrieve_number_of_threads() const
    {
        std::size_t result = 1;
        char* slurm_cpus_on_node = std::getenv("SLURM_CPUS_ON_NODE");
        if (slurm_cpus_on_node) {
            try {
                std::string value(slurm_cpus_on_node);
                result = boost::lexical_cast<std::size_t>(value);
                if (debug_) {
                    std::cerr << "retrieve_number_of_threads (SLURM_CPUS_ON_NODE): "
                              << result << std::endl;
                }
                return result;
            }
            catch (boost::bad_lexical_cast const&) {
                ; // just ignore the error
            }
        }

        if (!nodes_.empty()) {
            // fall back to counting the number of occurrences of this node
            // in the node-file
            boost::asio::io_service io_service;
            boost::asio::ip::tcp::endpoint ep = util::resolve_hostname(
                host_name(), 0, io_service);

            node_map_type::const_iterator it = nodes_.find(ep);
            if (it == nodes_.end()) {
                throw std::logic_error("Cannot retrieve number of OS threads "
                    "for host_name: " + host_name());
            }
            result = (*it).second.second;
        }
        if (debug_)
            std::cerr << "retrieve_number_of_threads: " << result << std::endl;
        return result;
    }

    // The number of localities is either one (if no PBS information
    // was found), or it is the same as the number of distinct node
    // names listed in the node file.
    std::size_t batch_environment::retrieve_number_of_localities() const
    {
        std::size_t result = nodes_.empty() ? 1 : nodes_.size();
        if (debug_) {
            std::cerr << "retrieve_number_of_localities: " << result
                << std::endl;
        }
        return result;
    }

    // Try to retrieve the node number from the PBS environment
    std::size_t batch_environment::retrieve_node_number() const
    {
        char* nodenum_env = std::getenv("PBS_NODENUM");
        if (!nodenum_env)
            nodenum_env = std::getenv("SLURM_PROCID");

        if (nodenum_env) {
            try {
                std::string value(nodenum_env);
                std::size_t result = boost::lexical_cast<std::size_t>(value);
                if (debug_) {
                    std::cerr << "retrieve_node_number (PBS_NODENUM/SLURM_PROCID): "
                              << result << std::endl;
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

    bool batch_environment::run_with_pbs() const
    {
        return std::getenv("PBS_NODENUM") != 0 ? true : false;
    }

    bool batch_environment::run_with_slurm() const
    {
        return std::getenv("SLURM_PROCID") != 0 ? true : false;
    }

    // Return a string containing the name of the batch system
    std::string batch_environment::get_batch_name() const
    {
        if (run_with_pbs())
            return "PBS";
        if (run_with_slurm())
            return "SLURM";
        return "";
    }
}}

