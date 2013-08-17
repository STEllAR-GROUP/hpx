//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
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
#include <boost/format.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3

#include <boost/spirit/include/qi.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/operator.hpp>
#include <boost/phoenix/statement.hpp>
#include <boost/phoenix/stl.hpp>
#include <boost/phoenix/fusion.hpp>
#include <boost/phoenix/scope.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/fusion/include/pair.hpp>

namespace {
    struct construct_nodes
    {
        typedef void result_type;

        typedef std::vector<std::string> vector_type;

        typedef
            boost::optional<
                std::vector<
                    vector_type
                >
            >
            optional_type;

        typedef
            std::vector<
                boost::fusion::vector<
                    std::string
                  , optional_type
                >
            >
            param_type;

        result_type operator()(std::vector<std::string> & nodes, param_type const & p) const
        {
            typedef param_type::value_type value_type;

            std::vector<std::string> tmp_nodes;

            BOOST_FOREACH(value_type const & value, p)
            {
                std::string const & prefix = boost::fusion::at_c<0>(value);
                optional_type const & ranges = boost::fusion::at_c<1>(value);
                bool push_now = tmp_nodes.empty();
                if(ranges)
                {
                    BOOST_FOREACH(vector_type const & range, *ranges)
                    {
                        if(range.size() == 1)
                        {
                            std::string s(prefix);
                            s += range[0];
                            if(push_now)
                            {
                                tmp_nodes.push_back(s);
                            }
                            else
                            {
                                BOOST_FOREACH(std::string & node, tmp_nodes)
                                {
                                    node += s;
                                }
                            }
                        }
                        else
                        {
                            std::size_t begin = boost::lexical_cast<std::size_t>(range[0]);
                            std::size_t end = boost::lexical_cast<std::size_t>(range[1]);
                            if(begin > end) std::swap(begin, end);

                            std::vector<std::string> vs;

                            for(std::size_t i = begin; i <= end; ++i)
                            {
                                std::string s(prefix);
                                if(i < 10 && range[0].length() > 1)
                                    s += "0";
                                if(i < 100 && range[0].length() > 2)
                                    s += "0";
                                s += boost::lexical_cast<std::string>(i);
                                if(push_now)
                                {
                                    tmp_nodes.push_back(s);
                                }
                                else
                                {
                                    vs.push_back(s);
                                }
                            }
                            if(!push_now)
                            {
                                std::vector<std::string> tmp;
                                std::swap(tmp, tmp_nodes);
                                BOOST_FOREACH(std::string s, tmp)
                                {
                                    BOOST_FOREACH(std::string const & s2, vs)
                                    {
                                        s += s2;
                                        tmp_nodes.push_back(s);
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(push_now)
                    {
                        tmp_nodes.push_back(prefix);
                    }
                    else
                    {
                        BOOST_FOREACH(std::string & node, tmp_nodes)
                        {
                            node += prefix;
                        }
                    }
                }
            }
            nodes.insert(nodes.end(), tmp_nodes.begin(), tmp_nodes.end());
        }
    };

    void add_tasks_per_node(std::size_t idx, std::size_t n, std::vector<std::size_t> & nodes)
    {
        for(std::size_t i = 0; i < n; ++i)
        {
            nodes.push_back(idx);
        }
    }
}

namespace hpx { namespace util
{
    // The function will analyze the current environment and return true
    // if it finds sufficient information to deduce its running as a batch job.
    bool batch_environment::found_batch_environment() const
    {
        // All we have to do for now is to identify SLURM, as PBS does not
        // provide sufficient environment variable to deduce all required
        // information.
        // (https://computing.llnl.gov/linux/slurm/srun.html)
        return std::getenv("SLURM_NODELIST") &&
               std::getenv("SLURM_PROCID");
    }

    // this function tries to read from a PBS node-file, filling our
    // map of nodes and thread counts
    std::string batch_environment::init_from_file(std::string const& nodefile,
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
            else {
                if (debug_)
                    std::cerr << "failed opening: " << nodefile << std::endl;

                // raise hard error if nodefile could not be opened
                throw std::logic_error(boost::str(boost::format(
                    "Could not open nodefile: '%s'") % nodefile));
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

        /** FIXME: use that info
        char* tasks_per_node_env = std::getenv("SLURM_TASKS_PER_NODE");

        std::vector<std::size_t> tasks_per_node;
        if (tasks_per_node_env)
        {
            if (debug_) {
                std::cerr << "SLURM tasks per node found: " << tasks_per_node_env
                          << std::endl;
            }

            std::string tasks_per_node_str(tasks_per_node_env);
            std::string::iterator begin = tasks_per_node_str.begin();
            std::string::iterator end = tasks_per_node_str.end();
            int i = 0;

            namespace qi = boost::spirit::qi;
            namespace phoenix = boost::phoenix;

            qi::parse(
                begin
              , end
              , (
                    (qi::int_ >> "(x" >> qi::int_ >> ')')[
                        phoenix::bind(::add_tasks_per_node, qi::_1, qi::_2, phoenix::ref(tasks_per_node))
                    ]
                | qi::int_[
                        phoenix::push_back(
                            phoenix::ref(tasks_per_node)
                        , qi::_1
                        )
                  ]
                ) % ','
            );
        }
        **/

        if (slurm_nodelist_env)
        {
            if (debug_) {
                std::cerr << "SLURM nodelist found: " << slurm_nodelist_env
                          << std::endl;
            }

            std::string nodelist_str(slurm_nodelist_env);
            std::string::iterator begin = nodelist_str.begin();
            std::string::iterator end = nodelist_str.end();

            std::vector<std::string> nodes;

            namespace qi = boost::spirit::qi;
            namespace phoenix = boost::phoenix;

            qi::rule<std::string::iterator, std::string()>
                prefix;
            prefix %= +(qi::print - (qi::char_("[") | qi::char_(",")));

            qi::rule<std::string::iterator, std::string()>
                range_str;
            range_str %= +(qi::print - (qi::char_("]") | qi::char_( ",") | qi::char_("-")));

            qi::rule<std::string::iterator, std::vector<std::string>()>
                range;
            range %= range_str >> *('-' >> range_str);

            qi::rule<std::string::iterator,
                    boost::fusion::vector<
                        std::string,
                        boost::optional<std::vector<std::vector<std::string> > >
                    >()>
                ranges;
            ranges %= prefix >> -(qi::lit("[") >> (range % ',') >> qi::lit("]"));

            qi::rule<std::string::iterator>
                hostlist = (+ranges)[
                                phoenix::bind(::construct_nodes(),
                                    phoenix::ref(nodes), qi::_1)
                           ];

            qi::rule<std::string::iterator>
                nodelist = hostlist % ',';


            qi::parse(
                begin
              , end
              , nodelist
            );

            /*
            std::cout << nodelist_str << "\n";
            std::cout << "---\n";
            for(std::string const & s: nodes) std::cout << s << "\n";
            std::cout << "---\n";
            */

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
        char* slurm_cpus_per_task = std::getenv("SLURM_CPUS_PER_TASK");
        if(slurm_cpus_per_task)
        {
            try {
                std::string value(slurm_cpus_per_task);
                result = boost::lexical_cast<std::size_t>(value);
                if (debug_) {
                    std::cerr << "retrieve_number_of_threads (SLURM_CPUS_PER_TASK): "
                              << result << std::endl;
                }
                return result;
            }
            catch (boost::bad_lexical_cast const&) {
                ; // just ignore the error
            }
        }
        else
        {
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

