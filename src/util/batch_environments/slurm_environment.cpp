//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/batch_environments/slurm_environment.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>

#include <hpx/util/safe_lexical_cast.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3

#include <boost/spirit/include/qi.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/fusion/include/pair.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <iostream>
#include <string>

namespace hpx { namespace util { namespace batch_environments
{
    slurm_environment::slurm_environment(std::vector<std::string> & nodelist,
            bool debug)
      : node_num_(0)
      , num_threads_(0)
      , num_tasks_(0)
      , num_localities_(0)
      , valid_(false)
    {
        char *node_num = std::getenv("SLURM_PROCID");
        valid_ = node_num != 0;
        if(valid_)
        {
            // Initialize our node number
            node_num_ = safe_lexical_cast<std::size_t>(node_num);

            // Retrieve number of localities
            retrieve_number_of_localities(debug);

            // Get the list of nodes
            if(nodelist.empty())
            {
                retrieve_nodelist(nodelist, debug);
            }

            // Determine how many threads to use
            retrieve_number_of_threads();
        }
    }

    void slurm_environment::retrieve_number_of_localities(bool debug)
    {
        char* tasks_per_node = std::getenv("SLURM_TASKS_PER_NODE");
        char *total_num_tasks = std::getenv("SLURM_NTASKS");
        char *num_nodes = std::getenv("SLURM_NNODES");

        if(total_num_tasks)
        {
            num_localities_ = safe_lexical_cast<std::size_t>(total_num_tasks);
        }
        else
        {
            num_localities_ = 1;
        }

        std::size_t task_count = 0;
        if (tasks_per_node)
        {
            if (debug) {
                std::cerr << "SLURM tasks per node found (SLURM_TASKS_PER_NODE): "
                    << tasks_per_node << std::endl;
            }

            std::vector<std::string> node_tokens;
            boost::split(node_tokens, tasks_per_node, boost::is_any_of(","));

            if (node_tokens.size() == 1 &&
                node_tokens[0].find_first_of('x') == std::string::npos)
            {
                num_tasks_ = safe_lexical_cast<std::size_t>(node_tokens[0]);
            }
            else
            {
                for(std::string & node_token: node_tokens)
                {
                    std::size_t paren_pos = node_token.find_first_of('(');
                    HPX_ASSERT(paren_pos != std::string::npos);
                    std::size_t num_tasks
                        = safe_lexical_cast<std::size_t>(
                                node_token.substr(0, paren_pos));
                    std::size_t repetition = 1;
                    if(paren_pos != std::string::npos)
                    {
                        HPX_ASSERT(node_token[paren_pos + 1] == 'x');
                        HPX_ASSERT(node_token[node_token.size() - 1] == ')');
                        std::size_t begin = paren_pos + 2;
                        std::size_t end = node_token.size() - 1;
                        repetition =
                            safe_lexical_cast<std::size_t>(
                                node_token.substr(paren_pos + 2, end - begin));
                    }

                    std::size_t next_task_count
                        = task_count + num_tasks * repetition;
                    if(node_num_ >= task_count && node_num_ < next_task_count)
                    {
                        num_tasks_ = num_tasks;
                    }

                    task_count = next_task_count;

                    if(task_count > node_num_)
                    {
                        if (debug) {
                            std::cerr
                                << "SLURM node number outside of available "
                                    "list of tasks"
                                << std::endl;
                        }
                        break;
                    }
                }
            }
        }
        else
        {
            num_tasks_ = 1;
            task_count = 1;
        }

        if (task_count != num_localities_ && num_nodes)
        {
            num_tasks_
                = num_localities_
                / safe_lexical_cast<std::size_t>(num_nodes);
        }
    }

    struct construct_nodelist
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

        result_type operator()(std::vector<std::string> & nodes,
            param_type const & p) const
        {
            typedef param_type::value_type value_type;

            std::vector<std::string> tmp_nodes;

            for (value_type const& value : p)
            {
                std::string const & prefix = boost::fusion::at_c<0>(value);
                optional_type const & ranges = boost::fusion::at_c<1>(value);
                bool push_now = tmp_nodes.empty();
                if(ranges)
                {
                    for (vector_type const& range : *ranges)
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
                                for (std::string& node : tmp_nodes)
                                {
                                    node += s;
                                }
                            }
                        }
                        else
                        {
                            using hpx::util::safe_lexical_cast;
                            std::size_t begin = safe_lexical_cast<std::size_t>
                                (range[0]);
                            std::size_t end = safe_lexical_cast<std::size_t>
                                (range[1]);
                            if(begin > end) std::swap(begin, end);

                            std::vector<std::string> vs;

                            for(std::size_t i = begin; i <= end; ++i)
                            {
                                std::string s(prefix);
                                std::size_t dec = 10;
                                // pad with zeros
                                for(std::size_t j = 0; j < range[0].length()-1; ++j)
                                {
                                    if(i < dec)
                                    {
                                        s += "0";
                                    }
                                    dec *= 10;
                                }
                                s += std::to_string(i);
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
                                for (std::string s : tmp)
                                {
                                    for (std::string const& s2 : vs)
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
                        for (std::string& node : tmp_nodes)
                        {
                            node += prefix;
                        }
                    }
                }
            }
            nodes.insert(nodes.end(), tmp_nodes.begin(), tmp_nodes.end());
        }
    };

    void slurm_environment::retrieve_nodelist(std::vector<std::string> & nodes,
        bool debug)
    {
        char* slurm_nodelist_env = std::getenv("SLURM_NODELIST");
        if (slurm_nodelist_env)
        {
            if (debug) {
                std::cerr << "SLURM nodelist found (SLURM_NODELIST): "
                    << slurm_nodelist_env << std::endl;
            }

            std::string nodelist_str(slurm_nodelist_env);
            std::string::iterator begin = nodelist_str.begin();
            std::string::iterator end = nodelist_str.end();

            namespace qi = boost::spirit::qi;
            namespace phoenix = boost::phoenix;

            qi::rule<std::string::iterator, std::string()> prefix;
            qi::rule<std::string::iterator, std::string()> range_str;
            qi::rule<std::string::iterator, std::vector<std::string>()> range;
            qi::rule<std::string::iterator,
                boost::fusion::vector<
                    std::string,
                    boost::optional<std::vector<std::vector<std::string> > >
                >()> ranges;
            qi::rule<std::string::iterator> hostlist;
            qi::rule<std::string::iterator> nodelist;

            // grammar definition
            prefix %=
                    +(qi::print - (qi::char_("[") | qi::char_(",")))
                ;

            range_str %=
                    +(qi::print - (qi::char_("]") | qi::char_( ",") | qi::char_("-")))
                ;

            range %= range_str >> *('-' >> range_str);

            ranges %= prefix >> -(qi::lit("[") >> (range % ',') >> qi::lit("]"));

            hostlist =
                (+ranges)[
                    phoenix::bind(construct_nodelist(),
                        phoenix::ref(nodes), qi::_1)
                ];

            nodelist = hostlist % ',';

            if (!qi::parse(begin, end, nodelist) || begin != end)
            {
                if (debug) {
                    std::cerr << "failed to parse SLURM nodelist (SLURM_NODELIST): "
                        << slurm_nodelist_env << std::endl;
                }
            }
        }
    }

    void slurm_environment::retrieve_number_of_threads()
    {
        char *slurm_cpus_on_node = std::getenv("SLURM_CPUS_ON_NODE");
        if(slurm_cpus_on_node)
        {
            std::size_t slurm_num_cpus =
                safe_lexical_cast<std::size_t>(slurm_cpus_on_node);
            threads::topology const & top = threads::create_topology();
            std::size_t num_pus = top.get_number_of_pus();

            // Figure out if we got the number of logical cores (including
            // hyper-threading) or just the number of physical cores.
            char *slurm_cpus_per_task = std::getenv("SLURM_CPUS_PER_TASK");
            if(slurm_num_cpus == num_pus)
            {
                if(slurm_cpus_per_task)
                    num_threads_
                        = safe_lexical_cast<std::size_t>(slurm_cpus_per_task);
                else
                    num_threads_ = num_pus / num_tasks_;
            }
            else
            {
                std::size_t num_cores = 0;
                if(slurm_cpus_per_task)
                {
                    num_cores
                        = safe_lexical_cast<std::size_t>(slurm_cpus_per_task);
                }
                else
                {
                    num_cores = slurm_num_cpus / num_tasks_;
                }
                HPX_ASSERT(num_cores <= top.get_number_of_cores());

                for(std::size_t core = 0; core != num_cores; ++core)
                {
                    num_threads_ += top.get_number_of_core_pus(core);
                }
            }
        }
    }
}}}
