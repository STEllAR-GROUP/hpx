//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#include <hpx/config.hpp>
#include <hpx/util/batch_environments/slurm_environment.hpp>

#include <hpx/util/assert.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3

#include <boost/spirit/include/qi.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/fusion/include/pair.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
        valid_ = node_num != nullptr;
        if(valid_)
        {
            // Initialize our node number
            node_num_ = safe_lexical_cast<std::size_t>(node_num);

            // Retrieve number of localities
            retrieve_number_of_localities(debug);

            // Retrieve number of tasks
            retrieve_number_of_tasks(debug);

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
        char *total_num_tasks = std::getenv("SLURM_STEP_NUM_TASKS");
        if(total_num_tasks)
        {
            num_localities_ = safe_lexical_cast<std::size_t>(total_num_tasks);
        }
        else
        {
            if (debug) {
                std::cerr << "SLURM_STEP_NUM_TASKS not found: set num_localities to 1"
                          << std::endl;
            }
            num_localities_ = 1;
        }
    }

    void slurm_environment::retrieve_number_of_tasks(bool debug)
    {
        char *slurm_step_tasks_per_node = std::getenv("SLURM_STEP_TASKS_PER_NODE");
        if(slurm_step_tasks_per_node)
        {
            std::vector<std::string> tokens;
            boost::split(tokens, slurm_step_tasks_per_node, boost::is_any_of(","));

            char *slurm_node_id = std::getenv("SLURM_NODEID");
            HPX_ASSERT(slurm_node_id != nullptr);
            if(slurm_node_id)
            {
                std::size_t node_id = safe_lexical_cast<std::size_t>(slurm_node_id);
                std::size_t task_count = 0;
                for( auto& token : tokens )
                {
                    std::size_t paren_pos = token.find_first_of('(');
                    if(paren_pos != std::string::npos)
                    {
                        HPX_ASSERT(token[paren_pos + 1] == 'x');
                        HPX_ASSERT(token[token.size() - 1] == ')');
                        std::size_t begin = paren_pos + 2;
                        std::size_t end = token.size() - 1;
                        task_count +=
                              safe_lexical_cast<std::size_t>(
                                 token.substr(paren_pos + 2, end - begin));
                    }
                    else
                    {
                        task_count += 1;
                    }

                    if( task_count > node_id )
                    {
                       num_tasks_ =
                             safe_lexical_cast<std::size_t>(token.substr(0, paren_pos));
                       break;
                    }
                }
                HPX_ASSERT( num_tasks_ );
            }
        }
        else
        {
            if (debug) {
                std::cerr << "SLURM_STEP_TASKS_PER_NODE not found: set num_tasks to 1"
                          << std::endl;
            }
            num_tasks_ = 1;
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
        char* slurm_nodelist_env = std::getenv("SLURM_STEP_NODELIST");
        if (slurm_nodelist_env)
        {
            if (debug) {
                std::cerr << "SLURM nodelist found (SLURM_STEP_NODELIST): "
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
                    std::cerr << "failed to parse SLURM nodelist (SLURM_STEP_NODELIST): "
                        << slurm_nodelist_env << std::endl;
                }
            }
        }
    }

    void slurm_environment::retrieve_number_of_threads()
    {
        char *slurm_cpus_per_task = std::getenv("SLURM_CPUS_PER_TASK");
        if(slurm_cpus_per_task)
            num_threads_ = safe_lexical_cast<std::size_t>(slurm_cpus_per_task);
        else
        {
            char *slurm_job_cpus_on_node = std::getenv("SLURM_JOB_CPUS_PER_NODE");
            HPX_ASSERT(slurm_job_cpus_on_node != nullptr);
            if(slurm_job_cpus_on_node)
            {
                std::vector<std::string> tokens;
                boost::split(tokens, slurm_job_cpus_on_node, boost::is_any_of(","));

                char *slurm_node_id = std::getenv("SLURM_NODEID");
                HPX_ASSERT(slurm_node_id != nullptr);
                if(slurm_node_id)
                {
                    std::size_t node_id = safe_lexical_cast<std::size_t>(slurm_node_id);
                    std::size_t task_count = 0;
                    for( auto& token : tokens )
                    {
                        std::size_t paren_pos = token.find_first_of('(');
                        if(paren_pos != std::string::npos)
                        {
                            HPX_ASSERT(token[paren_pos + 1] == 'x');
                            HPX_ASSERT(token[token.size() - 1] == ')');
                            std::size_t begin = paren_pos + 2;
                            std::size_t end = token.size() - 1;
                            task_count +=
                                  safe_lexical_cast<std::size_t>(
                                     token.substr(paren_pos + 2, end - begin));
                        }
                        else
                        {
                            task_count += 1;
                        }
                        if( task_count > node_id )
                        {
                           num_threads_ =
                                 safe_lexical_cast<std::size_t>(
                                    token.substr(0, paren_pos)) / num_tasks_;
                           break;
                        }
                    }
                    HPX_ASSERT( num_threads_ );
                }
            }
        }
    }
}}}
