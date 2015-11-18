//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PBS_ENVIRONMENT_HPP)
#define HPX_UTIL_PBS_ENVIRONMENT_HPP

#include <hpx/exception.hpp>

#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/format.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace hpx { namespace util { namespace batch_environments {

    struct pbs_environment
    {
        pbs_environment(std::vector<std::string> & nodelist, bool debug)
          : node_num_(std::size_t(-1))
          , num_threads_(std::size_t(-1))
          , num_localities_(std::size_t(-1))
          , valid_(false)
        {
            char *node_num = std::getenv("PBS_NODENUM");
            valid_ = node_num != 0;
            if(valid_)
            {
                // Initialize our node number
                node_num_ = safe_lexical_cast<std::size_t>(node_num);

                // read the PBS node file. This initializes the number of threads
                // and number of localities
                if(nodelist.empty())
                {
                    read_nodefile(nodelist, debug);
                }
                else
                {
                    num_localities_ = nodelist.size();
                }
            }
        }

        bool valid() const
        {
            return valid_;
        }

        std::size_t node_num() const
        {
            return node_num_;
        }

        std::size_t num_threads() const
        {
            return num_threads_;
        }

        std::size_t num_localities() const
        {
            return num_localities_;
        }

    private:
        std::size_t node_num_;
        std::size_t num_threads_;
        std::size_t num_localities_;
        bool valid_;

        void read_nodefile(std::vector<std::string> & nodelist, bool debug)
        {
            char *node_file = std::getenv("PBS_NODEFILE");
            if(!node_file)
            {
                valid_ = false;
                return;
            }

            std::ifstream ifs(node_file);
            if (ifs.is_open()) {
                std::map<std::string, std::size_t> nodes;
                typedef std::map<std::string, std::size_t>::iterator nodes_iterator;

                bool fill_nodelist = nodelist.empty();

                if (debug)
                    std::cerr << "opened: " << node_file << std::endl;
                std::string line;
                while (std::getline(ifs, line)) {
                    if (!line.empty()) {
                        nodes_iterator it = nodes.find(line);
                        if(it != nodes.end())
                        {
                            ++it->second;
                        }
                        else
                        {
                            it = nodes.insert(std::make_pair(line, 1)).first;
                            if(fill_nodelist)
                            {
                                nodelist.push_back(line);
                            }
                        }
                        num_threads_ = it->second;
                    }
                }
                num_localities_ = nodes.size();
            }
            else {
                if (debug)
                    std::cerr << "failed opening: " << node_file << std::endl;

                // raise hard error if nodefile could not be opened
                throw hpx::detail::command_line_error(boost::str(boost::format(
                    "Could not open nodefile: '%s'") % node_file));
            }
        }
    };
}}}

#endif
