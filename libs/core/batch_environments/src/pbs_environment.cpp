//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/batch_environments/pbs_environment.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace hpx { namespace util { namespace batch_environments {
    pbs_environment::pbs_environment(
        std::vector<std::string>& nodelist, bool have_mpi, bool debug)
      : node_num_(std::size_t(-1))
      , num_localities_(std::size_t(-1))
      , num_threads_(std::size_t(-1))
      , valid_(false)
    {
        char* node_num = std::getenv("PBS_NODENUM");
        valid_ = node_num != nullptr;
        if (valid_)
        {
            // Initialize our node number
            node_num_ = from_string<std::size_t>(node_num, std::size_t(1));

            if (nodelist.empty())
            {
                // read the PBS node file. This initializes the number of
                // localities
                read_nodefile(nodelist, have_mpi, debug);
            }
            else
            {
                // read the PBS node list. This initializes the number of
                // localities
                read_nodelist(nodelist, debug);
            }

            char* thread_num = std::getenv("PBS_NUM_PPN");
            if (thread_num != nullptr)
            {
                // Initialize number of cores to run on
                num_threads_ =
                    from_string<std::size_t>(thread_num, std::size_t(-1));
            }
        }
    }

    void pbs_environment::read_nodefile(
        std::vector<std::string>& nodelist, bool have_mpi, bool debug)
    {
        char* node_file = std::getenv("PBS_NODEFILE");
        if (!node_file)
        {
            valid_ = false;
            return;
        }

        std::ifstream ifs(node_file);
        if (ifs.is_open())
        {
            std::set<std::string> nodes;
            typedef std::set<std::string>::iterator nodes_iterator;

            bool fill_nodelist = nodelist.empty();

            if (debug)
                std::cerr << "opened: " << node_file << std::endl;

            std::string line;
            while (std::getline(ifs, line))
            {
                if (!line.empty())
                {
                    nodes_iterator it = nodes.find(line);
                    if (it == nodes.end())
                    {
                        nodes.insert(line);
                        if (fill_nodelist)
                        {
                            nodelist.push_back(line);
                        }
                    }
                }
            }
            num_localities_ = nodes.size();
        }
        else
        {
            if (debug)
                std::cerr << "failed opening: " << node_file << std::endl;

            // if MPI is active we can ignore the missing node-file
            if (have_mpi)
            {
                return;
            }

            // raise hard error if nodefile could not be opened
            throw hpx::detail::command_line_error(
                hpx::util::format("Could not open nodefile: '{}'", node_file));
        }
    }

    void pbs_environment::read_nodelist(
        std::vector<std::string>& nodelist, bool debug)
    {
        if (nodelist.empty())
        {
            valid_ = false;
            return;
        }

        std::set<std::string> nodes;
        typedef std::set<std::string>::iterator nodes_iterator;

        if (debug)
            std::cerr << "parsing nodelist" << std::endl;

        for (std::string const& s : nodelist)
        {
            if (!s.empty())
            {
                nodes_iterator it = nodes.find(s);
                if (it == nodes.end())
                {
                    nodes.insert(s);
                }
            }
        }
        num_localities_ = nodes.size();
    }
}}}    // namespace hpx::util::batch_environments
