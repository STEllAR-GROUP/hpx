//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/batch_environments/pjm_environment.hpp>
#include <hpx/util/from_string.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

namespace hpx::util::batch_environments {

    // PJM_MPI_PROC: number of overall localities
    // PJM_NODE: number of physical nodes
    // PJM_PROC_BY_NODE: number of localities per node
    // PJM_NODE_CORE: number of cores per node
    // PJM_TOTAL_CORE: total number of cores (PJM_NODE * PJM_NODE_CORE)
    //
    // Fugaku:
    //      FLIB_AFFINITY_ON_PROCESS: list of indices for allocated
    //                                cores
    //
    // if launched by mpiexec:
    //      PMIX_RANK: current rank number
    pjm_environment::pjm_environment(
        std::vector<std::string>&, bool have_mpi, bool)
      : node_num_(static_cast<std::size_t>(-1))
      , num_threads_(static_cast<std::size_t>(-1))
      , num_localities_(0)
      , valid_(false)
    {
        char* num_nodes = std::getenv("PJM_NODE");
        valid_ = num_nodes != nullptr;
        if (valid_)
        {
            // Get the number of localities
            num_localities_ = from_string<std::size_t>(num_nodes);

            if (have_mpi)
            {
                // Initialize our node number, if available
                char* var = std::getenv("PMIX_RANK");
                if (var != nullptr)
                {
                    node_num_ = from_string<std::size_t>(var);
                }
            }
            else if (num_localities_ > 1)
            {
                valid_ = false;
            }

            // Get the number of threads, if available
            char* var = std::getenv("PJM_NODE_CORE");
            if (var != nullptr)
            {
                num_threads_ = from_string<std::size_t>(var);
            }
            else if ((var = std::getenv("PJM_TOTAL_CORE")) != nullptr)
            {
                num_threads_ = from_string<std::size_t>(var) / num_localities_;
            }
            else if ((var = std::getenv("FLIB_AFFINITY_ON_PROCESS")) != nullptr)
            {
                boost::char_separator<char> sep(",");
                boost::tokenizer<boost::char_separator<char>> tok(
                    std::string(var), sep);
                num_threads_ = static_cast<std::size_t>(
                    std::distance(std::begin(tok), std::end(tok)));
            }
        }
    }
}    // namespace hpx::util::batch_environments
