//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>

#include <asio/ip/tcp.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////
    // Try to retrieve default values from a batch environment
    struct HPX_CORE_EXPORT batch_environment
    {
        // the constructor tries to read initial values from a batch
        // environment, filling our map of nodes and thread counts
        batch_environment(std::vector<std::string>& nodelist,
            bool have_mpi = false, bool debug = false, bool enable = true);

        // this function initializes the map of nodes from the given (space
        // separated) list of nodes
        std::string init_from_nodelist(std::vector<std::string> const& nodes,
            std::string const& agas_host);

        // The number of threads is either one (if no PBS information was
        // found), or it is the same as the number of times this node has been
        // listed in the node file.
        std::size_t retrieve_number_of_threads() const noexcept;

        // The number of localities is either one (if no PBS information was
        // found), or it is the same as the number of distinct node names listed
        // in the node file.
        std::size_t retrieve_number_of_localities() const noexcept;

        // Try to retrieve the node number from the PBS environment
        std::size_t retrieve_node_number() const noexcept;

        std::string host_name() const;

        std::string host_name(std::string const& def_hpx_name) const;

        // We either select the first host listed in the node file or a given
        // host name to host the AGAS server.
        std::string agas_host_name(std::string const& def_agas) const;

        // The AGAS node number represents the number of the node which has been
        // selected as the AGAS host.
        std::size_t agas_node() const noexcept;

        // The function will analyze the current environment and return true if
        // it finds sufficient information to deduce its running as a batch job.
        bool found_batch_environment() const noexcept;

        // Return a string containing the name of the batch system
        std::string get_batch_name() const;

        using node_map_type = std::map<asio::ip::tcp::endpoint,
            std::pair<std::string, std::size_t>>;

        std::string agas_node_;
        std::size_t agas_node_num_;
        std::size_t node_num_;
        std::size_t num_threads_;
        node_map_type nodes_;
        std::size_t num_localities_;
        std::string batch_name_;
        bool debug_;
    };
}    // namespace hpx::util

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
