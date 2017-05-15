//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

namespace detail {

    std::string get_affinity_domain(util::command_line_handling const& cfg)
    {
        std::string affinity_domain("pu");
#if defined(HPX_HAVE_HWLOC)
        if (cfg.affinity_domain_ != "pu")
        {
            affinity_domain = cfg.affinity_domain_;
            if (0 != std::string("pu").find(affinity_domain) &&
                0 != std::string("core").find(affinity_domain) &&
                0 != std::string("numa").find(affinity_domain) &&
                0 != std::string("machine").find(affinity_domain))
            {
                throw detail::command_line_error("Invalid command line option "
                                                         "--hpx:affinity, value must be one of: pu, core, numa, "
                                                         "or machine.");
            }
        }
#endif
        return affinity_domain;
    }

    std::size_t get_affinity_description(
            util::command_line_handling const& cfg, std::string& affinity_desc)
    {
#if defined(HPX_HAVE_HWLOC)
        if (cfg.affinity_bind_.empty())
            return cfg.numa_sensitive_;

        if (cfg.pu_offset_ != std::size_t(-1) || cfg.pu_step_ != 1 ||
            cfg.affinity_domain_ != "pu")
        {
            throw detail::command_line_error(
                    "Command line option --hpx:bind "
                            "should not be used with --hpx:pu-step, --hpx:pu-offset, "
                            "or --hpx:affinity.");
        }

        affinity_desc = cfg.affinity_bind_;
#endif
        return cfg.numa_sensitive_;
    }

} // namespace detail


namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    // Return the number of the NUMA node the current thread is running on
    std::size_t get_numa_node_number()
    {
        std::size_t thread_num = hpx::get_worker_thread_num();
        return get_topology().get_numa_node_number(
            get_thread_manager().get_pu_num(thread_num));
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t get_thread_count(thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state);
    }

    std::int64_t get_thread_count(thread_priority priority,
        thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state, priority);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool enumerate_threads(util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state)
    {
        return get_thread_manager().enumerate_threads(f, state);
    }
}}
