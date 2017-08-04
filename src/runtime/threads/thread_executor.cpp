//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/thread_executor.hpp>

#include <hpx/runtime/threads/executors/default_executor.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/atomic.hpp>

#include <cstddef>

namespace hpx { namespace threads
{
    namespace detail
    {
        mask_cref_type executor_base::get_pu_mask(topology const& topology,
                std::size_t num_thread) const
        {
            auto &rp = hpx::get_resource_partitioner();
            return rp.get_pu_mask(num_thread);
        }
    }

    scheduled_executor& scheduled_executor::default_executor()
    {
        typedef util::reinitializable_static<
            executors::default_executor, tag
        > static_type;

        static_type instance;
        return instance.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::atomic<scheduled_executor> default_executor_instance;

    scheduled_executor default_executor()
    {
        if (!default_executor_instance.load())
        {
            scheduled_executor& default_exec =
                scheduled_executor::default_executor();
            scheduled_executor empty_exec;

            default_executor_instance.compare_exchange_strong(
                empty_exec, default_exec);
        }
        return default_executor_instance.load();
    }

    void set_default_executor(scheduled_executor executor)
    {
        default_executor_instance.store(executor);
    }
}}
