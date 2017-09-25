//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/thread_pool_helpers.hpp

#ifndef HPX_RUNTIME_THREAD_POOL_HELPERS_HPP
#define HPX_RUNTIME_THREAD_POOL_HELPERS_HPP

#include <hpx/config.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace threads { namespace detail
{
    class HPX_EXPORT thread_pool_base;
}}}

namespace hpx { namespace resource
{
    ///////////////////////////////////////////////////////////////////////////
    /// Return the number of thread pools currently managed by the
    /// \a resource_partitioner
    HPX_API_EXPORT std::size_t get_num_thread_pools();

    /// Return the number of threads in all thread pools currently
    /// managed by the \a resource_partitioner
    HPX_API_EXPORT std::size_t get_num_threads();

    /// Return the number of threads in the given thread pool currently
    /// managed by the \a resource_partitioner
    HPX_API_EXPORT std::size_t get_num_threads(std::string const& pool_name);

    /// Return the number of threads in the given thread pool currently
    /// managed by the \a resource_partitioner
    HPX_API_EXPORT std::size_t get_num_threads(std::size_t pool_index);

    /// Return the internal index of the pool given its name.
    HPX_API_EXPORT std::size_t get_pool_index(std::string const& pool_name);

    /// Return the name of the pool given its internal index
    HPX_API_EXPORT std::string const& get_pool_name(std::size_t pool_index);

    /// Return the name of the pool given its name
    HPX_API_EXPORT threads::detail::thread_pool_base& get_thread_pool(
        std::string const& pool_name);

    /// Return the thread pool given its internal index
    HPX_API_EXPORT threads::detail::thread_pool_base& get_thread_pool(
        std::size_t pool_index);
}}

#endif /*HPX_RUNTIME_GET_OS_THREAD_COUNT_HPP*/
