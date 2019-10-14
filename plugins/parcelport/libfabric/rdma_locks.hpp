// Copyright (C) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RDMA_LOCKS_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RDMA_LOCKS_HPP

#include <hpx/config/parcelport_defines.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <mutex>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
#ifdef HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS
    template<typename Mutex>
    struct scoped_lock: std::lock_guard<Mutex>
    {
        scoped_lock(Mutex &m) : std::lock_guard<Mutex>(m)
        {
            LOG_DEBUG_MSG("Creating scoped_lock RAII");
        }

        ~scoped_lock()
        {
            LOG_DEBUG_MSG("Destroying scoped_lock RAII");
        }
    };

    template<typename Mutex>
    struct unique_lock: std::unique_lock<Mutex>
    {
        unique_lock(Mutex &m) : std::unique_lock<Mutex>(m)
        {
            LOG_DEBUG_MSG("Creating unique_lock RAII");
        }

        unique_lock(Mutex& m, std::try_to_lock_t t) : std::unique_lock<Mutex>(m, t)
        {
            LOG_DEBUG_MSG("Creating unique_lock try_to_lock_t RAII");
        }

        ~unique_lock()
        {
            LOG_DEBUG_MSG("Destroying unique_lock RAII");
        }
    };
#else
    template<typename Mutex>
    using scoped_lock = std::lock_guard<Mutex>;

    template<typename Mutex>
    using unique_lock = std::unique_lock<Mutex>;
#endif
}}}}

#endif

