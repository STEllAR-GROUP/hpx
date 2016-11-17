// Copyright (C) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSEMutex_POLICIES_VERBS_RDMA_LOCKS_HPP
#define HPX_PARCELSEMutex_POLICIES_VERBS_RDMA_LOCKS_HPP

// Includes
//
#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <mutex>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
#if HPX_PARCELPORT_VERBS_DEBUG_LOCKS
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

