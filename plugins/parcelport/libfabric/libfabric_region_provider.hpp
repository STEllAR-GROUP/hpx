//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_MEMORY_REGION_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_MEMORY_REGION_HPP

#include <hpx/traits/rma_memory_region_traits.hpp>
//
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
//
#include <memory>
#include <utility>
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct libfabric_region_provider
    {
        // The internal memory region handle
        typedef struct fid_mr     provider_region;
        typedef struct fid_domain provider_domain;

        // register region
        template <typename... Args>
        static int register_memory(Args &&... args) {
            return fi_mr_reg(std::forward<Args>(args)...);
        }

        // unregister region
        static int unregister_memory(provider_region *region) {
            return fi_close(&region->fid);
        }

        // Default registration flags for this provider
        static int flags() { return
            FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
        }

        // Get the local descriptor of the memory region.
        static void *get_local_key(provider_region *region) {
            return fi_mr_desc(region);
        }

        // Get the remote key of the memory region.
        static uint64_t get_remote_key(provider_region *region) {
            return fi_mr_key(region);
        }
    };

}}}}

#endif
