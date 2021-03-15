//  Copyright (c) 2015-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

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

#include <plugins/parcelport/libfabric/rma_memory_region_traits.hpp>

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

        template <typename... Args>
        static int register_memory(Args &&... args) {
            return fi_mr_reg(std::forward<Args>(args)...);
        }

        static int unregister_memory(provider_region *region) {
            return fi_close(&region->fid);
        }

        static int flags() { return
            FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
        }
    };

}}}}

