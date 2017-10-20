//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_PROTECTION_DOMAIN
#define HPX_PARCELSET_POLICIES_VERBS_PROTECTION_DOMAIN

#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
//
#include <infiniband/verbs.h>
//
#include <memory>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{

    struct verbs_protection_domain
    {
        // ---------------------------------------------------------------------------
        verbs_protection_domain(struct ibv_context *context)
        {
            // Validate context pointer (since ibv_ functions won't check it).
            if (context == nullptr) {
                LOG_ERROR_MSG("error with context pointer " << context
                    << " when constructing protection domain");
                throw std::runtime_error("Null context in protection domain");
            }

            // Allocate a protection domain.
            pd_ = ibv_alloc_pd(context);
            if (pd_ == nullptr) {
                LOG_ERROR_MSG("error allocating protection domain");
                throw std::runtime_error("error allocating protection domain");
            }
            LOG_DEBUG_MSG("allocated protection domain " << pd_->handle);
        }

        // ---------------------------------------------------------------------------
        ~verbs_protection_domain()
        {
            if (pd_ != nullptr) {
                uint32_t handle = pd_->handle;
                int err = ibv_dealloc_pd(pd_);
                if (err == 0) {
                    pd_ = nullptr;
                    LOG_DEBUG_MSG("deallocated protection domain " << handle);
                }
                else {
                    LOG_ERROR_MSG("error deallocating protection domain " << handle
                        << ": " <<  rdma_error::error_string(errno));
                }
            }
        }

        // ---------------------------------------------------------------------------
        // get the infiniband verbs protection domain object
        struct ibv_pd *getDomain(void) const {
            return pd_;
        }

        // ---------------------------------------------------------------------------
        // get the infiniband verbs protection domain handle
        uint32_t get_handle(void) const {
            return pd_ != nullptr ? pd_->handle : 0;
        }

    private:

        // Protection domain.
        struct ibv_pd *pd_;

    };

    // Smart pointer for verbs_protection_domain object.
    typedef std::shared_ptr<verbs_protection_domain> verbs_protection_domain_ptr;

}}}}

#endif

