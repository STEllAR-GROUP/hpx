//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_PROTECTION_DOMAIN
#define HPX_PARCELSET_POLICIES_LIBFABRIC_PROTECTION_DOMAIN

#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/libfabric/fabric_error.hpp>
//
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
//
#include <memory>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{

    struct libfabric_domain
    {
        // ---------------------------------------------------------------------------
        libfabric_domain(struct fid_fabric *fabric, struct fi_info *info)
        {
            // Validate context pointer (since ibv_ functions won't check it).
            if (fabric == nullptr) {
                LOG_ERROR_MSG("error with fabric pointer " << hexpointer(fabric)
                    << " when constructing domain");
                throw std::runtime_error("Null fabric in domain");
            }

            // Allocate a  domain.
            int ret = fi_domain(fabric, info, &pd_, NULL);
            if (ret) {
                LOG_ERROR_MSG("error allocating domain");
                throw std::runtime_error(fi_strerror(ret));
            }
            LOG_DEBUG_MSG("allocated domain " << hexpointer(pd_));
        }

        // ---------------------------------------------------------------------------
        ~libfabric_domain()
        {
            if (pd_ != nullptr) {
                int err = 0; // fi_close(pd_);
                if (err == 0) {
                    pd_ = nullptr;
                }
                else {
                    LOG_ERROR_MSG("error deallocating domain " << hexpointer(pd_)
                        << ": " <<  fabric_error::error_string(err));
                }
            }
        }

        // ---------------------------------------------------------------------------
        // get the infiniband verbs  domain object
        struct fid_domain *getDomain(void) const {
            return pd_;
        }

    private:

        // Protection domain.
        struct fid_domain *pd_;
    };

    // Smart pointer for libfabric_domain object.
    typedef std::shared_ptr<libfabric_domain> libfabric_domain_ptr;

}}}}

#endif

