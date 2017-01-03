//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_SHARED_RECEIVE_QUEUE
#define HPX_PARCELSET_POLICIES_VERBS_SHARED_RECEIVE_QUEUE

#include <plugins/parcelport/verbs/rdma/verbs_protection_domain.hpp>
//
#include <rdma/rdma_verbs.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
//
#include <iostream>
#include <memory>

#define VERBS_EP_RX_CNT         (4096)  // default SRQ size
#define VERBS_EP_TX_CNT         (4096)  // default send count

// @TODO : This class is not used yet and not finished
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class verbs_shared_receive_queue
    {
    public:

        /*
     struct ibv_srq_init_attr {
             void                   *srq_context;    // Associated context of the SRQ
             struct ibvsrq_attr     attr;           // SRQ attributes
     };

     struct ibvsrq_attr {
             uint32_t                max_wr;
             // Requested max number of outstanding work requests (WRs) in the SRQ
             uint32_t                max_sge;
             // Requested max number of scatter elements per WR
             uint32_t                srq_limit;
             // The limit value of the SRQ (irrelevant for ibv_create_srq)
     };
         */

        // Default constructor.
        verbs_shared_receive_queue(verbs_protection_domain_ptr domain)
        {
            srq_    = nullptr;
            domain_ = domain;
            //
            struct ibv_srq_init_attr srq_attr;
            memset(&srq_attr, 0, sizeof(ibv_srq_init_attr));
            // @todo : need to query max before setting sge
            srq_attr.attr.max_wr  = VERBS_EP_RX_CNT;
            srq_attr.attr.max_sge = 3;
            //
            srq_ = ibv_create_srq(domain_->getDomain(), &srq_attr);
            if (srq_ == 0) {
                rdma_error e(errno, "ibv_create_srq() failed");
                LOG_ERROR_MSG("error creating shared receive queue : "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }

            LOG_DEVEL_MSG("created SRQ shared receive queue "
                /*<< _cmId->qp->srq*/ << " context " << hexpointer(srq_attr.srq_context)
                << " max wr " << srq_attr.attr.max_wr << " max sge "
                << srq_attr.attr.max_sge);
            return;
        }

        ~verbs_shared_receive_queue()
        {
            if (ibv_destroy_srq(srq_)) {
                rdma_error e(errno, "ibv_destroy_srq() failed");
                throw e;
            }
        }

        inline struct ibv_srq *getsrq() {
            return srq_;
        }

    private:
        verbs_protection_domain_ptr  domain_;
        struct ibv_srq *             srq_;
    };

    // Smart pointer for verbs_shared_receive_queue object.
    typedef std::shared_ptr<verbs_shared_receive_queue> verbs_shared_receive_queue_ptr;

}}}}

#endif

