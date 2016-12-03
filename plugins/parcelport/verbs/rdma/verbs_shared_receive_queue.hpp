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
             struct ibv_srq_attr     attr;           // SRQ attributes
     };

     struct ibv_srq_attr {
             uint32_t                max_wr;
             // Requested max number of outstanding work requests (WRs) in the SRQ
             uint32_t                max_sge;
             // Requested max number of scatter elements per WR
             uint32_t                srq_limit;
             // The limit value of the SRQ (irrelevant for ibv_create_srq)
     };
         */

        // Default constructor.
        verbs_shared_receive_queue(struct rdma_cm_id *cmId,
            verbs_protection_domain_ptr domain)
        {
            _domain = domain;
            _cmId   = cmId;
            memset(&_srq_attr, 0, sizeof(ibv_srq_init_attr));
            _srq_attr.attr.max_wr = VERBS_EP_RX_CNT;
            _srq_attr.attr.max_sge = 6; // @todo : need to query max before setting this

            int err = rdma_create_srq(_cmId, _domain->getDomain(), &_srq_attr);

            if (err != 0) {
                rdma_error e(errno, "rdma_create_srq() failed");
                LOG_ERROR_MSG("error creating shared receive queue : "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }

            LOG_DEBUG_MSG("created SRQ shared receive queue "
                /*<< _cmId->qp->srq*/ << " context " << _srq_attr.srq_context
                << " max wr " << _srq_attr.attr.max_wr << " max sge "
                << _srq_attr.attr.max_sge);
            return;
    }

        ~verbs_shared_receive_queue()
        {
            //  if (rdma_destroy_srq(_cmId)) {
            rdma_destroy_srq(_cmId);

            //    rdma_error e(errno, "rdma_destroy_srq() failed");
            //    LOG_ERROR_MSG("error deleting shared receive queue : "
            // << rdma_error::error_string(e.error_code()));
            //    throw e;
            //  }
        }

        inline struct ibv_srq *getsrq_() {
            if (_cmId->qp == nullptr) {
                LOG_ERROR_MSG("Trying to access SRQ before QP is ready! ");
                return nullptr;
            }
            return _cmId->qp->srq; }

    private:

        //! Memory region for inbound messages.
        verbs_protection_domain_ptr  _domain;
        struct ibv_srq_init_attr _srq_attr;
        struct rdma_cm_id       *_cmId;
    };

    //! Smart pointer for verbs_shared_receive_queue object.
    typedef std::shared_ptr<verbs_shared_receive_queue> verbs_shared_receive_queue_ptr;

}}}}

#endif

