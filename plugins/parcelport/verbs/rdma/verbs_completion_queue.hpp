//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_COMPLETION_QUEUE_HPP
#define HPX_PARCELSET_POLICIES_VERBS_COMPLETION_QUEUE_HPP

#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
//
#include <inttypes.h>
#include <infiniband/verbs.h>
#include <string>
#include <memory>
#include <mutex>

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class verbs_completion_queue
    {
    public:

        // ---------------------------------------------------------------------------
        verbs_completion_queue(ibv_context *context, int queue_size,
            ibv_comp_channel *completionChannel)
        {
            // Initialize private data.
            context_     = context;
            completionQ_ = nullptr;

            // Validate context pointer (since ibv_ functions won't check it).
            if (context == nullptr) {
                rdma_error e(EFAULT, "device context pointer is null");
                LOG_ERROR_MSG(
                    "error with context pointer "
                    << context << " when constructing completion queue");
                throw e;
            }

            LOG_DEVEL_MSG("Creating completion queue with size " << decnumber(queue_size)
                << "and context " << hexpointer(context_));

            completionQ_ = ibv_create_cq(context, queue_size, nullptr,
                completionChannel, 0);
            if (completionQ_ == nullptr) {
                rdma_error e(errno, "ibv_create_cq() failed");
                LOG_ERROR_MSG(
                    decnumber(completionQ_->handle)
                    << "error creating completion queue: "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }

            LOG_DEVEL_MSG("created completion queue "
                << decnumber(completionQ_->handle));

            // Request notification of events on the completion queue.
            try {
                request_events();
            }
            catch (const rdma_error& e) {
                LOG_ERROR_MSG("error requesting first completion queue notification: "
                    << decnumber(completionQ_->handle)
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }
            LOG_TRACE_MSG(
                decnumber(completionQ_->handle)
                << "requested first notification for completion queue");
        }

        // ---------------------------------------------------------------------------
        ~verbs_completion_queue()
        {
            if (completionQ_ != nullptr) {
                LOG_DEVEL_MSG("destroying completion queue "
                    << decnumber(completionQ_->handle));
                int err = ibv_destroy_cq(completionQ_);
                if (err == 0) {
                    completionQ_ = nullptr;
                }
                else {
                    LOG_ERROR_MSG(
                        "error destroying completion queue: "
                        << rdma_error::error_string(err));
                }
            }
        }

        // ---------------------------------------------------------------------------
        struct ibv_cq *getQueue(void) const {
            return completionQ_;
        }

        // ---------------------------------------------------------------------------
        uint32_t get_handle(void) const {
            return completionQ_ != nullptr ? completionQ_->handle : 0;
        }

        // ---------------------------------------------------------------------------
        void request_events(void)
        {
            int err = ibv_req_notify_cq(completionQ_, 0);
            if (err != 0) {
                rdma_error e(err, "ibv_req_notify_cq() failed");
                LOG_ERROR_MSG(
                    decnumber(completionQ_->handle)
                    << "error requesting notification for completion queue: "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }

            LOG_TRACE_MSG(decnumber(completionQ_->handle)
                << "requested notification for completion queue");
            return;
        }

        // ---------------------------------------------------------------------------
        int poll_completion(struct ibv_wc *completion)
        {
            int nc = ibv_poll_cq(completionQ_, 1, completion);
            if (nc < 0) {
                rdma_error e(EINVAL, "ibv_poll_cq() failed");
                LOG_ERROR_MSG(decnumber(completionQ_->handle)
                    << "error polling completion queue: "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }
            if (nc > 0) {
                if (completion->status != IBV_WC_SUCCESS) {
                    // if the completion was from a flushed queue pair
                    // then it was not a fatal error
                    if (completion->status == IBV_WC_WR_FLUSH_ERR) {
                        LOG_DEVEL_MSG("Received a flushed work completion on qp "
                            << decnumber(completion->qp_num));
                        return -1;
                    }
                    LOG_ERROR_MSG(
                           "CQ " << decnumber(completionQ_->handle)
                        << "work completion status '"
                        << ibv_wc_status_str(completion->status)
                        << "' for operation " << wc_opcode_str(completion->opcode)
                        << " (" << completion->opcode << ") "
                        << hexpointer(completion->wr_id)
                        << "qpnum " << decnumber(completion->qp_num)
                        );
                    std::terminate();
                }
                else {
                    LOG_TRACE_MSG(
                           "CQ " << decnumber(completionQ_->handle)
                        << "work completion status '"
                        << ibv_wc_status_str(completion->status)
                        << "' for operation " << wc_opcode_str(completion->opcode)
                        << " (" << completion->opcode << ")");
                }

                LOG_TRACE_MSG(
                       "CQ " << decnumber(completionQ_->handle)
                    << "removing " << hexpointer(completion->wr_id)
                    << verbs_completion_queue::wc_opcode_str(completion->opcode));
            }
            return nc;
        }

        // ---------------------------------------------------------------------------
        static const std::string wc_opcode_str(enum ibv_wc_opcode opcode)
        {
            std::string str;
            switch (opcode) {
            case IBV_WC_SEND:
                str = "IBV_WC_SEND";
                break;
            case IBV_WC_RDMA_WRITE:
                str = "IBV_WC_RDMA_WRITE";
                break;
            case IBV_WC_RDMA_READ:
                str = "IBV_WC_RDMA_READ";
                break;
            case IBV_WC_COMP_SWAP:
                str = "IBV_WC_COMP_SWAP";
                break;
            case IBV_WC_FETCH_ADD:
                str = "IBV_WC_FETCH_ADD";
                break;
            case IBV_WC_BIND_MW:
                str = "IBV_WC_BIND_MW";
                break;
            case IBV_WC_RECV:
                str = "IBV_WC_RECV";
                break;
            case IBV_WC_RECV_RDMA_WITH_IMM:
                str = "IBV_WC_RECV_RDMA_WITH_IMM";
                break;
            default:
                str = "Got an unknown opcode " + boost::lexical_cast
                < std::string > (opcode);
            }

            return str;
        }

        // ---------------------------------------------------------------------------
        // Maximum number of entries in completion queue.
        static const int MaxQueueSize = 256;

        // Infiniband for IB device.
        struct ibv_context *context_;

        // Completion queue.
        struct ibv_cq *completionQ_;
    };

    // Smart pointer for verbs_completion_queue object.
    typedef std::shared_ptr<verbs_completion_queue> verbs_completion_queue_ptr;

}}}}

#endif
