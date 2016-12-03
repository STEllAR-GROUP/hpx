// Copyright (c) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_VERBS_SENDER_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_VERBS_SENDER_RECEIVER_HPP

// Includes
#include <plugins/parcelport/verbs/rdma/rdma_memory_pool.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_shared_receive_queue.hpp>
//
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
//
#include <iomanip>
#include <atomic>
#include <cstddef>
#include <string>

// Base connection class for RDMA operations with a remote partner.
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{

    // ---------------------------------------------------------------------------
    enum endpoint_{
        active,
        passive
    };

    // ---------------------------------------------------------------------------
    enum receive_type {
        shared,
        unique
    };

    // ---------------------------------------------------------------------------
    // Provides the most low level access to send/receive read/write operations
    // using the verbs rdma API.
    // For performance monitoring and debugging, counters of send/recv/read/write
    // operations are maintained.
    //
    // The number of preposted receives is tracked to prevent the receive queue
    // from becoming empty and causing RNR retry errors.
    // NB. Allowing the preposted receives to drop to zero can be disastrous
    // if infinite retries are not enabled.

    struct verbs_sender_receiver
    {
        verbs_sender_receiver(struct rdma_cm_id *cmId) : cmId_(cmId)
        {
            clear_counters();
        }

        ~verbs_sender_receiver() {
            if (preposted_receives_>0) {
                LOG_ERROR_MSG("Closing connection with receives still pending "
                    ": implement FLUSH");
            }
        }

        // ---------------------------------------------------------------------------
        void clear_counters() {
            total_posted_recv_  = 0;
            total_posted_send_  = 0;
            total_posted_read_  = 0;
            total_write_posted_ = 0;
            preposted_receives_ = 0;
        }

        // ---------------------------------------------------------------------------
        inline void set_cm_id(struct rdma_cm_id *cmId) {
            cmId_ = cmId;
        }

        // ---------------------------------------------------------------------------
        struct rdma_cm_id *get_cm_id() { return cmId_; }

        // ---------------------------------------------------------------------------
        inline void pop_receive_count() const {
            LOG_EXCLUSIVE(uint64_t temp =) --preposted_receives_;
            LOG_DEBUG_MSG("After decrement size of waiting receives is "
                << decnumber(temp));
        }

        // ---------------------------------------------------------------------------
        inline void push_receive_count() const {
            LOG_EXCLUSIVE(uint64_t temp =) ++preposted_receives_;
            LOG_DEBUG_MSG("After increment size of waiting receives is "
                << decnumber(temp));
        }

        // ---------------------------------------------------------------------------
        // The number of outstanding work requests
        inline uint32_t get_receive_count() { return preposted_receives_; }

        // ---------------------------------------------------------------------------
        // The basic send of a single request operation
        inline uint64_t post_request(struct ibv_send_wr *request)
        {
            // Post the send request.
            struct ibv_send_wr *badRequest;
            LOG_TRACE_MSG("posting "
                << ibv_wc_opcode_string(request->opcode) << " (" << request->opcode
                << ") work request to send queue with " << request->num_sge
                << " sge, id=" << hexpointer(request->wr_id)
                << ", imm_data=" << hexuint32(request->imm_data));
            int err = ibv_post_send(cmId_->qp, request, &badRequest);
            if (err != 0) {
                if (err==EINVAL)
                {
                    rdma_error e(err, "EINVAL post_request");
                    throw e;
                }
                else {
                    LOG_ERROR_MSG("error posting to send queue: "
                        << rdma_error::error_string(err));
                    rdma_error e(err, "posting to send queue failed");
                    throw e;
                }
            }

            return request->wr_id;
        }

        // ---------------------------------------------------------------------------
        // Post a send operation using the specified memory region.
        uint64_t post_send(verbs_memory_region *region, bool signaled,
            bool withImmediate, uint32_t immediateData)
        {
            // Build scatter/gather element for outbound data.
            struct ibv_sge send_sge;
            send_sge.addr = (uint64_t)region->get_address();
            send_sge.length = region->get_message_length();
            send_sge.lkey = region->get_local_key();

            // Build a send work request.
            struct ibv_send_wr send_wr;
            memset(&send_wr, 0, sizeof(send_wr));
            send_wr.next = nullptr;
            send_wr.sg_list = &send_sge;
            send_wr.num_sge = 1;
            if (withImmediate) {
                send_wr.opcode = IBV_WR_SEND_WITH_IMM;
                send_wr.imm_data = immediateData;
            }
            else {
                send_wr.opcode = IBV_WR_SEND;
            }
            if (signaled) {
                send_wr.send_flags |= IBV_SEND_SIGNALED;
            }
            // use address for wr_id
            send_wr.wr_id = (uint64_t)region;

            ++total_posted_send_;

            LOG_TRACE_MSG("Posted Send wr_id " << hexpointer(send_wr.wr_id)
                << "with Length " << decnumber(send_sge.length)
                << hexpointer(send_sge.addr)
                << "total send posted " << decnumber(total_posted_send_));
            // Post a send for outbound message.
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post N regions in one send using scatter gather elements
        inline uint64_t post_send_xN(verbs_memory_region *region[], int N, bool signaled,
            bool withImmediate, uint32_t immediateData)
        {
            // Build scatter/gather element for outbound data.
            struct ibv_sge send_sge[4]; // caution, don't use more than this
            int total_length = 0;
            for (int i=0; i<N; i++) {
                send_sge[i].addr   = (uint64_t)region[i]->get_address();
                send_sge[i].length = region[i]->get_message_length();
                send_sge[i].lkey   = region[i]->get_local_key();
                total_length      += send_sge[i].length;
            }

            // Build a send work request.
            struct ibv_send_wr send_wr;
            memset(&send_wr, 0, sizeof(send_wr));
            send_wr.next = nullptr;
            send_wr.sg_list = &send_sge[0];
            send_wr.num_sge = N;
            if (withImmediate) {
                send_wr.opcode = IBV_WR_SEND_WITH_IMM;
                send_wr.imm_data = immediateData;
            }
            else {
                send_wr.opcode = IBV_WR_SEND;
            }
            if (signaled) {
                send_wr.send_flags |= IBV_SEND_SIGNALED;
            }
            // use address for wr_id
            send_wr.wr_id = (uint64_t)region[0];

            ++total_posted_send_;

            LOG_TRACE_MSG("Posted Send wr_id " << hexpointer(send_wr.wr_id)
                << hexpointer((uint64_t)region[1])
                << "num SGE " << decnumber(send_wr.num_sge)
                << "with Length " << decnumber(total_length)
                << hexpointer(send_sge[0].addr)
                << "total send posted " << decnumber(total_posted_send_));
            // Post a send for outbound message.
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post a zero byte message, this is used for a special ack signal
        inline uint64_t post_send_x0(verbs_memory_region *region, bool signaled,
            bool withImmediate, uint32_t immediateData)
        {
            // Build scatter/gather element for outbound data.
            struct ibv_sge send_sge;
            send_sge.addr   = 0;
            send_sge.length = 0;
            send_sge.lkey   = 0;

            // Build a send work request.
            struct ibv_send_wr send_wr;
            memset(&send_wr, 0, sizeof(send_wr));
            send_wr.next = nullptr;
            send_wr.sg_list = nullptr;
            send_wr.num_sge = 0;
            if (withImmediate) {
                send_wr.opcode = IBV_WR_SEND_WITH_IMM;
                send_wr.imm_data = immediateData;
            }
            else {
                send_wr.opcode = IBV_WR_SEND;
            }
            if (signaled) {
                send_wr.send_flags |= IBV_SEND_SIGNALED;
            }
            // use address for wr_id
            send_wr.wr_id = (uint64_t)region;

            ++total_posted_send_;

            LOG_TRACE_MSG("Posted Zero byte Send wr_id " << hexpointer(send_wr.wr_id)
                << "with Length " << decnumber(send_sge.length)
                << "address " << hexpointer(send_sge.addr)
                << "total send posted " << decnumber(total_posted_send_));
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post an RDMA read operation from a remote memory region to the
        // specified local memory region.
        inline uint64_t post_read(verbs_memory_region *localregion,
            uint32_t remoteKey, const void *remoteAddr, std::size_t length)
        {
            // Build scatter/gather element for message.
            struct ibv_sge read_sge;
            read_sge.addr   = (uint64_t)localregion->get_address();
            read_sge.length = length;
            read_sge.lkey   = localregion->get_local_key();

            // Build a work request for the read operation
            struct ibv_send_wr send_wr;
            memset(&send_wr, 0, sizeof(send_wr));
            send_wr.next                = nullptr;
            send_wr.sg_list             = &read_sge;
            send_wr.num_sge             = 1;
            send_wr.opcode              = IBV_WR_RDMA_READ;
            // Force completion queue to be posted with result.
            send_wr.send_flags          = IBV_SEND_SIGNALED;
            send_wr.wr_id               = (uint64_t)localregion;
            send_wr.wr.rdma.remote_addr = (uint64_t)remoteAddr;
            send_wr.wr.rdma.rkey        = remoteKey;

            ++total_posted_read_;

            // Post a send to read data.
            LOG_TRACE_MSG("Posted Read wr_id " << hexpointer(send_wr.wr_id)
                << " with Length " << decnumber(read_sge.length) << " "
                << hexpointer(read_sge.addr)
                << " remote key " << decnumber(send_wr.wr.rdma.rkey)
                << " remote addr " << hexpointer(send_wr.wr.rdma.remote_addr));
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post an rdma write operation to a remote region from the local region.
        inline uint64_t post_write(verbs_memory_region *localregion,
            uint32_t remoteKey, const void *remoteAddr, std::size_t length)
        {
            // Build scatter/gather element for inbound message.
            struct ibv_sge write_sge;
            write_sge.addr   = (uint64_t)localregion->get_address();
            write_sge.length = length;
            write_sge.lkey   = localregion->get_local_key();

            // Build a work request for the read operation
            struct ibv_send_wr send_wr;
            memset(&send_wr, 0, sizeof(send_wr));
            send_wr.next                = nullptr;
            send_wr.sg_list             = &write_sge;
            send_wr.num_sge             = 1;
            send_wr.opcode              = IBV_WR_RDMA_WRITE;
            // Force completion queue to be posted with result.
            send_wr.send_flags          = IBV_SEND_SIGNALED;
            send_wr.wr_id               = (uint64_t)localregion;
            send_wr.wr.rdma.remote_addr = (uint64_t)remoteAddr;
            send_wr.wr.rdma.rkey        = remoteKey;

            ++total_write_posted_;

            // Post a send to write data.
            LOG_ERROR_MSG("Post write has not been implemented and should be checked"
                " for correctness");
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        uint64_t  post_recv_region_as_id_counted(verbs_memory_region *region,
            uint32_t length)
        {
            uint64_t wr_id = post_recv_region_as_id(region, length);
            push_receive_count();
            return wr_id;
        }

    private:
        // ---------------------------------------------------------------------------
        uint64_t post_recv_region_as_id(verbs_memory_region *region, uint32_t length)
        {
            // Build scatter/gather element for inbound message.
            struct ibv_sge recv_sge;
            recv_sge.addr   = (uint64_t)region->get_address();
            recv_sge.length = length;
            recv_sge.lkey   = region->get_local_key();

            // Build receive work request.
            struct ibv_recv_wr recv_wr;
            memset(&recv_wr, 0, sizeof(recv_wr));
            recv_wr.next    = nullptr;
            recv_wr.sg_list = &recv_sge;
            recv_wr.num_sge = 1;
            recv_wr.wr_id   = (uint64_t)region;
            struct ibv_recv_wr *badRequest;
            int err = ibv_post_recv(cmId_->qp, &recv_wr, &badRequest);
            if (err!=0) {
                rdma_error e(err, LOG_FORMAT_MSG("post_recv_region_as_id failed on qp "
                    << decnumber(cmId_->qp->qp_num)));
                throw e;
            }

            ++total_posted_recv_;

            LOG_DEBUG_MSG("posting Recv wr_id " << hexpointer(recv_wr.wr_id)
                << " with Length " << hexlength(length)
                << " total recv posted " << decnumber(total_posted_recv_));
            return recv_wr.wr_id;
        }

        // ---------------------------------------------------------------------------
        uint64_t post_recv_region_as_id_shared(verbs_memory_region *region,
            uint32_t length, struct ibv_srq* srq)
        {
          // Build scatter/gather element for inbound message.
          struct ibv_sge recv_sge;
          recv_sge.addr   = (uint64_t)region->get_address();
          recv_sge.length = length;
          recv_sge.lkey   = region->get_local_key();

          // Build receive work request.
          struct ibv_recv_wr recv_wr;
          memset(&recv_wr, 0, sizeof(recv_wr));
          recv_wr.next    = nullptr;
          recv_wr.sg_list = &recv_sge;
          recv_wr.num_sge = 1;
          recv_wr.wr_id   = (uint64_t)region;
          ++total_posted_recv_;
          struct ibv_recv_wr *badRequest;
          int err = ibv_post_srq_recv(srq, &recv_wr, &badRequest);
          if (err!=0) {
              LOG_ERROR_MSG("post_recv_region_as_id SRQ failed");
              throw(std::runtime_error(std::string("post_recv_region_as_id SRQ failed")
                  + rdma_error::error_string(errno)));
         }
         LOG_DEBUG_MSG("posting SRQ Recv wr_id "
              << hexpointer(recv_wr.wr_id) << " with Length " << hexlength(length)
              << " " << hexpointer(region->get_address()));
         return recv_wr.wr_id;
        }

    public:
        // ---------------------------------------------------------------------------
        static std::string ibv_wc_opcode_string(enum ibv_wr_opcode opcode)
        {
           std::string str;
           switch (opcode) {
              case IBV_WR_RDMA_WRITE: str = "IBV_WR_RDMA_WRITE"; break;
              case IBV_WR_RDMA_WRITE_WITH_IMM: str = "IBV_WR_RDMA_WRITE_WITH_IMM"; break;
              case IBV_WR_SEND: str = "IBV_WR_SEND"; break;
              case IBV_WR_SEND_WITH_IMM: str = "IBV_WR_SEND_WITH_IMM"; break;
              case IBV_WR_RDMA_READ: str = "IBV_WR_RDMA_READ"; break;
              case IBV_WR_ATOMIC_CMP_AND_SWP: str = "IBV_WR_ATOMIC_CMP_AND_SWAP"; break;
              case IBV_WR_ATOMIC_FETCH_AND_ADD: str = "IBV_WR_ATOMIC_FETCH_AND_ADD";break;
           }
           return str;
        }

        // ---------------------------------------------------------------------------
        int get_total_posted_recv_count()  { return total_posted_recv_; }
        int get_total_posted_send_count()  { return total_posted_send_; }
        int get_total_posted_read_count()  { return total_posted_read_; }
        int get_total_posted_write_count() { return total_write_posted_; }

    protected:
        // RDMA connection management id.
        struct rdma_cm_id *cmId_;

        // Number of receives that are preposted and (as yet) uncompleted
        mutable std::atomic<uint64_t> preposted_receives_;

        // Total number of receive operations posted to queue pair.
        std::atomic<uint64_t> total_posted_recv_;

        // Total number of send operations posted to queue pair.
        std::atomic<uint64_t> total_posted_send_;

        // Total number of rdma read operations posted to queue pair.
        std::atomic<uint64_t> total_posted_read_;

        // Total number of rdma write operations posted to queue pair.
        std::atomic<uint64_t> total_write_posted_;

    };

}}}}

#endif
