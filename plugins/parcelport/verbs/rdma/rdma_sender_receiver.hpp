// Copyright (c) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_VERBS_CONNECTION_BASE
#define HPX_PARCELSET_POLICIES_VERBS_CONNECTION_BASE

// Includes
#include <plugins/parcelport/verbs/rdma/rdma_memory_pool.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_shared_receive_queue.hpp>
//
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
//
#include <iomanip>
#include <atomic>

//! Base connection class for RDMA operations with a remote partner.
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
    enum connection_state {
        connecting,
        accepting,
        rejecting,
        connected,
        disconnecting,
        being_disconnected
    };
//    std::atomic<connection_state> state_;

    // ---------------------------------------------------------------------------
    // Provides the most low level access to send/receive read/write operations
    // using the verbs rdma API.
    // For debugging purposes we provide a count of current outstanging receives
    // posted. Allowing the preposted receives to drop to zero can be disastrous
    // if infinit retries are not enabled.
    struct rdma_sender_receiver {

        rdma_sender_receiver()
            : cmId_(nullptr),
              waiting_receives_(0) {}

        rdma_sender_receiver(struct rdma_cm_id *cmId)
            : cmId_(cmId),
              waiting_receives_(0) {}

        ~rdma_sender_receiver() {
            if (waiting_receives_>0) {
                LOG_ERROR_MSG("Closing connection with receives still pending "
                    ": implement FLUSH");
            }
        }

        // ---------------------------------------------------------------------------
        inline void set_cm_id(struct rdma_cm_id *cmId) {
            cmId_ = cmId;
        }

        // ---------------------------------------------------------------------------
        inline void pop_receive_count() const {
            LOG_EXCLUSIVE(uint64_t temp =) --waiting_receives_;
            LOG_DEBUG_MSG("After decrement size of waiting receives is "
                << decnumber(temp));
        }

        // ---------------------------------------------------------------------------
        inline void push_receive_count() const {
            LOG_EXCLUSIVE(uint64_t temp =) ++waiting_receives_;
            LOG_DEBUG_MSG("After increment size of waiting receives is "
                << decnumber(temp));
        }

        // ---------------------------------------------------------------------------
        //! The number of outstanding work requests
        inline uint32_t get_receive_count() { return waiting_receives_; }

        // ---------------------------------------------------------------------------
        // The basic send of a single request operation
        inline uint64_t post_request(struct ibv_send_wr *request)
        {
            // Post the send request.
            struct ibv_send_wr *badRequest;
            LOG_TRACE_MSG(tag_ << "posting "
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
                    LOG_ERROR_MSG(tag_ << "error posting to send queue: "
                        << rdma_error::error_string(err));
                    rdma_error e(err, "posting to send queue failed");
                    throw e;
                }
            }

            return request->wr_id;
        }

        // ---------------------------------------------------------------------------
        // Post a send operation using the specified memory region.
        uint64_t post_send(rdma_memory_region *region, bool signaled,
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

            ++_totalSendPosted;

            LOG_TRACE_MSG(tag_ << "Posted Send wr_id " << hexpointer(send_wr.wr_id)
                << "with Length " << decnumber(send_sge.length) << hexpointer(send_sge.addr)
                << "total send posted " << decnumber(_totalSendPosted));
            // Post a send for outbound message.
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post N regions in one send using scatter gather elements
        inline uint64_t post_send_xN(rdma_memory_region *region[], int N, bool signaled,
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

            ++_totalSendPosted;

            LOG_TRACE_MSG(tag_ << "Posted Send wr_id " << hexpointer(send_wr.wr_id) << hexpointer((uint64_t)region[1])
                << "num SGE " << decnumber(send_wr.num_sge)
                << "with Length " << decnumber(total_length) << hexpointer(send_sge[0].addr)
                << "total send posted " << decnumber(_totalSendPosted));
            // Post a send for outbound message.
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post a zero byte message, this is used for a special ack signal
        inline uint64_t post_send_x0(rdma_memory_region *region, bool signaled,
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

            ++_totalSendPosted;

            LOG_TRACE_MSG(tag_ << "Posted Zero byte Send wr_id " << hexpointer(send_wr.wr_id)
                << "with Length " << decnumber(send_sge.length)
                << "address " << hexpointer(send_sge.addr)
                << "total send posted " << decnumber(_totalSendPosted));
            //   ++_waitingSendPosted;
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        // Post an RDMA read operation from a remote memory region to the
        // specified local memory region.
        inline uint64_t post_read(rdma_memory_region *localregion,
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
            send_wr.send_flags          = IBV_SEND_SIGNALED; // Force completion queue to be posted with result.
            send_wr.wr_id               = (uint64_t)localregion;
            send_wr.wr.rdma.remote_addr = (uint64_t)remoteAddr;
            send_wr.wr.rdma.rkey        = remoteKey;

            ++_totalReadPosted;

            // Post a send to read data.
            LOG_TRACE_MSG(tag_ << "Posted Read wr_id " << hexpointer(send_wr.wr_id)
                << " with Length " << decnumber(read_sge.length) << " " << hexpointer(read_sge.addr)
                << " remote key " << decnumber(send_wr.wr.rdma.rkey) << " remote addr " << hexpointer(send_wr.wr.rdma.remote_addr));
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        //! \brief  Post a rdma write operation to a remote memory region from the specified memory region.
        inline uint64_t post_write(rdma_memory_region *localregion,
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
            send_wr.send_flags          = IBV_SEND_SIGNALED; // Force completion queue to be posted with result.
            send_wr.wr_id               = (uint64_t)localregion;
            send_wr.wr.rdma.remote_addr = (uint64_t)remoteAddr;
            send_wr.wr.rdma.rkey        = remoteKey;

            ++_totalWritePosted;

            // Post a send to write data.
            LOG_ERROR_MSG("Post write has not been implemented and should be checked"
                " for correctness");
            return post_request(&send_wr);
        }

        // ---------------------------------------------------------------------------
        uint64_t  postRecvRegionAsIDChecked(rdma_memory_region *region,
            uint32_t length, bool expected=false)
        {
            uint64_t wr_id = postRecvRegionAsID(region, length, expected);
            push_receive_count();
            return wr_id;
        }

    private:
        // ---------------------------------------------------------------------------
        uint64_t postRecvRegionAsID(rdma_memory_region *region, uint32_t length,
            bool expected=false)
        {
            // Build scatter/gather element for inbound message.
            struct ibv_sge recv_sge;
            recv_sge.addr   = (uint64_t)region->get_address();
            recv_sge.length = length;
            recv_sge.lkey   = region->get_local_key();

            // Build receive work request.
            struct ibv_recv_wr recv_wr;
            memset(&recv_wr, 0, sizeof(recv_wr));
            recv_wr.next    = NULL;
            recv_wr.sg_list = &recv_sge;
            recv_wr.num_sge = 1;
            recv_wr.wr_id   = (uint64_t)region;
            struct ibv_recv_wr *badRequest;
            int err = ibv_post_recv(cmId_->qp, &recv_wr, &badRequest);
            if (err!=0) {
                LOG_ERROR_MSG("postRecvRegionAsID failed");
                throw(std::runtime_error(std::string("postRecvRegionAsID failed")
                    + rdma_error::error_string(errno)));
            }

            ++_totalRecvPosted;

            LOG_DEBUG_MSG(tag_.c_str() << "posting Recv wr_id " << hexpointer(recv_wr.wr_id)
                << " with Length " << hexlength(length)
                << " total recv posted " << decnumber(_totalRecvPosted));
            return recv_wr.wr_id;
        }

        // ---------------------------------------------------------------------------
        // @TODO support shared receive queue
        uint64_t
        postRecvRegionAsID_shared(rdma_memory_region *region, uint32_t length,
            struct ibv_srq* srq,
            bool expected=false)
        {
          // Build scatter/gather element for inbound message.
          struct ibv_sge recv_sge;
          recv_sge.addr   = (uint64_t)region->get_address();
          recv_sge.length = length;
          recv_sge.lkey   = region->get_local_key();

          // Build receive work request.
          struct ibv_recv_wr recv_wr;
          memset(&recv_wr, 0, sizeof(recv_wr));
          recv_wr.next    = NULL;
          recv_wr.sg_list = &recv_sge;
          recv_wr.num_sge = 1;
          recv_wr.wr_id   = (uint64_t)region;
          ++_totalRecvPosted;
          struct ibv_recv_wr *badRequest;
          int err = ibv_post_srq_recv(srq, &recv_wr, &badRequest);
          if (err!=0) {
              LOG_ERROR_MSG("postRecvRegionAsID SRQ failed");
              throw(std::runtime_error(std::string("postRecvRegionAsID SRQ failed")
                  + rdma_error::error_string(errno)));
         }
         LOG_DEBUG_MSG(tag_.c_str() << "posting SRQ Recv wr_id "
              << hexpointer(recv_wr.wr_id) << " with Length " << hexlength(length)
              << " " << hexpointer(region->get_address()));
         return recv_wr.wr_id;
        }

    public:
        // ---------------------------------------------------------------------------
        //! \brief  Return a string naming a ibv_wr_opcode value.
        //! \param  opcode ibv_wr_opcode value.
        //! \return String representing value.

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
              case IBV_WR_ATOMIC_FETCH_AND_ADD: str = "IBV_WR_ATOMIC_FETCH_AND_ADD"; break;
           }
           return str;
        }

        // debugging info when logging
//        LOG_EXCLUSIVE(
        std::string& get_tag(void) { return tag_; }
        void setTag(std::string tag) { tag_ = tag; }
//        )

        int get_total_posted_recv_count()  { return _totalRecvPosted; }
        int get_total_posted_send_count()  { return _totalSendPosted; }
        int get_total_posted_read_count()  { return _totalReadPosted; }
        int get_total_posted_write_count() { return _totalWritePosted; }

    protected:
        // RDMA connection management id.
        struct rdma_cm_id *cmId_;

        //! Tag to identify this connection in trace points.
        std::string tag_;

        mutable std::atomic<uint64_t> waiting_receives_;

        //! Total number of receive operations posted to queue pair.
        std::atomic<uint64_t> _totalRecvPosted;

        //! Total number of send operations posted to queue pair.
        std::atomic<uint64_t> _totalSendPosted;

        //! Total number of rdma read operations posted to queue pair.
        std::atomic<uint64_t> _totalReadPosted;

        //! Total number of rdma write operations posted to queue pair.
        std::atomic<uint64_t> _totalWritePosted;


    };

}}}}

#endif
