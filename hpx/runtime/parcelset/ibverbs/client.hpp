//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_CLIENT_HPP)
#define HPX_PARCELSET_IBVERBS_CLIENT_HPP

#include <hpx/runtime/parcelset/ibverbs/messages.hpp>
#include <tcmalloc.h>

namespace hpx { namespace parcelset { namespace ibverbs { namespace detail {
    struct client
    {
        client()
          : buffer_(0)
          , buffer_size_(0)
          , buffer_mr_(0)
          , msg_(0)
          , msg_mr_(0)
          , peer_addr_(0)
          , peer_rkey_(0)
          , size_(0)
          , id_(0)
        {
            int ret = 0;
            ret = posix_memalign(reinterpret_cast<void **>(&msg_), EXEC_PAGESIZE, sizeof(message));
            BOOST_ASSERT(ret == 0);
        }

        ~client()
        {
            close();
            if(buffer_)
            {
                free(buffer_);
                buffer_ = 0;
            }
            if(msg_)
            {
                free(msg_);
                msg_ = 0;
            }
        }

        void set_buffer_size(std::size_t buffer_size, boost::system::error_code &ec)
        {
            BOOST_ASSERT(buffer_ == 0);
            int ret = 0;
            buffer_size_ = buffer_size;
            ret = posix_memalign(reinterpret_cast<void **>(&buffer_), EXEC_PAGESIZE, buffer_size_);
            BOOST_ASSERT(ret == 0);
        }


        void close()
        {
            if(buffer_mr_)
            {
                ibv_dereg_mr(buffer_mr_);
                buffer_mr_ = 0;
            }
            if(msg_mr_)
            {
                ibv_dereg_mr(msg_mr_);
                msg_mr_ = 0;
            }
            
            if(id_)
            {
                rdma_disconnect(id_);
                id_ = 0;
            }

            peer_addr_ = 0;
            peer_rkey_ = 0;

            size_ = 0;
        }

        void post_receive(rdma_cm_id * id)
        {
            struct ibv_recv_wr wr, *bad_wr = NULL;
            struct ibv_sge sge;
                
            std::memset(&wr, 0, sizeof(ibv_recv_wr));
            
            wr.wr_id = (uintptr_t)id;
            wr.sg_list = &sge;
            wr.num_sge = 1;

            sge.addr = (uintptr_t)msg_;
            sge.length = sizeof(message);
            sge.lkey = msg_mr_->lkey;

            int ret = ibv_post_recv(id->qp, &wr, &bad_wr);
            if(ret)
            {
            }
        }

        void on_preconnect(rdma_cm_id * id, ibv_pd * pd)
        {
            close();
            BOOST_ASSERT(buffer_);
            buffer_mr_ = ibv_reg_mr(pd, buffer_, buffer_size_, IBV_ACCESS_LOCAL_WRITE);
            if(!buffer_mr_)
            {
                // FIXME: error
            }
            msg_mr_ = ibv_reg_mr(pd, msg_, sizeof(message), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            if(!msg_mr_)
            {
                // FIXME: error
            }

            post_receive(id);
        }

        void on_connect(rdma_cm_id * id)
        {
            id_ = id;
        }

        void write_remote(rdma_cm_id * id = NULL, std::size_t chunk_size = 0, bool no_data = true)
        {
            ibv_send_wr wr, *bad_wr = NULL;
            ibv_sge sge;

            std::memset(&wr, 0, sizeof(wr));

            wr.wr_id = (uintptr_t)id_;
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.imm_data = htonl(chunk_size);
            wr.wr.rdma.remote_addr = peer_addr_;
            wr.wr.rdma.rkey = peer_rkey_;

            if(chunk_size && no_data)
            {
                wr.sg_list = &sge;
                wr.num_sge = 1;

                sge.addr = (uintptr_t)buffer_;
                sge.length = chunk_size;
                sge.lkey = buffer_mr_->lkey;
            }
            int ret = ibv_post_send(id_->qp, &wr, &bad_wr);
            if(ret)
            {
            }
        }

        void send_next_chunk(rdma_cm_id * id, std::size_t chunk_size)
        {
            write_remote(id, chunk_size);
        }
        
        message_type on_completion(ibv_wc * wc)
        {
            rdma_cm_id * id = (rdma_cm_id *)(uintptr_t)(wc->wr_id);
            if(wc->opcode & IBV_WC_RECV)
            {
                switch(msg_->id)
                {
                    case MSG_MR:
                        peer_addr_ = msg_->addr;
                        peer_rkey_ = msg_->rkey;
                        write_remote(id, 0);
                        post_receive(id);
                        return MSG_MR;
                    case MSG_READY:
                        BOOST_ASSERT(size_ != 0);
                        write_remote(id, size_, false);
                        post_receive(id);
                        return MSG_READY;
                    case MSG_DATA:
                        write_remote(id, size_);
                        post_receive(id);
                        return MSG_DATA;
                    case MSG_DONE:
                        write_remote(id, 0);
                        post_receive(id);
                        return MSG_DONE;
                }
                return MSG_INVALID;
            }

            return MSG_RETRY;
        }
        
        void on_disconnect(rdma_cm_id * id)
        {
        }

        char *buffer_;
        std::size_t buffer_size_;
        ibv_mr *buffer_mr_;

        message *msg_;
        ibv_mr *msg_mr_;

        uint64_t peer_addr_;
        uint32_t peer_rkey_;

        std::size_t size_;
        
        rdma_cm_id *id_;
    };
}}}}

#endif
