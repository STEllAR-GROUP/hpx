//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_SERVER_HPP)
#define HPX_PARCELSET_IBVERBS_SERVER_HPP

#include <hpx/runtime/parcelset/ibverbs/messages.hpp>
#include <tcmalloc.h>

namespace hpx { namespace parcelset { namespace ibverbs { namespace detail {
    struct server
    {
        server()
          : buffer_(0)
          , buffer_size_(0)
          , buffer_mr_(0)
          , msg_(0)
          , msg_mr_(0)
          , size_(0)
          , id_(0)
        {
            int ret = 0;
            ret = posix_memalign(reinterpret_cast<void **>(&msg_), EXEC_PAGESIZE, sizeof(message));
            BOOST_ASSERT(ret == 0);
        }

        ~server()
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

            size_ = 0;
        }

        void post_receive(rdma_cm_id * id = NULL)
        {
            struct ibv_recv_wr wr, *bad_wr = NULL;

            std::memset(&wr, 0, sizeof(ibv_recv_wr));

            if(id == NULL)
            {
                BOOST_ASSERT(id_);
                wr.wr_id = (uintptr_t)id_;
            }
            else
            {
                wr.wr_id = (uintptr_t)id;
            }
            wr.sg_list = NULL;
            wr.num_sge = 0;

            int ret = 0;
            if(id == NULL)
            {
                BOOST_ASSERT(id_);
                ret = ibv_post_recv(id_->qp, &wr, &bad_wr);
            }
            else
            {
                ret = ibv_post_recv(id->qp, &wr, &bad_wr);
            }
            if(ret)
            {
                // FIXME: error
            }
        }

        void send_message(rdma_cm_id * id = NULL)
        {
            struct ibv_send_wr wr, *bad_wr = NULL;
            struct ibv_sge sge;
                
            std::memset(&wr, 0, sizeof(ibv_recv_wr));
            
            if(id == NULL)
            {
                BOOST_ASSERT(id_);
                wr.wr_id = (uintptr_t)id_;
            }
            else
            {
                wr.wr_id = (uintptr_t)id;
            }
            wr.opcode = IBV_WR_SEND;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.send_flags = IBV_SEND_SIGNALED;

            sge.addr = (uintptr_t)msg_;
            sge.length = sizeof(hpx::parcelset::ibverbs::message);
            sge.lkey = msg_mr_->lkey;

            int ret = 0;
            if(id == NULL)
            {
                BOOST_ASSERT(id_);
                ret = ibv_post_send(id_->qp, &wr, &bad_wr);
            }
            else
            {
                ret = ibv_post_send(id->qp, &wr, &bad_wr);
            }
            if(ret)
            {
                // FIXME: error
            }
        }
        
        void on_preconnect(rdma_cm_id * id, ibv_pd * pd)
        {
            close();
            BOOST_ASSERT(buffer_);
            buffer_mr_ = ibv_reg_mr(
                pd
              , buffer_
              , buffer_size_
              , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
            );
            if(!buffer_mr_)
            {
                // FIXME: error
            }
            msg_mr_ = ibv_reg_mr(
                pd
              , msg_
              , sizeof(hpx::parcelset::ibverbs::message)
              , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
            );
            if(!msg_mr_)
            {
                // FIXME: error
            }

            post_receive(id);
        }

        void on_connection(rdma_cm_id *id)
        {
            id_ = id;
            msg_->id = hpx::parcelset::ibverbs::message_type::MSG_MR;
            msg_->addr = (uintptr_t)buffer_mr_->addr;
            msg_->rkey = buffer_mr_->rkey;

            send_message(id);
        }
        
        message_type on_completion(ibv_wc * wc)
        {
            rdma_cm_id * id = (rdma_cm_id *)(uintptr_t)(wc->wr_id);
            if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
            {
                boost::uint32_t len = ntohl(wc->imm_data);

                if(len == 0)
                {
                    if(size_ == 0)
                    {
                        size_ = 0;
                        post_receive(id);
                        msg_->id = MSG_READY;
                        send_message(id);
                        return MSG_READY;
                    }
                    else
                    {
                        size_ = 0;
                        post_receive(id);
                        msg_->id = MSG_DONE;
                        send_message(id);
                        return MSG_DONE;
                    }
                }
                else
                {
                    if(size_ == 0)
                    {
                        size_ = len;
                        post_receive(id);
                        msg_->id = MSG_DATA;
                        send_message(id);
                        return MSG_DATA;
                    }
                    else
                    {
                        size_ = len;
                        id_ = id;
                        // We don't do anything here, the handler takes care of
                        // sending the next message and register the next receive as
                        // we need to copy stuff out of the rdma buffer first
                        return MSG_DATA;
                    }
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

        std::size_t size_;

        rdma_cm_id *id_;
    };
}}}}

#endif
