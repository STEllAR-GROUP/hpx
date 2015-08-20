//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_POLICIES_IBVERBS_CONTEXT_HPP)
#define HPX_PARCELSET_POLICIES_IBVERBS_CONTEXT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/parcelport/ibverbs/ibverbs_errors.hpp>
#include <hpx/plugins/parcelport/ibverbs/helper.hpp>
#include <hpx/plugins/parcelport/ibverbs/client.hpp>
#include <hpx/plugins/parcelport/ibverbs/server.hpp>
#include <hpx/plugins/parcelport/ibverbs/data_buffer.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/apply.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/system/system_error.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/scope_exit.hpp>
#include <boost/atomic.hpp>
#include <boost/lexical_cast.hpp>

#include <netdb.h>
#include <rdma/rdma_cma.h>

#include <poll.h>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    template <bool Retry, int N = 1>
    struct next_wc;

    template <int N>
    struct next_wc<true, N>
    {
        template <typename This>
        static message_type call(This * this_, boost::system::error_code &ec)
        {
            int ret = -1;

            HPX_IBVERBS_RESET_EC(ec);

            ibv_cq * cq;
            ibv_wc wc[N];
            void *dummy = NULL;

            while(ret == -1)
            {
                ret = ibv_get_cq_event(this_->comp_channel_, &cq, &dummy);
                if(ret == -1)
                {
                    int err = errno;
                    if(err != EAGAIN)
                    {
                        HPX_IBVERBS_THROWS_IF(ec,
                            boost::asio::error::connection_aborted);
                        return MSG_INVALID;
                    }
                }
            }
            ibv_ack_cq_events(cq, N);
            ibv_req_notify_cq(cq, 0);


            message_type m = MSG_RETRY;
            while(m == MSG_RETRY)
            {
                int n = ibv_poll_cq(cq, N, wc);
                if(n)
                {
                    for(int i = 0; i < n-1; ++i)
                    {
                        if(wc[i].status == IBV_WC_SUCCESS)
                        {
                            if(this_->connection_.on_completion(&wc[i],
                                ec) == MSG_SHUTDOWN)
                            {
                                return MSG_SHUTDOWN;
                            }
                        }
                    }
                    if(wc[n-1].status == IBV_WC_SUCCESS)
                    {
                        m = this_->connection_.on_completion(&wc[n-1], ec);
                    }
                    else
                    {
                        HPX_IBVERBS_THROWS_IF(ec,
                            boost::asio::error::connection_aborted);
                        return MSG_INVALID;
                    }
                }
            }
            return m;
        }
    };

    template <int N>
    struct next_wc<false, N>
    {
        template <typename This>
        static message_type call(This * this_, boost::system::error_code &ec)
        {
            int ret = 0;

            HPX_IBVERBS_RESET_EC(ec);

            ibv_cq * cq;
            ibv_wc wc[N];
            void *dummy = NULL;
            ret = ibv_get_cq_event(this_->comp_channel_, &cq, &dummy);
            if(ret == -1)
            {
                int err = errno;
                if(err == EAGAIN) return MSG_RETRY;
                else
                {
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
                    return MSG_INVALID;
                }
            }
            ibv_ack_cq_events(cq, N);
            ibv_req_notify_cq(cq, 0);

            int n = ibv_poll_cq(cq, N, wc);
            if(n)
            {
                for(int i = 0; i < n-1; ++i)
                {
                    if(wc[i].status == IBV_WC_SUCCESS)
                    {
                        if(this_->connection_.on_completion(&wc[i], ec) == MSG_SHUTDOWN)
                        {
                            return MSG_SHUTDOWN;
                        }
                    }
                }
                if(wc[n-1].status == IBV_WC_SUCCESS)
                {
                    return this_->connection_.on_completion(&wc[n-1], ec);
                }
                else
                {
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
                    return MSG_INVALID;
                }
            }
            return MSG_RETRY;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Connection>
    class context_impl
    {
    public:

        typedef util::function_nonser<void(rdma_cm_id *)> callback_function;

        explicit context_impl()
          : pd_(0)
          , event_channel_(0)
          , conn_(0)
          , ctx_(0)
          , cq_(0)
          , comp_channel_(0)
          , connection_()
        {}

        ~context_impl()
        {
            boost::system::error_code ec;
            shutdown(ec);
            close(ec);
        }

    public:

        void on_preconnect(rdma_cm_id * id, boost::system::error_code & ec)
        {
            connection_.on_preconnect(id, pd_, ec);
        }

        void on_connection(rdma_cm_id * id, boost::system::error_code & ec)
        {
            connection_.on_connection(id, ec);
            if(ec)
            {
                return;
            }
        }

        void post_receive(boost::system::error_code & ec, bool payload = false)
        {
            connection_.post_receive(ec, payload);
        }

        template <bool Retry>
        bool check_wc(message_type state, boost::system::error_code & ec)
        {
            message_type m = next_wc<Retry, 1>::call(this, ec);
            if(ec) return false;
            if(m == MSG_SHUTDOWN)
            {
                close(ec);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::eof);
                return false;
            }
            if(m != state)
            {
                return false;
            }
            return true;
        }

        void register_buffer(char * buffer, std::size_t size,
            boost::system::error_code & ec)
        {
            connection_.register_buffer(pd_, buffer, size, ec);
        }

        void on_completion(ibv_wc * wc, boost::system::error_code & ec)
        {
            connection_.on_completion(wc, ec);
        }

        void on_disconnect(rdma_cm_id * id)
        {
            connection_.on_disconnect(id);
        }

        void open(boost::system::error_code &ec)
        {
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep, boost::system::error_code &ec)
        {
            HPX_IBVERBS_RESET_EC(ec);
            if(ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::already_connected);
            }
        }

        void close(boost::system::error_code &ec)
        {
            if(!ctx_)
            {
                HPX_IBVERBS_RESET_EC(ec);
                return;
            }

            connection_.close();

            if(conn_)
            {
                rdma_destroy_id(conn_);
                conn_ = 0;
            }
            if(event_channel_)
            {
                rdma_destroy_event_channel(event_channel_);
                event_channel_ = 0;
            }
            if(comp_channel_)
            {
                ibv_destroy_comp_channel(comp_channel_);
                event_channel_ = 0;
            }

            if(pd_)
            {
                pd_ = 0;
            }

            if(cq_)
            {
                ibv_destroy_cq(cq_);
                cq_ = 0;
            }

            ctx_ = 0;
            HPX_IBVERBS_RESET_EC(ec);
        }

        void shutdown(boost::system::error_code &ec)
        {
            if(ctx_)
            {
                connection_.send_shutdown(ec);
            }
        }

        void destroy()
        {
        }

        template <typename Parcelport>
        void connect(Parcelport & parcelport,
            boost::asio::ip::tcp::endpoint const & there, boost::system::error_code &ec)
        {
            if(ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::already_connected);
                return;
            }
            else
            {
                HPX_IBVERBS_RESET_EC(ec);

                event_channel_ = rdma_create_event_channel();
                if(!event_channel_)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno,
                        boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }
                set_nonblocking(event_channel_->fd, ec);
                if(ec)
                {
                    close(ec);
                    return;
                }

                int ret = 0;

                ret = rdma_create_id(event_channel_, &conn_, NULL, RDMA_PS_TCP);
                if(ret)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno,
                        boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }

                addrinfo *addr;
                ret = getaddrinfo(
                    there.address().to_string().c_str()
                  , boost::lexical_cast<std::string>(there.port()).c_str()
                  , NULL
                  , &addr
                );

                if(ret)
                {
                    close(ec);
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , boost::asio::detail::socket_ops::translate_addrinfo_error(ret)
                    );
                    return;
                }
                const int TIMEOUT_IN_MS = 500;
                ret = rdma_resolve_addr(conn_, NULL, addr->ai_addr, TIMEOUT_IN_MS);
                freeaddrinfo(addr);

                if(ret)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno,
                        boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }

                // build params ...
                rdma_conn_param cm_params;
                std::memset(&cm_params, 0, sizeof(rdma_conn_param));
                cm_params.initiator_depth = cm_params.responder_resources = 1;
                cm_params.rnr_retry_count = 7; /* infinite retry */

                rdma_cm_event event;
                std::size_t k = 0;
                while(!get_next_event(event_channel_, event, this, ec))
                {
                    if(ec)
                    {
                        close(ec);
                    }
                    hpx::lcos::local::spinlock::yield(k);
                    ++k;
                }

                if(event.event == RDMA_CM_EVENT_ADDR_RESOLVED)
                {
                    // building connection ...
                    build_connection(parcelport, event.id, ec);

                    connection_.on_preconnect(event.id, pd_, ec);
                    if(ec)
                    {
                        close(ec);
                        return;
                    }

                    ret = rdma_resolve_route(event.id, TIMEOUT_IN_MS);
                    if(ret)
                    {
                        int verrno = errno;
                        close(ec);
                        boost::system::error_code err(verrno,
                            boost::system::system_category());
                        HPX_IBVERBS_THROWS_IF(
                            ec
                          , err
                        );
                        return;
                    }
                    k = 0;
                    while(!get_next_event(event_channel_, event, this, ec))
                    {
                        if(ec)
                        {
                            close(ec);
                            return;
                        }
                        hpx::lcos::local::spinlock::yield(k);
                        ++k;
                    }
                    if(event.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
                    {
                        ret = rdma_connect(event.id, &cm_params);

                        if(ret)
                        {
                            int verrno = errno;
                            close(ec);
                            boost::system::error_code err(verrno,
                                boost::system::system_category());
                            HPX_IBVERBS_THROWS_IF(
                                ec
                              , err
                            );
                            return;
                        }

                        k = 0;
                        while(!get_next_event(event_channel_, event, this, ec))
                        {
                            if(ec)
                            {
                                close(ec);
                                return;
                            }
                            hpx::lcos::local::spinlock::yield(k);
                            ++k;
                        }

                        if(event.event == RDMA_CM_EVENT_ESTABLISHED)
                        {
                            connection_.on_connection(event.id);

                            HPX_IBVERBS_RESET_EC(ec);
                            return;
                        }
                    }
                }
                else
                {
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::host_unreachable);
                }
            }
        }

        void send_size(boost::uint64_t size, boost::system::error_code &ec)
        {
            connection_.send_size(size, ec);
        }

        template <typename Parcelport>
        void build_connection(Parcelport & parcelport, rdma_cm_id * id,
            boost::system::error_code &ec)
        {
            ibv_qp_init_attr qp_attr;
            build_context(parcelport, id->verbs, ec);
            build_qp_attr(&qp_attr);

            int ret = rdma_create_qp(id, pd_, &qp_attr);

            if(ret)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
            }
        }

        rdma_cm_id *conn_id()
        {
            return connection_.id_;
        }

        Connection const & connection() const
        {
            return connection_;
        }

        Connection & connection()
        {
            return connection_;
        }

        ibv_pd * pd_;
    private:
        template <typename Parcelport>
        void build_context(Parcelport & parcelport, ibv_context *verbs,
            boost::system::error_code &ec)
        {
            if(ctx_)
            {
                if(ctx_ != verbs)
                {
                    HPX_IBVERBS_THROWS_IF(ec,
                        boost::asio::error::operation_not_supported);
                    return;
                }
            }

            ctx_ = verbs;

            pd_ = parcelport.get_pd(ctx_, ec);

            comp_channel_ = ibv_create_comp_channel(ctx_);
            if(!comp_channel_)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return;
            }
            set_nonblocking(comp_channel_->fd, ec);
            if(ec)
            {
                return;
            }

            cq_ = ibv_create_cq(ctx_, 10, NULL, comp_channel_, 0);
            if(!cq_)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return;
            }

            int ret = ibv_req_notify_cq(cq_, 0);
            if(ret)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return;
            }
        }

        void build_qp_attr(ibv_qp_init_attr *qp_attr)
        {
            std::memset(qp_attr, 0, sizeof(ibv_qp_init_attr));

            qp_attr->send_cq = cq_;
            qp_attr->recv_cq = cq_;
            qp_attr->qp_type = IBV_QPT_RC;

            qp_attr->cap.max_send_wr = 10;
            qp_attr->cap.max_recv_wr = 10;
            qp_attr->cap.max_send_sge = 2;
            qp_attr->cap.max_recv_sge = 1;
        }

        rdma_event_channel *event_channel_;
        rdma_cm_id *conn_;

        ibv_context *ctx_;
        ibv_cq * cq_;
        ibv_comp_channel *comp_channel_;

        Connection connection_;

        template <bool, int> friend struct next_wc;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef context_impl<detail::client> client_context;

    typedef context_impl<detail::server> server_context;
}}}}

#endif
