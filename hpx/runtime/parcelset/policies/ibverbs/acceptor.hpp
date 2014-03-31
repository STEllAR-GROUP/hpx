//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_POLICIES_IBVERBS_ACCEPTOR_HPP)
#define HPX_PARCELSET_POLICIES_IBVERBS_ACCEPTOR_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/ibverbs_errors.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/context.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/receiver.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/helper.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/system/system_error.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/scope_exit.hpp>
#include <boost/atomic.hpp>

#include <netdb.h>
#include <rdma/rdma_cma.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    ///////////////////////////////////////////////////////////////////////////
    class acceptor
    {
    public:
        acceptor()
          : event_channel_(0),
            listener_(0),
            executing_operation_(0),
            aborted_(false),
            close_operation_(false)
        {}

        ~acceptor()
        {
            boost::system::error_code ec;
            close(ec);
        }

        void open(boost::system::error_code &ec)
        {
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep
          , boost::system::error_code &ec)
        {
            if(event_channel_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else
            {
                event_channel_ = rdma_create_event_channel();
                if(!event_channel_)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
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
                ret = rdma_create_id(event_channel_, &listener_, NULL, RDMA_PS_TCP);

                if(ret)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }

                std::string host = ep.address().to_string();
                std::string port = boost::lexical_cast<std::string>(ep.port());

                addrinfo *addr;

                getaddrinfo(host.c_str(), port.c_str(), NULL, &addr);

                ret = rdma_bind_addr(listener_, addr->ai_addr);

                freeaddrinfo(addr);
                if(ret)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }
                ret = rdma_listen(listener_, 10); /* backlog = 10 is arbitrary */
                if(ret)
                {
                    int verrno = errno;
                    close(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }
                HPX_IBVERBS_RESET_EC(ec);
            }
        }

        void close(boost::system::error_code &ec)
        {
            if(!event_channel_)
            {
                HPX_IBVERBS_RESET_EC(ec);
                return;
            }

            close_operation_.store(true);

            BOOST_SCOPE_EXIT(&close_operation_) {
                close_operation_.store(false);
            } BOOST_SCOPE_EXIT_END


            // wait for pending operations to return
            while (executing_operation_.load())
                ;

            if(event_channel_)
            {
                rdma_destroy_event_channel(event_channel_);
                event_channel_ = 0;
            }
            else {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
            }

            if(listener_)
            {
                rdma_destroy_id(listener_);
                listener_ = 0;
            }

            HPX_IBVERBS_RESET_EC(ec);
        }

        void destroy()
        {
            aborted_.store(true);
            BOOST_SCOPE_EXIT(&aborted_) {
                aborted_.store(false);
            } BOOST_SCOPE_EXIT_END

            // cancel operation
            while (executing_operation_.load())
                ;
        }

        bool get_next_rdma_event(rdma_cm_event & event, boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END

            if (close_operation_.load() || !event_channel_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return false;
            }

            if (aborted_.load()) {
                aborted_.store(false);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
                return false;
            }

            return get_next_event(event_channel_, event, this, ec);
        }

        template <typename Parcelport>
        boost::shared_ptr<receiver> accept(
            Parcelport & parcelport, memory_pool & pool, boost::system::error_code &ec)
        {
            boost::shared_ptr<receiver> rcv;
            if (close_operation_.load() || !event_channel_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return rcv;
            }

            rdma_cm_event event;
            if(!get_next_rdma_event(event, ec))
            {
                return rcv;
            }

            if(event.event == RDMA_CM_EVENT_CONNECT_REQUEST)
            {
                rdma_conn_param cm_params;
                std::memset(&cm_params, 0, sizeof(rdma_conn_param));
                cm_params.initiator_depth = cm_params.responder_resources = 1;
                cm_params.rnr_retry_count = 7; // infinite retry

                rcv.reset(new receiver(parcelport, pool));
                rcv->context().build_connection(parcelport, event.id, ec);
                if(ec)
                {
                    rcv.reset();
                    return rcv;
                }

                rcv->context().on_preconnect(event.id, ec);
                if(ec)
                {
                    rcv.reset();
                    return rcv;
                }

                rdma_accept(event.id, &cm_params);
                pending_recv_list.push_back(std::make_pair(event, rcv));
                rcv.reset();
                return rcv;
            }

            if(event.event == RDMA_CM_EVENT_ESTABLISHED)
            {
                for(pending_recv_list_type::iterator it = pending_recv_list.begin();
                    it != pending_recv_list.end();)
                {
                    if(it->first.id == event.id)
                    {
                        rcv = it->second;
                        rcv->context().on_connection(event.id, ec);
                        it = pending_recv_list.erase(it);
                        break;
                    }
                    else
                    {
                        ++it;
                    }
                }
                HPX_ASSERT(rcv);
            }

            return rcv;
        }

        void on_disconnect(rdma_cm_id * id)
        {
        }

    private:
        rdma_event_channel *event_channel_;
        rdma_cm_id *listener_;

        boost::atomic<boost::uint16_t> executing_operation_;
        boost::atomic<bool> aborted_;
        boost::atomic<bool> close_operation_;

        typedef
            std::list<std::pair<rdma_cm_event, boost::shared_ptr<receiver> > >
            pending_recv_list_type;

        pending_recv_list_type pending_recv_list;
    };
}}}}

#endif


