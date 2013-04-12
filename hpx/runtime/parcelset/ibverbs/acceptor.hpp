//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_ACCEPTOR_HPP)
#define HPX_PARCELSET_IBVERBS_ACCEPTOR_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/ibverbs/ibverbs_errors.hpp>
#include <hpx/runtime/parcelset/ibverbs/context.hpp>
#include <hpx/runtime/parcelset/ibverbs/helper.hpp>
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
namespace hpx { namespace parcelset { namespace ibverbs
{
    ///////////////////////////////////////////////////////////////////////////
    //
    struct basic_acceptor_options
    {
        template <typename T>
        struct option
        {
            /*
            option(T const& num) : val_(num) {}
            T val_;
            */
        };

        /*
        typedef option<std::size_t> msg_num;
        typedef option<bool> manage;
        */
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Service>
    class basic_acceptor
      : public boost::asio::basic_io_object<Service>,
        public basic_acceptor_options
    {
    public:
        explicit basic_acceptor(boost::asio::io_service &io_service)
          : boost::asio::basic_io_object<Service>(io_service)
        {
        }

        void open(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.open(this->implementation, ec);
        }

        void close(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.close(this->implementation, ec);
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.bind(this->implementation, ep, ec);
        }

        // synchronous and asynchronous accept
        template <typename Service_>
        void accept(basic_context<Service_> & conn,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.accept(this->implementation, conn, ec);
        }

        template <typename Service_, typename Handler>
        void async_accept(basic_context<Service_> & conn, Handler handler)
        {
            this->service.async_accept(this->implementation, conn, handler);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation, typename Service>
        class accept_operation 
          : public boost::enable_shared_from_this<
                accept_operation<Handler, Implementation, Service> >
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            accept_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  basic_context<Service> & conn, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                conn_(conn),
                handler_(handler)
            {}

            void call() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    if (!impl->get_connect_request(conn_, ec) && !ec) {
                        // repost this function
                        io_service_.post(boost::bind(
                            &accept_operation::call, this->shared_from_this()));
                    }
                    else {
                        io_service_.post(boost::bind(
                            &accept_operation::try_accept, this->shared_from_this()));
                    }
                }
                else
                {
                    io_service_.post(
                        boost::asio::detail::bind_handler(handler_,
                            boost::asio::error::operation_aborted));
                }
            }

        private:
            void try_accept() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    if (!impl->try_accept(conn_, ec) && !ec) {
                        // repost this function
                        io_service_.post(boost::bind(
                            &accept_operation::try_accept, this->shared_from_this()));
                    }
                    else {
                        io_service_.post(
                            boost::asio::detail::bind_handler(handler_, ec));
                    }
                }
                else
                {
                    io_service_.post(
                        boost::asio::detail::bind_handler(handler_,
                            boost::asio::error::operation_aborted));
                }
            }

            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            basic_context<Service> & conn_;
            Handler handler_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Implementation>
    class basic_acceptor_service
      : public boost::asio::io_service::service
    {
    public:
        static boost::asio::io_service::id id;

        explicit basic_acceptor_service(boost::asio::io_service &io_service)
          : boost::asio::io_service::service(io_service)
        {}

        ~basic_acceptor_service()
        {}

        typedef boost::shared_ptr<Implementation> implementation_type;

        void construct(implementation_type &impl)
        {
            impl.reset(new Implementation());
        }

        void destroy(implementation_type &impl)
        {
            impl->destroy();
            impl.reset();
        }

        void open(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->open(ec);
            boost::asio::detail::throw_error(ec);
        }

        void close(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->close(ec);
            boost::asio::detail::throw_error(ec);
        }

        void bind(implementation_type &impl,
            boost::asio::ip::tcp::endpoint const & ep,
            boost::system::error_code &ec)
        {
            impl->bind(ep, ec);
            boost::asio::detail::throw_error(ec);
        }

        // synchronous and asynchronous accept
        template <typename Service>
        void accept(implementation_type &impl,
            basic_context<Service> & conn, boost::system::error_code &ec)
        {
            while (!impl->try_get_connect_request(conn, ec) && !ec)
                /* just wait for operation to succeed */;
            while (!impl->try_accept(conn, ec) && !ec)
                /* just wait for operation to succeed */;
            boost::asio::detail::throw_error(ec);
        }

        template <typename Service, typename Handler>
        void async_accept(implementation_type &impl,
            basic_context<Service> & conn, Handler handler)
        {
            typedef detail::accept_operation<Handler, Implementation, Service>
                operation_type;

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), conn, handler));

            this->get_io_service().post(boost::bind(&operation_type::call, op));
        }

    private:
        void shutdown_service()
        {}
    };

    template <typename Implementation>
    boost::asio::io_service::id basic_acceptor_service<Implementation>::id;

    ///////////////////////////////////////////////////////////////////////////
    class acceptor_impl
    {
    public:
        acceptor_impl()
          : event_channel_(0),
            listener_(0),
            executing_operation_(0),
            aborted_(false),
            close_operation_(false)
        {}

        ~acceptor_impl()
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

                std::string host = ep.address().to_v4().to_string();
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

        template <typename Service>
        bool get_connect_request(basic_context<Service> & conn, boost::system::error_code &ec)
        {
            if (close_operation_.load() || !event_channel_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return false;
            }

            rdma_cm_event event;
            if(!get_next_rdma_event(event, ec))
            {
                return false;
            }

            if(event.event == RDMA_CM_EVENT_CONNECT_REQUEST)
            {
                rdma_conn_param cm_params;
                std::memset(&cm_params, 0, sizeof(rdma_conn_param));
                cm_params.initiator_depth = cm_params.responder_resources = 1;
                cm_params.rnr_retry_count = 7; // infinite retry

                conn.build_connection(event.id, ec);
                if(ec)
                {
                    return false;
                }

                conn.on_preconnect(event.id, ec);
                if(ec)
                {
                    return false;
                }

                rdma_accept(event.id, &cm_params);
                return true;
            }
            return false;
        }

        template <typename Service>
        bool try_accept(basic_context<Service> & conn, boost::system::error_code &ec)
        {
            if (close_operation_.load() || !event_channel_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return false;
            }

            rdma_cm_event event;
            if(!get_next_rdma_event(event, ec))
            {
                return false;
            }
            if(ec)
            {
                return false;
            }
            if(event.event == RDMA_CM_EVENT_ESTABLISHED)
            {
                conn.on_connection(event.id, ec);
                if(ec)
                {
                    return false;
                }
                HPX_IBVERBS_RESET_EC(ec);
                return true;
            }

            return false;
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
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_acceptor<
        basic_acceptor_service<acceptor_impl>
    > acceptor;
}}}

#endif


