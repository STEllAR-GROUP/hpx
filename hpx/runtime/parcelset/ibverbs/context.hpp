//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_CONTEXT_HPP)
#define HPX_PARCELSET_IBVERBS_CONTEXT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/ibverbs/ibverbs_errors.hpp>
#include <hpx/runtime/parcelset/ibverbs/helper.hpp>
#include <hpx/runtime/parcelset/ibverbs/client.hpp>
#include <hpx/runtime/parcelset/ibverbs/server.hpp>
#include <hpx/runtime/parcelset/ibverbs/data_buffer.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/apply.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
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

namespace hpx { namespace parcelset { namespace ibverbs
{
    struct basic_context_options
    {
    };

    template <typename Service>
    struct basic_context
      : public boost::asio::basic_io_object<Service>,
        public basic_context_options
    {
    public:

        basic_context(
            boost::asio::io_service &io_service
        )
          : boost::asio::basic_io_object<Service>(io_service)
        {
        }

        void create(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.create(this->implementation, ec);
        }

        void open(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.open(this->implementation, ec);
        }

        void close(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.close(this->implementation, ec);
        }

        void shutdown(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.shutdown(this->implementation, ec);
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.bind(this->implementation, ep, ec);
        }

        char * set_buffer_size(std::size_t buffer_size, boost::system::error_code &ec)
        {
            return this->service.set_buffer_size(this->implementation, buffer_size, ec);
        }

        void build_connection(rdma_cm_id * id, boost::system::error_code &ec)
        {
            this->service.build_connection(this->implementation, id, ec);
        }

        rdma_cm_id *conn_id()
        {
            return this->service.conn_id(this->implementation);
        }

        void on_preconnect(rdma_cm_id * id, boost::system::error_code &ec)
        {
            this->service.on_preconnect(this->implementation, id, ec);
        }

        void on_connection(rdma_cm_id * id, boost::system::error_code &ec)
        {
            this->service.on_connection(this->implementation, id, ec);
        }

        void on_completion(ibv_wc * wc, boost::system::error_code &ec)
        {
            this->service.on_completion(this->implementation, wc, ec);
        }

        void on_disconnect(rdma_cm_id * id)
        {
            this->service.on_disconnect(this->implementation, id);
        }

        // synchronous and asynchronous connect
        void connect(
            boost::asio::ip::tcp::endpoint const & there,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.connect(this->implementation, there, ec);
        }

        template <typename Handler>
        void async_connect(
            boost::asio::ip::tcp::endpoint const & there,
            Handler handler)
        {
            this->service.async_connect(this->implementation, there, handler);
        }

        // synchronous and asynchronous read/write/read_ack/write_ack
        void read(std::vector<char>& data,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.read(this->implementation, data, ec);
        }

        std::size_t write(data_buffer const& data,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.write(this->implementation, data, ec);
        }

        template <typename Handler>
        void async_read(data_buffer & data, Handler handler)
        {
            this->service.async_read(this->implementation, data, handler);
        }

        void read_ack(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.read_ack(this->implementation, ec);
        }

        template <typename Handler>
        void async_read_ack(Handler handler)
        {
            this->service.async_read_ack(this->implementation, handler);
        }

        template <typename Handler>
        void async_write(data_buffer const & data, Handler handler)
        {
            this->service.async_write(this->implementation, data, handler);
        }

        template <typename Handler>
        void async_write_ack(Handler handler)
        {
            this->service.async_write_ack(this->implementation, handler);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Handler, typename Implementation>
        class connect_operation
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            connect_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  boost::asio::ip::tcp::endpoint const & there, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                there_(there),
                handler_(handler)
            {}

            void operator()() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    impl->connect(there_, ec);
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, ec));
                }
                else
                {
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, boost::asio::error::operation_aborted));
                }
            }

        private:
            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            boost::asio::ip::tcp::endpoint const& there_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class read_operation
          : public boost::enable_shared_from_this<
                read_operation<Handler, Implementation> >
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            read_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  data_buffer & data, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                data_(data),
                handler_(handler)
            {}

            void call() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;

                    std::size_t size = impl->try_read_data(data_, ec);
                    if (size == 0 && !ec) {
                        // repost this handler
                        io_service_.post(boost::bind(
                            &read_operation::call, this->shared_from_this()));
                    }
                    else if(!ec) {
                        // post completion handler
                        io_service_.post(boost::asio::detail::bind_handler(
                            handler_, ec, size));
                        /*
                        handler_(ec, size);
                        */
                    }
                }
                else
                {
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, boost::asio::error::operation_aborted, 0));
                }
            }

        private:

            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            data_buffer & data_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class read_ack_operation
          : public boost::enable_shared_from_this<
                read_ack_operation<Handler, Implementation> >
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            read_ack_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  Handler handler)
              : impl_(impl),
                io_service_(io_service),
                handler_(handler)
            {}

            void call() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;

                    bool ack = impl->try_read_ack(ec);
                    if (!ack && !ec) {
                        // repost this handler
                        io_service_.post(boost::bind(
                            &read_ack_operation::call, this->shared_from_this()));
                    }
                    else if(!ec) {
                        // post completion handler
                        io_service_.post(boost::asio::detail::bind_handler(
                            handler_, ec));
                        /*
                        handler_(ec);
                        */
                    }
                }
                else
                {
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, boost::asio::error::operation_aborted));
                }
            }

        private:

            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class write_operation
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            write_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  data_buffer const& data, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                data_(data),
                handler_(handler)
            {}

            void operator()() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    std::size_t size = impl->write(data_, ec);
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, ec, size));
                    /*
                    handler_(ec, size);
                    */
                }
                else
                {
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, boost::asio::error::operation_aborted, 0));
                }
            }

        private:
            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            data_buffer const& data_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class write_ack_operation
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            write_ack_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  Handler handler)
              : impl_(impl),
                io_service_(io_service),
                handler_(handler)
            {}

            void operator()() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    impl->write_ack(ec);
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, ec));
                    /*
                    handler_(ec);
                    */
                }
                else
                {
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, boost::asio::error::operation_aborted));
                }
            }

        private:
            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            Handler handler_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Implementation>
    class basic_context_service
      : public boost::asio::io_service::service
    {
    public:
        typedef HPX_STD_FUNCTION<void(rdma_cm_id *)> callback_function;

        static boost::asio::io_service::id id;

        explicit basic_context_service(boost::asio::io_service &io_service)
          : boost::asio::io_service::service(io_service)
        {}

        ~basic_context_service()
        {}

        typedef boost::shared_ptr<Implementation> implementation_type;

        ///////////////////////////////////////////////////////////////////////
        void construct(implementation_type &impl)
        {
            impl.reset(new Implementation());
        }

        void destroy(implementation_type &impl)
        {
            impl->destroy();
            impl.reset();
        }

        ///////////////////////////////////////////////////////////////////////
        void create(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->create(false, ec);      // create only
        }

        void open(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->open(ec);
        }

        void close(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->close(ec);
        }

        void shutdown(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->shutdown(ec);
        }

        char * set_buffer_size(implementation_type &impl, std::size_t buffer_size, boost::system::error_code &ec)
        {
            return impl->set_buffer_size(buffer_size, ec);
        }

        void build_connection(implementation_type &impl, rdma_cm_id * id, boost::system::error_code &ec)
        {
            impl->build_connection(id, ec);
        }

        rdma_cm_id *conn_id(implementation_type &impl)
        {
            return impl->conn_id();
        }

        void on_preconnect(implementation_type &impl, rdma_cm_id * id, boost::system::error_code &ec)
        {
            return impl->on_preconnect(id, ec);
        }

        void on_connection(implementation_type &impl, rdma_cm_id * id, boost::system::error_code &ec)
        {
            return impl->on_connection(id, ec);
        }

        void on_completion(implementation_type &impl, ibv_wc * wc, boost::system::error_code &ec)
        {
            return impl->on_completion(wc, ec);
        }

        void on_disconnect(implementation_type &impl, rdma_cm_id * id)
        {
            return impl->on_disconnect(id);
        }

        void bind(implementation_type &impl,
            boost::asio::ip::tcp::endpoint const & ep,
            boost::system::error_code &ec)
        {
            impl->bind(ep, ec);
        }

        // synchronous and asynchronous connect
        void connect(implementation_type &impl,
            boost::asio::ip::tcp::endpoint const & there,
            boost::system::error_code &ec)
        {
            impl->connect(there, ec);
        }

        template <typename Handler>
        void async_connect(implementation_type &impl,
            boost::asio::ip::tcp::endpoint const & there,
            Handler handler)
        {
            this->get_io_service().post(
                detail::connect_operation<Handler, Implementation>(
                    impl, this->get_io_service(), there, handler));
        }

        std::size_t write(implementation_type &impl, data_buffer const& data,
            boost::system::error_code &ec)
        {
            return impl->write(data, ec);
        }

        template <typename Handler>
        void async_read(implementation_type &impl, data_buffer & data,
            Handler handler)
        {
            typedef detail::read_operation<Handler, Implementation> operation_type;

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), data, handler));

            this->get_io_service().post(boost::bind(
                &operation_type::call, op));
        }

        void read_ack(implementation_type & impl, boost::system::error_code &ec)
        {
            while(!impl->try_read_ack(ec))
                ;
        }

        template <typename Handler>
        void async_read_ack(implementation_type &impl,
            Handler handler)
        {
            typedef detail::read_ack_operation<Handler, Implementation> operation_type;

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), handler));

            this->get_io_service().post(boost::bind(
                &operation_type::call, op));
        }

        template <typename Handler>
        void async_write(implementation_type &impl, data_buffer const& data,
            Handler handler)
        {
            this->get_io_service().post(
                detail::write_operation<Handler, Implementation>(
                    impl, this->get_io_service(), data, handler));
        }

        template <typename Handler>
        void async_write_ack(implementation_type &impl, Handler handler)
        {
            this->get_io_service().post(
                detail::write_ack_operation<Handler, Implementation>(
                    impl, this->get_io_service(), handler));
        }


    private:
        void shutdown_service()
        {}
    };

    template <typename Implementation>
    boost::asio::io_service::id basic_context_service<Implementation>::id;

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
                        HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
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
                    for(int i = 0; i < n; ++i)
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
                        m = this_->connection_.on_completion(&wc[n-1], ec);
                    }
                    else
                    {
                        HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
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
                for(int i = 0; i < n; ++i)
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

        typedef HPX_STD_FUNCTION<void(rdma_cm_id *)> callback_function;

        context_impl()
          : event_channel_(0)
          , conn_(0)
          , ctx_(0)
          , pd_(0)
          , cq_(0)
          , comp_channel_(0)
          , executing_operation_(0)
          , aborted_(false)
          , close_operation_(false)
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

            HPX_IBVERBS_NEXT_WC(ec, MSG_MR, 1, , true);
            connection_.post_receive(ec);
        }

        void on_completion(ibv_wc * wc, boost::system::error_code & ec)
        {
            connection_.on_completion(wc, ec);
        }

        void on_disconnect(rdma_cm_id * id)
        {
            connection_.on_disconnect(id);
        }

        char * set_buffer_size(std::size_t buffer_size, boost::system::error_code &ec)
        {
            return connection_.set_buffer_size(buffer_size, ec);
        }

        void open(boost::system::error_code &ec)
        {
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep, boost::system::error_code &ec)
        {
            util::spinlock::scoped_lock lk(mtx_);
            HPX_IBVERBS_RESET_EC(ec);
            if(ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::already_connected);
            }
        }

        void close(boost::system::error_code &ec)
        {
            util::spinlock::scoped_lock lk(mtx_);
            close_locked(ec);
        }

        void close_locked(boost::system::error_code &ec)
        {

            if(!ctx_)
            {
                HPX_IBVERBS_RESET_EC(ec);
                return;
            }

            close_operation_.store(true);
            BOOST_SCOPE_EXIT_TPL(&close_operation_) {
                close_operation_.store(false);
            } BOOST_SCOPE_EXIT_END

            // wait for pending operations to exit
            while (executing_operation_.load())
                ;

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
                ibv_dealloc_pd(pd_);
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
            util::spinlock::scoped_lock lk(mtx_);
            if(ctx_)
            {
                connection_.send_shutdown(ec);
            }
        }

        void destroy()
        {
        }

        void connect(boost::asio::ip::tcp::endpoint const & there, boost::system::error_code &ec)
        {
            util::spinlock::scoped_lock lk(mtx_);
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
                    close_locked(ec);
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
                    close_locked(ec);
                    return;
                }

                int ret = 0;

                ret = rdma_create_id(event_channel_, &conn_, NULL, RDMA_PS_TCP);
                if(ret)
                {
                    int verrno = errno;
                    close_locked(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
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
                    close_locked(ec);
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
                    close_locked(ec);
                    boost::system::error_code err(verrno, boost::system::system_category());
                    HPX_IBVERBS_THROWS_IF(
                        ec
                      , err
                    );
                    return;
                }

                int * i = new int(8);

                conn_->context = i;

                // build params ...
                rdma_conn_param cm_params;
                std::memset(&cm_params, 0, sizeof(rdma_conn_param));
                cm_params.initiator_depth = cm_params.responder_resources = 1;
                cm_params.rnr_retry_count = 7; /* infinite retry */

                rdma_cm_event event;
                int k = 0;
                while(!get_next_event(event_channel_, event, this, ec))
                {
                    if(ec)
                    {
                        close_locked(ec);
                    }
                    if(k > 4096)
                    {
                        close_locked(ec);
                        HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::try_again);
                        return;
                    }
                    ++k;
                }

                if(event.event == RDMA_CM_EVENT_ADDR_RESOLVED)
                {
                    // building connection ...
                    build_connection(event.id, ec);

                    connection_.on_preconnect(event.id, pd_, ec);
                    if(ec)
                    {
                        close_locked(ec);
                        return;
                    }

                    ret = rdma_resolve_route(event.id, TIMEOUT_IN_MS);
                    if(ret)
                    {
                        int verrno = errno;
                        close_locked(ec);
                        boost::system::error_code err(verrno, boost::system::system_category());
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
                            close_locked(ec);
                            return;
                        }
                        if(k > 4096)
                        {
                            close_locked(ec);
                            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::try_again);
                            return;
                        }
                        ++k;
                    }
                    if(event.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
                    {
                        ret = rdma_connect(event.id, &cm_params);

                        if(ret)
                        {
                            int verrno = errno;
                            close_locked(ec);
                            boost::system::error_code err(verrno, boost::system::system_category());
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
                                close_locked(ec);
                                return;
                            }
                            if(k > 4096)
                            {
                                close_locked(ec);
                                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::try_again);
                                return;
                            }
                            ++k;
                        }

                        if(event.event == RDMA_CM_EVENT_ESTABLISHED)
                        {
                            connection_.on_connection(event.id);
                            k = 0;
                            message_type m = MSG_RETRY;
                            while(m != MSG_MR)
                            {
                                m = next_wc<false, 1>::call(this, ec);
                                if(ec) return;
                                if(k > 4096 && m != MSG_MR)
                                {
                                    close_locked(ec);
                                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::try_again);
                                    return;
                                }
                                if(m == MSG_SHUTDOWN)
                                {
                                    close_locked(ec);
                                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::eof);
                                    return;
                                }
                                ++k;
                            }

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

        // read, write, and acknowledge operations
        std::size_t try_read_data(data_buffer & data, boost::system::error_code &ec)
        {
            util::spinlock::scoped_lock lk(mtx_);
            if (!ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return 0;
            }
            else
            {
                HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 1, 0, false);

                connection_.post_receive(ec);
                if(ec)
                {
                    close_locked(ec);
                    return 0;
                }

                boost::uint64_t header[2] = {0};

                std::memcpy(header, connection_.buffer_, sizeof(header));

                data.resize(header[0]);

                if(!data.zero_copy_)
                {
                    char * data_ptr = data.begin();
                    HPX_ASSERT(data_ptr != data.mr_buffer_);

                    HPX_ASSERT(header[1]);

                    std::memcpy(
                        data_ptr
                      , connection_.buffer_ + 2 * sizeof(boost::uint64_t)
                      , header[1]
                    );

                    boost::uint64_t chunk_size = 0;

                    connection_.send_message(MSG_DATA, ec);
                    if(ec)
                    {
                        close_locked(ec);
                        return 0;
                    }
                    HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 1, 0, true);

                    data_ptr += header[1];
                    const char * buffer_ptr = connection_.buffer_ + sizeof(boost::uint64_t);

                    for(std::size_t bytes_read = header[1]; bytes_read < header[0];)
                    {
                        HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 1, 0, true);

                        connection_.post_receive(ec);
                        if(ec)
                        {
                            close_locked(ec);
                            return 0;
                        }

                        std::memcpy(&chunk_size, connection_.buffer_, sizeof(boost::uint64_t));

                        HPX_ASSERT(chunk_size);

                        std::memcpy(
                            data_ptr
                          , buffer_ptr
                          , chunk_size
                        );

                        data_ptr += chunk_size;
                        bytes_read += chunk_size;

                        HPX_ASSERT(data.size() >= bytes_read);

                        if(bytes_read < header[0])
                        {
                            connection_.send_message(MSG_DATA, ec);
                            if(ec)
                            {
                                close_locked(ec);
                                return 0;
                            }
                            HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 1, 0, true);
                        }
                    }
                }

                {
                    boost::uint64_t tmp[2] = {0};
                    std::memcpy(connection_.buffer_, tmp, sizeof(tmp));
                }

                HPX_IBVERBS_RESET_EC(ec);
                return header[0];
            }
        }

        void write_ack(boost::system::error_code & ec)
        {
            util::spinlock::scoped_lock lk(mtx_);
            if (!ctx_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return;
            }

            connection_.send_ready(ec);
            if(ec)
            {
                close_locked(ec);
                return;
            }
            HPX_IBVERBS_NEXT_WC(ec, MSG_DONE, 1, , true);

            HPX_IBVERBS_RESET_EC(ec);
        }

        std::size_t write(data_buffer const& data, boost::system::error_code &ec)
        {
            util::spinlock::scoped_lock lk(mtx_);

            if(!ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return 0;
            }

            boost::uint64_t header[2] =
                {
                    data.size()
                  , std::min(
                        connection_.buffer_size_ - 2 * sizeof(boost::uint64_t)
                      , data.size()
                    )
                };

            std::memcpy(connection_.buffer_, header, 2 * sizeof(boost::uint64_t));

            if(data.zero_copy_)
            {
                connection_.post_receive(ec);
                if(ec)
                {
                    close_locked(ec);
                    return 0;
                }
                connection_.write_remote(header[1] + 2 * sizeof(boost::uint64_t), ec);
                if(ec)
                {
                    close_locked(ec);
                    return 0;
                }
            }
            else
            {
                const char * data_ptr = data.begin();

                std::memcpy(connection_.buffer_ + 2 * sizeof(boost::uint64_t), data_ptr, header[1]);

                connection_.post_receive(ec);
                if(ec)
                {
                    close_locked(ec);
                    return 0;
                }
                connection_.write_remote(header[1] + 2 * sizeof(boost::uint64_t), ec);
                if(ec)
                {
                    close_locked(ec);
                    return 0;
                }

                boost::uint64_t bytes_sent = header[1];
                data_ptr += header[1];

                char * buffer_ptr = connection_.buffer_ + sizeof(boost::uint64_t);

                while(bytes_sent != header[0])
                {
                    HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 3, 0, true);
                    boost::uint64_t chunk_size
                        = std::min(
                            connection_.buffer_size_ - sizeof(boost::uint64_t)
                          , header[0] - bytes_sent
                        );

                    std::memcpy(connection_.buffer_, &chunk_size, sizeof(boost::uint64_t));

                    std::memcpy(buffer_ptr, data_ptr, chunk_size);

                    connection_.post_receive(ec);
                    if(ec)
                    {
                        close_locked(ec);
                        return 0;
                    }
                    connection_.write_remote(chunk_size + sizeof(boost::uint64_t), ec);
                    if(ec)
                    {
                        close_locked(ec);
                        return 0;
                    }

                    bytes_sent += chunk_size;
                    data_ptr += chunk_size;
                }
            }

            HPX_IBVERBS_RESET_EC(ec);
            return header[0];
        }

        bool try_read_ack(boost::system::error_code & ec)
        {
            util::spinlock::scoped_lock lk(mtx_);

            if(!ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return false;
            }

            HPX_IBVERBS_NEXT_WC(ec, MSG_DONE, 3, false, false);

            HPX_IBVERBS_RESET_EC(ec);
            return true;
        }

        void build_connection(rdma_cm_id * id, boost::system::error_code &ec)
        {
            ibv_qp_init_attr qp_attr;
            build_context(id->verbs, ec);
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

    private:
        void build_context(ibv_context *verbs, boost::system::error_code &ec)
        {
            if(ctx_)
            {
                if(ctx_ != verbs)
                {
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::operation_not_supported);
                    return;
                }
            }

            ctx_ = verbs;

            pd_ = ibv_alloc_pd(ctx_);
            if(!pd_)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return;
            }
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
        ibv_pd * pd_;
        ibv_cq * cq_;
        ibv_comp_channel *comp_channel_;

        Connection connection_;

        template <bool, int> friend struct next_wc;

        boost::atomic<boost::uint16_t> executing_operation_;
        boost::atomic<bool> aborted_;
        boost::atomic<bool> close_operation_;

        util::spinlock mtx_;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_context<
        basic_context_service<context_impl<detail::client> >
    > client_context;

    typedef basic_context<
        basic_context_service<context_impl<detail::server> >
    > server_context;
}}}

#endif
