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

        void set_buffer_size(std::size_t buffer_size, boost::system::error_code &ec)
        {
            this->service.set_buffer_size(this->implementation, buffer_size, ec);
        }
        
        void build_connection(rdma_cm_id * id, boost::system::error_code &ec)
        {
            this->service.build_connection(this->implementation, id, ec);
        }

        rdma_cm_id *conn_id()
        {
            return this->service.conn_id(this->implementation);
        }

        void on_preconnect(rdma_cm_id * id)
        {
            this->service.on_preconnect(this->implementation, id);
        }

        void on_connection(rdma_cm_id * id)
        {
            this->service.on_connection(this->implementation, id);
        }

        void on_completion(ibv_wc * wc)
        {
            this->service.on_completion(this->implementation, wc);
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

        std::size_t write(std::vector<char> const& data,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.write(this->implementation, data, ec);
        }

        template <typename Handler>
        void async_read(std::vector<char>& data, Handler handler)
        {
            this->service.async_read(this->implementation, data, handler);
        }

        template <typename Handler>
        void async_write(std::vector<char> const& data, Handler handler)
        {
            this->service.async_write(this->implementation, data, handler);
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
                  std::vector<char>& data, Handler handler)
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
                        /*
                        // post reading of size
                        io_service_.post(boost::asio::detail::bind_handler(
                            handler_, ec, data_.size()));
                        */
                        handler_(ec, data_.size());
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
            std::vector<char>& data_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class write_operation
          : public boost::enable_shared_from_this<
                write_operation<Handler, Implementation> >
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            write_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  std::vector<char> const& data, Handler handler)
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
                    std::size_t size = impl->write(data_, ec);
                    /*
                    io_service_.post(boost::asio::detail::bind_handler(
                        handler_, ec, size));
                    */
                    handler_(ec, size);
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
            std::vector<char> const& data_;
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

        void set_buffer_size(implementation_type &impl, std::size_t buffer_size, boost::system::error_code &ec)
        {
            impl->set_buffer_size(buffer_size, ec);
        }
        
        void build_connection(implementation_type &impl, rdma_cm_id * id, boost::system::error_code &ec)
        {
            impl->build_connection(id, ec);
        }

        rdma_cm_id *conn_id(implementation_type &impl)
        {
            return impl->conn_id();
        }

        void on_preconnect(implementation_type &impl, rdma_cm_id * id)
        {
            return impl->on_preconnect(id);
        }

        void on_connection(implementation_type &impl, rdma_cm_id * id)
        {
            return impl->on_connection(id);
        }

        void on_completion(implementation_type &impl, ibv_wc * wc)
        {
            return impl->on_completion(wc);
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

        // synchronous and asynchronous read/write/read_ack/write_ack
        std::size_t read(implementation_type &impl, std::vector<char>& data, 
            boost::system::error_code &ec)
        {
#if 0
            std::size_t size = 0;
        bool try_read_ready(boost::system::error_code &ec)

        std::size_t try_read_size(boost::system::error_code &ec)

        // read, write, and acknowledge operations
        std::size_t try_read_data(std::vector<char>& data, boost::system::error_code &ec)

        bool try_read_done(boost::system::error_code &ec)
            while (0 == (size = impl->try_read(data, ec)) && !ec)
                /* just wait for operation to succeed */;
            return size;
            */
#endif
            BOOST_ASSERT(false);
            return 0;
        }

        std::size_t write(implementation_type &impl, std::vector<char> const& data, 
            boost::system::error_code &ec)
        {
            return impl->write(data, ec);
        }

        template <typename Handler>
        void async_read(implementation_type &impl, std::vector<char>& data,
            Handler handler)
        {
            typedef detail::read_operation<Handler, Implementation> operation_type;

            BOOST_ASSERT(hpx::threads::get_self_id());

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), data, handler));

            this->get_io_service().post(boost::bind(
                &operation_type::call, op));
        }

        template <typename Handler>
        void async_write(implementation_type &impl, std::vector<char> const& data,
            Handler handler)
        {
            typedef detail::write_operation<Handler, Implementation> operation_type;
            
            BOOST_ASSERT(hpx::threads::get_self_id());

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), data, handler));

            this->get_io_service().post(boost::bind(
                &operation_type::call, op));
        }
        

    private:
        void shutdown_service()
        {}
    };

    template <typename Implementation>
    boost::asio::io_service::id basic_context_service<Implementation>::id;

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

        void on_preconnect(rdma_cm_id * id)
        {
            connection_.on_preconnect(id, pd_);
        }

        void on_connection(rdma_cm_id * id)
        {
            connection_.on_connection(id);
        }

        void on_completion(ibv_wc * wc)
        {
            connection_.on_completion(wc);
        }

        void on_disconnect(rdma_cm_id * id)
        {
            connection_.on_disconnect(id);
        }

        void set_buffer_size(std::size_t buffer_size, boost::system::error_code &ec)
        {
            connection_.set_buffer_size(buffer_size, ec);
        }

        void open(boost::system::error_code &ec)
        {
        }

        void bind(
            boost::asio::ip::tcp::endpoint const & ep, boost::system::error_code &ec)
        {
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
        }

        void destroy()
        {
            aborted_.store(true);
            BOOST_SCOPE_EXIT_TPL(&aborted_) {
                aborted_.store(false);
            } BOOST_SCOPE_EXIT_END
            
            // cancel operation
            while (executing_operation_.load())
                ;
        }

        void connect(boost::asio::ip::tcp::endpoint const & there, boost::system::error_code &ec)
        {
            if(ctx_)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::already_connected);
            }

            addrinfo *addr;
            rdma_conn_param cm_params;

            int ret = 0;

            ret = getaddrinfo(
                there.address().to_string().c_str()
              , boost::lexical_cast<std::string>(there.port()).c_str()
              , NULL
              , &addr
            );

            if(ret)
            {
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
            }

            event_channel_ = rdma_create_event_channel();
            if(!event_channel_)
            {
                // FIXME: better error here
                freeaddrinfo(addr);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
            }
            set_nonblocking(event_channel_->fd);

            ret = rdma_create_id(event_channel_, &conn_, NULL, RDMA_PS_TCP);
            if(ret)
            {
                // FIXME: better error here
                freeaddrinfo(addr);
                rdma_destroy_event_channel(event_channel_);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
            }
            const int TIMEOUT_IN_MS = 500;
            ret = rdma_resolve_addr(conn_, NULL, addr->ai_addr, TIMEOUT_IN_MS);
            if(ret)
            {
                // FIXME: better error here
                freeaddrinfo(addr);
                rdma_destroy_id(conn_);
                rdma_destroy_event_channel(event_channel_);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
            }

            int * i = new int(8);

            conn_->context = i;

            freeaddrinfo(addr);

            // build params ...
            std::memset(&cm_params, 0, sizeof(rdma_conn_param));
            cm_params.initiator_depth = cm_params.responder_resources = 1;
            cm_params.rnr_retry_count = 7; /* infinite retry */

            rdma_cm_event event;
            std::size_t k = 0;
            while(!get_next_event(event_channel_, event, this))
            {
                hpx::util::detail::yield_k(k, "hpx::parcelset::ibverbs::context::connect");
                ++k;
            }
            if(event.event == RDMA_CM_EVENT_ADDR_RESOLVED)
            {
                // building connection ...
                build_connection(event.id, ec);


                connection_.on_preconnect(event.id, pd_);

                ret = rdma_resolve_route(event.id, TIMEOUT_IN_MS);
                if(ret)
                {
                    // FIXME: better error here
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                }
                k = 0;
                while(!get_next_event(event_channel_, event, this))
                {
                    hpx::util::detail::yield_k(k, "hpx::parcelset::ibverbs::context::connect");
                    ++k;
                }
                if(event.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
                {
                    ret = rdma_connect(event.id, &cm_params);
                    k = 0;
                    while(!get_next_event(event_channel_, event, this))
                    {
                        hpx::util::detail::yield_k(k, "hpx::parcelset::ibverbs::context::connect");
                        ++k;
                    }
                    if(event.event == RDMA_CM_EVENT_ESTABLISHED)
                    {
                        connection_.on_connect(event.id);
                        message_type m = next_wc(ec, true);
                        if(ec) return;
                        if(m != MSG_MR)
                        {
                            HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                        }
                        HPX_IBVERBS_RESET_EC(ec);
                        return;
                    }
                }
            }
            else
            {
                rdma_destroy_id(conn_);
                rdma_destroy_event_channel(event_channel_);
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
            }
        }

        message_type next_wc(boost::system::error_code &ec, bool retry)
        {
            pollfd pfd;
            pfd.fd = comp_channel_->fd;
            pfd.events = POLLIN;
            pfd.revents = 0;

            int ret = 0;
            std::size_t k = 0;
            do
            {
                ret = poll(&pfd, 1, 1);
                if(ret == 0 && !retry) return MSG_RETRY;
                hpx::util::detail::yield_k(k, "hpx::parcelset::ibverbs::context::next_wc");
                k++;
            } while (ret == 0);
            if(ret < 0)
            {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                return MSG_INVALID;
            }

            HPX_IBVERBS_RESET_EC(ec);

            ibv_cq * cq;
            ibv_wc wc;
            void *dummy = NULL;
            ret = ibv_get_cq_event(comp_channel_, &cq, &dummy);
            ibv_ack_cq_events(cq, 1);
            if(ret == -1)
            {
            }
            ibv_req_notify_cq(cq, 0);
            

            k = 0;
            message_type m = MSG_RETRY;
            while(m == MSG_RETRY)
            {
                if(ibv_poll_cq(cq, 1, &wc))
                {
                    if(wc.status == IBV_WC_SUCCESS)
                    {
                        m = connection_.on_completion(&wc);
                    }
                    else
                    {
                        aborted_.store(false);
                        HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::connection_aborted);
                        return MSG_INVALID;
                    }
                }
                // As this loop might spin quite a while, we are implementing
                // an exponential backup strategy
                if(m == MSG_RETRY && !retry) return MSG_RETRY;
                hpx::util::detail::yield_k(k, "hpx::parcelset::ibverbs::context::next_wc");
                ++k;
            }
            return m;
        }
        
        bool is_closed(boost::system::error_code &ec)
        {
            if (close_operation_.load() || !ctx_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return true;
            }
            return false;
        }

        // read, write, and acknowledge operations
        std::size_t try_read_data(std::vector<char>& data, boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT_TPL(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END
            if(is_closed(ec)) return 0;
            
            connection_.post_receive();

            //std::cout << "try read data ...\n";

            HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 0, false);
            //BOOST_ASSERT(connection_.size_ > 0);

            boost::uint64_t total_len = 0;
            std::memcpy(&total_len, connection_.buffer_, sizeof(boost::uint64_t));

            boost::uint64_t chunk_size = 0;
            std::memcpy(&chunk_size, connection_.buffer_ + sizeof(boost::uint64_t), sizeof(boost::uint64_t));

            //std::cout << "got len: " << total_len << " " << chunk_size << "\n";

            data.resize(total_len);

            //std::vector<char>::iterator data_itr = data.begin();
            char * data_ptr = &data[0];

            std::memcpy(
                data_ptr
              , connection_.buffer_ + 2 * sizeof(boost::uint64_t)
              , chunk_size
            );

            if(chunk_size == total_len)
            {
                //std::cout << "reading finished ... sending done ...\n";
                connection_.msg_->id = MSG_DONE;
                connection_.send_message();
            }
            else
            {
                connection_.post_receive();
                connection_.msg_->id = MSG_DATA;
                connection_.send_message();

                data_ptr += chunk_size;
                for(std::size_t bytes_read = chunk_size; bytes_read < total_len;)
                {
                    //std::cout << "get remaining chunks ...\n";
                    HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 0, true);
                    chunk_size = 0;
                    std::memcpy(&chunk_size, connection_.buffer_, sizeof(boost::uint64_t));
                    //std::cout << "get remaining chunks len: " << total_len << " " << chunk_size << "\n";

                    std::memcpy(
                        data_ptr
                      , connection_.buffer_ + sizeof(boost::uint64_t)
                      , chunk_size
                    );
                    data_ptr += chunk_size;
                    bytes_read += chunk_size;

                    if(bytes_read >= data.size())
                    {
                        //std::cout << "finished ... reading done ...\n";
                        connection_.msg_->id = MSG_DONE;
                        connection_.send_message();
                    }
                    else
                    {
                        connection_.post_receive();
                        connection_.msg_->id = MSG_DATA;
                        connection_.send_message();
                    }
                }
            }

            HPX_IBVERBS_RESET_EC(ec);
            return data.size();
        }

        std::size_t write(std::vector<char> const& data, boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT_TPL(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END

            if (close_operation_.load() || !ctx_) {
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::not_connected);
                return 0;
            }

            boost::uint64_t total_len = data.size();

            std::memcpy(connection_.buffer_, &total_len, sizeof(total_len));

            boost::uint64_t bytes_sent = 0;
            boost::uint64_t chunk_size = std::min(connection_.buffer_size_ - 2 * sizeof(boost::uint64_t), total_len);
            
            std::memcpy(connection_.buffer_ + sizeof(boost::uint64_t), &chunk_size, sizeof(boost::uint64_t));

            const char * data_ptr = &data[0];
            
            std::memcpy(connection_.buffer_ + 2 * sizeof(boost::uint64_t), data_ptr, chunk_size);

            connection_.write_remote(chunk_size + 2 * sizeof(boost::uint64_t));
            connection_.post_receive();

            //std::cout << "send len: " << total_len << "(" << chunk_size << ") " << sizeof(total_len) << "\n";

            bytes_sent += chunk_size;
            data_ptr += chunk_size;

            while(bytes_sent < total_len)
            {
                //std::cout << "waiting for ready from server ...\n";
                HPX_IBVERBS_NEXT_WC(ec, MSG_DATA, 0, true);
                chunk_size = std::min(connection_.buffer_size_ - sizeof(boost::uint64_t), total_len - bytes_sent);
                std::memcpy(connection_.buffer_, &chunk_size, sizeof(boost::uint64_t));
                
                //std::cout << "sending len: " << bytes_sent << " (" << chunk_size << ") " << sizeof(total_len) << "\n";

                std::memcpy(connection_.buffer_ + sizeof(boost::uint64_t), data_ptr, chunk_size);
                
                connection_.write_remote(chunk_size + sizeof(boost::uint64_t));
                connection_.post_receive();
                //std::cout << "sent next chunk ...\n";

                bytes_sent += chunk_size;
                data_ptr += chunk_size;
            }

            //std::cout << "waiting for done from server ...\n";
            HPX_IBVERBS_NEXT_WC(ec, MSG_DONE, 0, true);
            //std::cout << "writing finished ... got done\n";
            HPX_IBVERBS_RESET_EC(ec);
            return total_len;
        }

        void build_connection(rdma_cm_id * id, boost::system::error_code &ec)
        {
            ibv_qp_init_attr qp_attr;
            build_context(id->verbs, ec);
            build_qp_attr(&qp_attr);

            int ret = rdma_create_qp(id, pd_, &qp_attr);
            if(ret)
            {
                close(ec);
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
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
                    close(ec);
                    // FIXME: better error here
                    HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                    return;
                }
            }

            ctx_ = verbs;

            pd_ = ibv_alloc_pd(ctx_);
            if(!pd_)
            {
                close(ec);
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                return;
            }
            comp_channel_ = ibv_create_comp_channel(ctx_);
            if(!comp_channel_)
            {
                close(ec);
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                return;
            }
            set_nonblocking(comp_channel_->fd);
            cq_ = ibv_create_cq(ctx_, 10, NULL, comp_channel_, 0);
            if(!cq_)
            {
                close(ec);
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
                return;
            }

            int ret = ibv_req_notify_cq(cq_, 0);
            if(ret)
            {
                close(ec);
                // FIXME: better error here
                HPX_IBVERBS_THROWS_IF(ec, boost::asio::error::fault);
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
            qp_attr->cap.max_send_sge = 1;
            qp_attr->cap.max_recv_sge = 1;
        }

        rdma_event_channel *event_channel_;
        rdma_cm_id *conn_;

        ibv_context *ctx_;
        ibv_pd * pd_;
        ibv_cq * cq_;
        ibv_comp_channel *comp_channel_;

        Connection connection_;
        
        boost::atomic<boost::uint16_t> executing_operation_;
        boost::atomic<bool> aborted_;
        boost::atomic<bool> close_operation_;
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
