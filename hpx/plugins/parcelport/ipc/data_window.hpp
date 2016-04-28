//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IPC_DATA_WINDOWS_NOV_25_2012_0429PM)
#define HPX_PARCELSET_IPC_DATA_WINDOWS_NOV_25_2012_0429PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IPC)

#include <hpx/plugins/parcelport/ipc/interprocess_errors.hpp>
#include <hpx/plugins/parcelport/ipc/message.hpp>
#include <hpx/plugins/parcelport/ipc/data_buffer.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/system/system_error.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/scope_exit.hpp>
#include <boost/atomic.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>

#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    ///////////////////////////////////////////////////////////////////////////
    //
    struct basic_data_window_options
    {
        template <typename T>
        struct option
        {
            option(T const& num) : val_(num) {}
            T val_;
        };

        typedef option<std::size_t> msg_num;
        typedef option<std::string> bound_to;
        typedef option<bool> manage;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Service>
    class basic_data_window
      : public boost::asio::basic_io_object<Service>,
        public basic_data_window_options
    {
    public:
        explicit basic_data_window(boost::asio::io_service &io_service)
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

        void bind(std::string const& endpoint,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.bind(this->implementation, endpoint, ec);
        }

        // synchronous and asynchronous connect
        void connect(std::string const& there,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.connect(this->implementation, there, ec);
        }

        template <typename Handler>
        void async_connect(std::string const& there, Handler handler)
        {
            this->service.async_connect(this->implementation, there, handler);
        }

        // synchronous and asynchronous read/write/read_ack/write_ack
        void read(data_buffer& data,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.read(this->implementation, data, ec);
        }

        void write(data_buffer const& data,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.write(this->implementation, data, ec);
        }

        void read_ack(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.read_ack(this->implementation, ec);
        }

        void write_ack(boost::system::error_code &ec = boost::system::throws)
        {
            this->service.write_ack(this->implementation, ec);
        }

        template <typename Handler>
        void async_read(data_buffer& data, Handler handler)
        {
            this->service.async_read(this->implementation, data, handler);
        }

        template <typename Handler>
        void async_write(data_buffer const& data, Handler handler)
        {
            this->service.async_write(this->implementation, data, handler);
        }

        template <typename Handler>
        void async_read_ack(Handler handler)
        {
            this->service.async_read_ack(this->implementation, handler);
        }

        template <typename Handler>
        void async_write_ack(Handler handler)
        {
            this->service.async_write_ack(this->implementation, handler);
        }

        // options
        template <typename Opt>
        void set_option(Opt opt, boost::system::error_code &ec =
            boost::system::throws)
        {
            this->service.set_option(this->implementation, opt, ec);
        }

        template <typename Opt>
        Opt get_option(boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.get_option<Opt>(this->implementation, ec);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Handler, typename Implementation>
        class connect_operation
        {
            typedef std::shared_ptr<Implementation> implementation_type;

        public:
            connect_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  std::string const& there, Handler handler)
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
            std::string const& there_;
            Handler handler_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Handler, typename Implementation>
        class read_operation
          : public std::enable_shared_from_this<
                read_operation<Handler, Implementation> >
        {
            typedef std::shared_ptr<Implementation> implementation_type;

        public:
            read_operation(implementation_type &impl,
                  boost::asio::io_service &io_service,
                  data_buffer& data, Handler handler)
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

                    std::size_t size = 0;
                    if (0 == (size = impl->try_read(data_, ec)) && !ec) {
                        // repost this handler
                        io_service_.post(util::bind(
                            &read_operation::call, this->shared_from_this()));
                    }
                    else {
                        // successfully read next message
                        io_service_.post(boost::asio::detail::bind_handler(
                            handler_, ec, size));
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
            data_buffer& data_;
            Handler handler_;
        };

        template <typename Handler, typename Implementation>
        class read_ack_operation
        {
            typedef std::shared_ptr<Implementation> implementation_type;

        public:
            read_ack_operation(implementation_type &impl,
                  boost::asio::io_service &io_service, Handler handler)
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
                    if (!impl->try_read_ack(ec) && !ec) {
                        // repost this handler
                        io_service_.post(detail::read_ack_operation<
                            Handler, Implementation>(
                                impl, io_service_, handler_));
                    }
                    else {
                        // successfully read next message
                        io_service_.post(boost::asio::detail::bind_handler(
                            handler_, ec));
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
            typedef std::shared_ptr<Implementation> implementation_type;

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

        template <typename Handler, typename Implementation>
        class write_ack_operation
        {
            typedef std::shared_ptr<Implementation> implementation_type;

        public:
            write_ack_operation(implementation_type &impl,
                  boost::asio::io_service &io_service, Handler handler)
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
    class basic_data_window_service
      : public boost::asio::io_service::service
    {
    public:
        static boost::asio::io_service::id id;

        explicit basic_data_window_service(boost::asio::io_service &io_service)
          : boost::asio::io_service::service(io_service)
        {}

        ~basic_data_window_service()
        {}

        typedef std::shared_ptr<Implementation> implementation_type;

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

        void bind(implementation_type &impl, std::string const& endpoint,
            boost::system::error_code &ec)
        {
            impl->bind(endpoint, ec);
        }

        // synchronous and asynchronous connect
        void connect(implementation_type &impl, std::string const& there,
            boost::system::error_code &ec)
        {
            impl->connect(there, ec);
        }

        template <typename Handler>
        void async_connect(implementation_type &impl, std::string const& there,
            Handler handler)
        {
            this->get_io_service().post(
                detail::connect_operation<Handler, Implementation>(
                    impl, this->get_io_service(), there, handler));
        }

        // synchronous and asynchronous read/write/read_ack/write_ack
        std::size_t read(implementation_type &impl, data_buffer& data,
            boost::system::error_code &ec)
        {
            std::size_t size = 0;
            while (0 == (size = impl->try_read(data, ec)) && !ec)
                /* just wait for operation to succeed */;
            return size;
        }

        std::size_t write(implementation_type &impl, data_buffer const& data,
            boost::system::error_code &ec)
        {
            return impl->write(data, ec);
        }

        void read_ack(implementation_type &impl, boost::system::error_code &ec)
        {
            while (!impl->try_read_ack(ec) && !ec)
                /* just wait for operation to succeed */;
        }

        void write_ack(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->write_ack(ec);
        }

        template <typename Handler>
        void async_read(implementation_type &impl, data_buffer& data,
            Handler handler)
        {
            typedef detail::read_operation<Handler, Implementation> operation_type;

            std::shared_ptr<operation_type> op(
                std::make_shared<operation_type>(
                    impl, this->get_io_service(), data, handler));

            this->get_io_service().post(util::bind(&operation_type::call, op));
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
        void async_read_ack(implementation_type &impl, Handler handler)
        {
            this->get_io_service().post(
                detail::read_ack_operation<Handler, Implementation>(
                    impl, this->get_io_service(), handler));
        }

        template <typename Handler>
        void async_write_ack(implementation_type &impl, Handler handler)
        {
            this->get_io_service().post(
                detail::write_ack_operation<Handler, Implementation>(
                    impl, this->get_io_service(), handler));
        }

        //
        template <typename Opt>
        void set_option(implementation_type &impl, Opt opt,
            boost::system::error_code &ec)
        {
            impl->set_option(opt, ec);
        }

        template <typename Opt>
        Opt get_option(implementation_type &impl, boost::system::error_code &ec)
        {
            return impl->get_option<Opt>(ec);
        }

    private:
        void shutdown_service()
        {}
    };

    template <typename Implementation>
    boost::asio::io_service::id basic_data_window_service<Implementation>::id;

    ///////////////////////////////////////////////////////////////////////////
    class data_window_impl
    {
    public:
        data_window_impl()
          : msg_num_(10),
            executing_operation_(0),
            aborted_(false),
            close_operation_(false),
            manage_lifetime_(false)
        {}

        ~data_window_impl()
        {
            boost::system::error_code ec;
            shutdown(ec);
            close(ec);
        }

    protected:
        std::shared_ptr<boost::interprocess::message_queue>
        open_helper(bool open_queue_only, std::string const& name,
            boost::system::error_code &ec)
        {
            try {
                HPX_IPC_RESET_EC(ec);

                if (!open_queue_only) {
                    using namespace boost::interprocess;
                    if (manage_lifetime_)
                        message_queue::remove(name.c_str());

                    return std::make_shared<message_queue>(
                        open_or_create, name.c_str(), msg_num_, sizeof(message));
                }
                else {
                    using namespace boost::interprocess;
                    return std::make_shared<message_queue>(
                        open_only, name.c_str());
                }
            }
            catch(boost::interprocess::interprocess_exception const& e) {
                HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
            }
            return std::shared_ptr<boost::interprocess::message_queue>();
        }

        void open(bool open_only, std::string const& read_name,
            std::string const& write_name, boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                read_mq_ = open_helper(open_only, read_name, ec);
                if (!ec) {
                    write_mq_ = open_helper(open_only, write_name, ec);
                }
                else {
                    boost::system::error_code ec1;
                    close(ec1);
                }
            }
        }

    public:
        void open(boost::system::error_code &ec)
        {
            open(true, there_ + ".write", there_ + ".read", ec);
        }

        void bind(std::string const& endpoint, boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                there_ = endpoint;
                HPX_IPC_RESET_EC(ec);
            }
        }

        void close(boost::system::error_code &ec)
        {
            if (!read_mq_ && !write_mq_) {
                HPX_IPC_RESET_EC(ec);
                return;
            }

            close_operation_.store(true);
            BOOST_SCOPE_EXIT(&close_operation_) {
                close_operation_.store(false);
            } BOOST_SCOPE_EXIT_END

//             std::cout << "data_window: " << here_ << "/" << there_
//                       << ": close" << std::endl;

            // wait for pending operations to exit
            while (executing_operation_.load())
                ;

            // close does nothing if the data window was already closed
            if (read_mq_) {
                read_mq_.reset();
                if (manage_lifetime_) {
                    boost::interprocess::message_queue::remove(
                        (there_ + ".write").c_str());
                }
            }
            if (write_mq_) {
                write_mq_.reset();
                if (manage_lifetime_) {
                    boost::interprocess::message_queue::remove(
                        (there_ + ".read").c_str());
                }
            }
            HPX_IPC_RESET_EC(ec);
        }

        void shutdown(boost::system::error_code &ec)
        {
            if (write_mq_) {
//                 std::cout << "data_window: " << here_ <<  "/"
//                           << there_ << ": shutdown" << std::endl;
                send_command(*write_mq_, message::shutdown, 0, ec);
            }
        }

        void destroy()
        {
            aborted_.store(true);
            BOOST_SCOPE_EXIT(&aborted_) {
                aborted_.store(false);
            } BOOST_SCOPE_EXIT_END

//             std::cout << "data_window: " << here_ <<  "/" << there_
//                       << ": destroy" << std::endl;

            // cancel operation
            while (executing_operation_.load())
                ;
        }

        void connect(std::string const& there, boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                std::string portstr = there.substr(there.find_last_of("."));
                there_ = there + here_.substr(here_.find_last_of("."));

                // create/open endpoint
                std::string here(here_ + portstr);
                open(false, here + ".read", here + ".write", ec);
                if (!ec) {
                    // send connect command to corresponding acceptor
                    std::string acceptor_name(there + ".acceptor");
                    boost::interprocess::message_queue mq(
                        boost::interprocess::open_only, acceptor_name.c_str());
                    send_command(mq, message::connect, here.c_str(), ec);
                }
            }
        }

        // read, write, and acknowledge operations
        std::size_t try_read(data_buffer& data, boost::system::error_code &ec)
        {
            if (close_operation_.load() || !read_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                try {
                    message msg;
                    if (!try_receive_command(msg, ec))
                        return 0;

                    if (!ec) {
                        if (msg.command_ == message::shutdown) {
                            close(ec);
                            HPX_IPC_THROWS_IF(ec, boost::asio::error::eof);
                        }
                        else if (msg.command_ != message::data) {
                            HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
                        }
                        else {
                            // open the data_buffer as specified
                            data = data_buffer(msg.data_);
                            return data.size();
                        }
                    }
                }
                catch (boost::interprocess::interprocess_exception const& e) {
                    HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
                }
            }
            return 0;
        }

        std::size_t write(data_buffer const& data, boost::system::error_code &ec)
        {
            if (close_operation_.load() || !write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                send_command(*write_mq_, message::data, data.get_segment_name(), ec);
                if (!ec) return data.size();
            }
            return 0;
        }

        bool try_read_ack(boost::system::error_code &ec)
        {
            if (close_operation_.load() || !read_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                try {
                    message msg;
                    if (!try_receive_command(msg, ec))
                        return false;

                    if (!ec) {
                        if (msg.command_ == message::shutdown) {
                            close(ec);
                            HPX_IPC_THROWS_IF(ec, boost::asio::error::eof);
                        }
                        else if (msg.command_ != message::acknowledge) {
                            HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
                        }
                        else {
                            return true;
                        }
                    }
                }
                catch (boost::interprocess::interprocess_exception const& e) {
                    HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
                }
            }
            return false;
        }

        void write_ack(boost::system::error_code &ec)
        {
            if (close_operation_.load() || !write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                send_command(*write_mq_, message::acknowledge, "", ec);
            }
        }

        // set options
        void set_option(basic_data_window_options::msg_num const& opt,
            boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                msg_num_ = opt.val_;
                HPX_IPC_RESET_EC(ec);
            }
        }

        void set_option(basic_data_window_options::bound_to const& opt,
            boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                here_ = opt.val_;
                HPX_IPC_RESET_EC(ec);
            }
        }

        void set_option(basic_data_window_options::manage const& opt,
            boost::system::error_code &ec)
        {
            if (read_mq_ || write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                manage_lifetime_ = opt.val_;
                HPX_IPC_RESET_EC(ec);
            }
        }

        basic_data_window_options::bound_to get_option(
            boost::system::error_code &ec)
        {
            if (close_operation_.load() || !read_mq_ || !write_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
                return basic_data_window_options::bound_to("");
            }

            HPX_IPC_RESET_EC(ec);
            return basic_data_window_options::bound_to(here_);
        }

    protected:
        bool try_receive_command(message& msg, boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END

            if (close_operation_.load() || !read_mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
                return false;
            }

            if (aborted_.load()) {
                aborted_.store(false);
                HPX_IPC_THROWS_IF(ec, boost::asio::error::connection_aborted);
                return false;
            }

            HPX_IPC_RESET_EC(ec);

            try {
                boost::interprocess::message_queue::size_type recvd_size;
                unsigned int priority;
                return read_mq_->try_receive(&msg, sizeof(msg), recvd_size, priority);
            }
            catch (boost::interprocess::interprocess_exception const& e) {
                aborted_.store(false);
                HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
            }
            return false;
        }

        void send_command(boost::interprocess::message_queue& mq,
            message::commands cmd, char const* data,
            boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END

            if (close_operation_.load()) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
                return;
            }

            HPX_IPC_RESET_EC(ec);

            try {
                message msg;
                msg.command_ = cmd;
                if (data) {
                    std::strncpy(msg.data_, data, sizeof(msg.data_));
                    msg.data_[sizeof(msg.data_)-1] = '\0';
                }

                while (!mq.timed_send(&msg, sizeof(msg), 0,
                    boost::get_system_time() + boost::posix_time::milliseconds(1)))
                {
                    if (aborted_.load() || close_operation_.load()) {
                        aborted_.store(false);
                        HPX_IPC_THROWS_IF(ec, boost::asio::error::connection_aborted);
                        break;
                    }
                }
            }
            catch (boost::interprocess::interprocess_exception const& e) {
                aborted_.store(false);
                HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
            }
        }

    private:
        std::size_t msg_num_;
        std::string there_;
        std::shared_ptr<boost::interprocess::message_queue> read_mq_;
        std::string here_;
        std::shared_ptr<boost::interprocess::message_queue> write_mq_;
        boost::atomic<boost::uint16_t> executing_operation_;
        boost::atomic<bool> aborted_;
        boost::atomic<bool> close_operation_;
        bool manage_lifetime_;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_data_window<
        basic_data_window_service<data_window_impl>
    > data_window;
}}}}

#endif

#endif
