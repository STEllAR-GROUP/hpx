//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_DATA_WINDOWS_NOV_25_2012_0429PM)
#define HPX_PARCELSET_SHMEM_DATA_WINDOWS_NOV_25_2012_0429PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/shmem/interprocess_errors.hpp>
#include <hpx/runtime/parcelset/shmem/message.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/system/system_error.hpp>
#include <boost/thread/thread_time.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    ///////////////////////////////////////////////////////////////////////////
    //
    struct basic_data_window_options
    {
        struct msg_num
        {
            msg_num(std::size_t num) : val_(num) {}
            std::size_t val_;
        };
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
        void connect(std::string const& endpoint,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.connect(this->implementation, endpoint, ec);
        }

        template <typename Handler>
        void async_connect(std::string const& endpoint, Handler handler)
        {
            this->service.async_connect(this->implementation, endpoint, handler);
        }

        // options
        template <typename Opt>
        void set_option(Opt opt, boost::system::error_code &ec =
            boost::system::throws)
        {
            this->service.set_option(this->implementation, opt, ec);
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
                  std::string const& endpoint, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                endpoint_(endpoint),
                handler_(handler)
            {}

            void operator()() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    impl->connect(endpoint_, ec);
                    io_service_.post(
                        boost::asio::detail::bind_handler(handler_, ec));
                }
                else
                {
                    io_service_.post(
                        boost::asio::detail::bind_handler(handler_,
                            boost::asio::error::operation_aborted));
                }
            }

        private:
            boost::weak_ptr<Implementation> impl_;
            boost::asio::io_service &io_service_;
            std::string endpoint_;
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
            impl->open(true, ec);         // open only
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
        void connect(implementation_type &impl, std::string const& endpoint,
            boost::system::error_code &ec)
        {
            impl->connect(endpoint, ec);
        }

        template <typename Handler>
        void async_connect(implementation_type &impl,
            std::string const& endpoint, Handler handler)
        {
            this->get_io_service().post(
                detail::connect_operation<Handler, Implementation>(
                    impl, this->get_io_service(), endpoint, handler));
        }

        template <typename Opt>
        void set_option(implementation_type &impl, Opt opt,
            boost::system::error_code &ec)
        {
            impl->set_option(opt, ec);
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
          : msg_num_(1), aborted_(false), operation_active_(false)
        {}

        ~data_window_impl()
        {
            if (mq_) {
                mq_.reset();
                boost::interprocess::message_queue::remove(endpoint_.c_str());
            }
        }

        void open(bool open_queue_only, boost::system::error_code &ec)
        {
            if (mq_) {
                if (&ec != &boost::system::throws)
                    ec = boost::asio::error::already_connected;
                else
                    boost::asio::detail::throw_error(boost::asio::error::already_connected);
            }
            else if (!open_queue_only) {
                using namespace boost::interprocess;
                message_queue::remove(endpoint_.c_str());
                mq_ = boost::make_shared<message_queue>(
                    create_only, endpoint_.c_str(), msg_num_, sizeof(message));
            }
            else {
                using namespace boost::interprocess;
                mq_ = boost::make_shared<message_queue>(
                    open_only, endpoint_.c_str());
            }
        }

        void bind(std::string const& endpoint, boost::system::error_code &ec)
        {
            if (mq_) {
                if (&ec != &boost::system::throws)
                    ec = boost::asio::error::already_connected;
                else
                    boost::asio::detail::throw_error(boost::asio::error::already_connected);
            }
            else {
                endpoint_ = endpoint;
            }
        }

        void close(boost::system::error_code &ec)
        {
            if (mq_) {
                mq_.reset();
                boost::interprocess::message_queue::remove(endpoint_.c_str());
            }
            else {
                if (&ec != &boost::system::throws)
                    ec = boost::asio::error::not_connected;
                else
                    boost::asio::detail::throw_error(boost::asio::error::not_connected);
            }
        }

        void shutdown(boost::system::error_code &ec)
        {
            if (mq_) 
                send_command(message::shutdown, 0, ec);
        }

        void destroy()
        {
            // cancel operation
            aborted_ = true;
            while (operation_active_)
                ;
        }

        void connect(std::string const& endpoint, boost::system::error_code &ec)
        {
            if (mq_) {
                if (&ec != &boost::system::throws)
                    ec = boost::asio::error::already_connected;
                else
                    boost::asio::detail::throw_error(boost::asio::error::already_connected);
            }
            else {
                // set endpoint to connect to
                endpoint_ = endpoint;

                // open endpoint
                open(true, ec);
                if (!ec) {
                    operation_active_ = true;
                    try {
                        // send connect command
                        send_command(message::connect, endpoint.c_str(), ec);
                    }
                    catch (boost::interprocess::interprocess_exception const& e) {
                        if (&ec != &boost::system::throws)
                            ec = make_error_code(e.get_error_code());
                        else
                            boost::asio::detail::throw_error(make_error_code(e.get_error_code()));
                    }
                    operation_active_ = false;
                }
            }
        }

        // set options
        void set_option(basic_data_window_options::msg_num opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                if (&ec != &boost::system::throws)
                    ec = boost::asio::error::already_connected;
                else
                    boost::asio::detail::throw_error(boost::asio::error::already_connected);
            }
            else {
                msg_num_ = opt.val_;
            }
        }

    protected:
        void send_command(message::commands cmd, char const* data,
            boost::system::error_code &ec)
        {
            message msg;
            msg.command_ = cmd;
            if (data) {
                std::strncpy(msg.data_, data, sizeof(msg.data_));
                msg.data_[sizeof(msg.data_)-1] = '\0';
            }

            while(!mq_->timed_send(&msg, sizeof(msg), 0,
                boost::get_system_time() + boost::posix_time::milliseconds(1)))
            {
                if (aborted_) {
                    aborted_ = false;
                    if (&ec != &boost::system::throws)
                        ec = boost::asio::error::connection_aborted;
                    else
                        boost::asio::detail::throw_error(boost::asio::error::connection_aborted);
                    break;
                }
            }
        }

    private:
        std::size_t msg_num_;
        std::string endpoint_;
        boost::shared_ptr<boost::interprocess::message_queue> mq_;
        bool aborted_;
        bool operation_active_;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_data_window<
        basic_data_window_service<data_window_impl> 
    > data_window;
}}}

#endif


