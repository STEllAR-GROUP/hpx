//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_ACCEPTOR_NOV_25_2012_0710PM)
#define HPX_PARCELSET_SHMEM_ACCEPTOR_NOV_25_2012_0710PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/shmem/interprocess_errors.hpp>
#include <hpx/runtime/parcelset/shmem/message.hpp>
#include <hpx/runtime/parcelset/shmem/data_window.hpp>
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
    struct basic_acceptor_options
    {
        struct msg_num
        {
            msg_num(std::size_t num) : val_(num) {}
            std::size_t val_;
        };
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

        void bind(std::string const& endpoint,
            boost::system::error_code &ec = boost::system::throws)
        {
            this->service.bind(this->implementation, endpoint, ec);
        }

        // synchronous and asynchronous accept
        template <typename Service>
        void accept(basic_data_window<Service>& window,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.accept(this->implementation, window, ec);
        }

        template <typename Service, typename Handler>
        void async_accept(basic_data_window<Service>& window, Handler handler)
        {
            this->service.async_accept(this->implementation, window, handler);
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
        ///////////////////////////////////////////////////////////////////////
        template <typename Service, typename Handler, typename Implementation>
        class accept_operation
        {
            typedef boost::shared_ptr<Implementation> implementation_type;

        public:
            accept_operation(implementation_type &impl,
                  boost::asio::io_service &io_service, 
                  basic_data_window<Service>& window, Handler handler)
              : impl_(impl),
                io_service_(io_service),
                window_(window),
                handler_(handler)
            {}

            void operator()() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    impl->accept(window_, ec);
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
            basic_data_window<Service>& window_;
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
            impl->open(false, ec);      // create only
            boost::asio::detail::throw_error(ec);
        }

        void close(implementation_type &impl, boost::system::error_code &ec)
        {
            impl->close(ec);
            boost::asio::detail::throw_error(ec);
        }

        void bind(implementation_type &impl, std::string const& endpoint,
            boost::system::error_code &ec)
        {
            impl->bind(endpoint, ec);
            boost::asio::detail::throw_error(ec);
        }

        // synchronous and asynchronous accept
        template <typename Service>
        void accept(implementation_type &impl, 
            basic_data_window<Service>& window, boost::system::error_code &ec)
        {
            impl->accept(window, ec);
            boost::asio::detail::throw_error(ec);
        }

        template <typename Service, typename Handler>
        void async_accept(implementation_type &impl, 
            basic_data_window<Service>& window, Handler handler)
        {
            this->get_io_service().post(
                detail::accept_operation<Service, Handler, Implementation>(
                    impl, this->get_io_service(), window, handler));
        }

        template <typename Opt>
        void set_option(implementation_type &impl, Opt opt,
            boost::system::error_code &ec)
        {
            impl->set_option(opt, ec);
            boost::asio::detail::throw_error(ec);
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
          : msg_num_(1), aborted_(false), operation_active_(false)
        {}

        ~acceptor_impl()
        {
            if (mq_) {
                mq_.reset();
                boost::interprocess::message_queue::remove(endpoint_.c_str());
            }
        }

        void open(bool open_queue_only, boost::system::error_code &ec)
        {
            if (mq_) {
                ec = boost::asio::error::already_connected;
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
                ec = boost::asio::error::already_connected;
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
                ec = boost::asio::error::not_connected;
            }
        }

        void destroy()
        {
            // cancel operation
            aborted_ = true;
            while (operation_active_)
                ;
        }

        template <typename Service>
        void accept(basic_data_window<Service>& window, boost::system::error_code &ec)
        {
            if (!mq_) {
                ec = boost::asio::error::not_connected;
            }
            else {
                operation_active_ = true;
                try {
                    message msg;
                    boost::interprocess::message_queue::size_type recvd_size;
                    unsigned int priority;
                    while(!mq_->timed_receive(&msg, sizeof(msg), recvd_size, priority,
                        boost::get_system_time() + boost::posix_time::milliseconds(1)))
                    {
                        if (aborted_) {
                            ec = boost::asio::error::connection_aborted;
                            break;
                        }
                    }

                    // verify that the received command was 'connect'
                    if (!aborted_) {
                        if (msg.command_ != message::connect) {
                            ec = boost::asio::error::not_connected;
                        }
                        else {
                            // establish connection with given data window
                            window.bind(msg.data_, ec);
                            if (!ec)
                                window.open(ec);
                        }
                    }
                    else {
                        aborted_ = false;
                    }
                }
                catch (boost::interprocess::interprocess_exception const& e) {
                    ec = make_error_code(e.get_error_code());
                }
                operation_active_ = false;
            }
        }

        // set options
        void set_option(basic_acceptor_options::msg_num opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                ec = boost::asio::error::already_connected;
            }
            else {
                msg_num_ = opt.val_;
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
    typedef basic_acceptor<
        basic_acceptor_service<acceptor_impl> 
    > acceptor;
}}}

#endif


