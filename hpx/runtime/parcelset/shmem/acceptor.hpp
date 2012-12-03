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
#include <boost/scope_exit.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    ///////////////////////////////////////////////////////////////////////////
    //
    struct basic_acceptor_options
    {
        template <typename T>
        struct option
        {
            option(T const& num) : val_(num) {}
            T val_;
        };

        typedef option<std::size_t> msg_num;
        typedef option<bool> manage;
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
            this->service.bind(this->implementation, endpoint + ".acceptor", ec);
        }

        // synchronous and asynchronous accept
        template <typename Service_>
        void accept(basic_data_window<Service_>& window,
            boost::system::error_code &ec = boost::system::throws)
        {
            return this->service.accept(this->implementation, window, ec);
        }

        template <typename Service_, typename Handler>
        void async_accept(basic_data_window<Service_>& window, Handler handler)
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
                    while (!impl->try_accept(window_, ec) && !ec)
                        io_service_.poll_one();   // try to do other stuff

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
            while (!impl->try_accept(window, ec) && !ec)
                /* just wait for operation to succeed */;
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
          : msg_num_(1), aborted_(false),
            executing_operation_(false),
            close_operation_(false),
            manage_lifetime_(false)
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
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else if (!open_queue_only) {
                using namespace boost::interprocess;
                message_queue::remove(endpoint_.c_str());
                mq_ = boost::make_shared<message_queue>(
                    create_only, endpoint_.c_str(), msg_num_, sizeof(message));
                HPX_SHMEM_RESET_EC(ec);
            }
            else {
                using namespace boost::interprocess;
                mq_ = boost::make_shared<message_queue>(
                    open_only, endpoint_.c_str());
                HPX_SHMEM_RESET_EC(ec);
            }
        }

        void bind(std::string const& endpoint, boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                endpoint_ = endpoint;
                boost::interprocess::message_queue::remove(endpoint_.c_str());
                HPX_SHMEM_RESET_EC(ec);
            }
        }

        void close(boost::system::error_code &ec)
        {
            close_operation_ = true;
            BOOST_SCOPE_EXIT(&close_operation_) {
                close_operation_ = false;
            } BOOST_SCOPE_EXIT_END

            // wait for pending operations to return
            destroy();

            if (mq_) {
                mq_.reset();
                boost::interprocess::message_queue::remove(endpoint_.c_str());
                HPX_SHMEM_RESET_EC(ec);
            }
            else {
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::not_connected);
            }
        }

        void destroy()
        {
            aborted_ = true;
            BOOST_SCOPE_EXIT(&aborted_) {
                aborted_ = false;
            } BOOST_SCOPE_EXIT_END

            // cancel operation
            while (executing_operation_)
                ;
        }

        template <typename Service>
        bool try_accept(basic_data_window<Service>& window, boost::system::error_code &ec)
        {
            if (close_operation_ || !mq_) {
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                message msg;
                if (!try_receive_command(msg, ec))
                    return false;

                // verify that the received command was 'connect'
                if (!ec) {
                    if (msg.command_ == message::shutdown) {
                        close(ec);
                        HPX_SHMEM_THROWS_IF(ec, boost::asio::error::eof);
                    }
                    else if (msg.command_ != message::connect) {
                        HPX_SHMEM_THROWS_IF(ec, boost::asio::error::not_connected);
                    }
                    else {
                        // establish connection with given data window
                        window.set_option(data_window::manage(manage_lifetime_), ec);
                        if (!ec) {
                            window.bind(msg.data_, ec);
                            if (!ec) window.open(ec);
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        // set options
        void set_option(basic_acceptor_options::msg_num opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                msg_num_ = opt.val_;
                HPX_SHMEM_RESET_EC(ec);
            }
        }

        void set_option(basic_acceptor_options::manage opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_SHMEM_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                manage_lifetime_ = opt.val_;
                HPX_SHMEM_RESET_EC(ec);
            }
        }

    protected:
        bool try_receive_command(message& msg, boost::system::error_code &ec)
        {
            executing_operation_ = true;
            BOOST_SCOPE_EXIT(&executing_operation_) {
                executing_operation_ = false;
            } BOOST_SCOPE_EXIT_END

            try {
                HPX_SHMEM_RESET_EC(ec);

                boost::interprocess::message_queue::size_type recvd_size;
                unsigned int priority;
                if (!mq_->timed_receive(&msg, sizeof(msg), recvd_size, priority,
                    boost::get_system_time() + boost::posix_time::milliseconds(1)))
                {
                    if (aborted_) {
                        aborted_ = false;
                        HPX_SHMEM_THROWS_IF(ec, boost::asio::error::connection_aborted);
                    }
                    return false;
                }
            }
            catch (boost::interprocess::interprocess_exception const& e) {
                HPX_SHMEM_THROWS_IF(ec, make_error_code(e.get_error_code()));
            }
            return true;
        }

    private:
        std::size_t msg_num_;
        std::string endpoint_;
        boost::shared_ptr<boost::interprocess::message_queue> mq_;
        bool aborted_;
        bool executing_operation_;
        bool close_operation_;
        bool manage_lifetime_;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_acceptor<
        basic_acceptor_service<acceptor_impl>
    > acceptor;
}}}

#endif


