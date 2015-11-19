//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IPC_ACCEPTOR_NOV_25_2012_0710PM)
#define HPX_PARCELSET_IPC_ACCEPTOR_NOV_25_2012_0710PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/parcelport/ipc/interprocess_errors.hpp>
#include <hpx/plugins/parcelport/ipc/message.hpp>
#include <hpx/plugins/parcelport/ipc/data_window.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/asio/basic_io_object.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/system/system_error.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/scope_exit.hpp>
#include <boost/atomic.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ipc
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
          : public boost::enable_shared_from_this<
                accept_operation<Service, Handler, Implementation> >
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

            void call() const
            {
                implementation_type impl = impl_.lock();
                if (impl)
                {
                    boost::system::error_code ec;
                    if (!impl->try_accept(window_, ec) && !ec) {
                        // repost this function
                        io_service_.post(boost::bind(
                            &accept_operation::call, this->shared_from_this()));
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
            typedef detail::accept_operation<Service, Handler, Implementation>
                operation_type;

            boost::shared_ptr<operation_type> op(
                boost::make_shared<operation_type>(
                    impl, this->get_io_service(), window, handler));

            this->get_io_service().post(boost::bind(&operation_type::call, op));
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
          : msg_num_(1),
            executing_operation_(false),
            aborted_(false),
            close_operation_(false),
            manage_lifetime_(false)
        {}

        ~acceptor_impl()
        {
            boost::system::error_code ec;
            close(ec);
        }

        void open(bool open_queue_only, boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else if (!open_queue_only) {
                using namespace boost::interprocess;
                message_queue::remove(endpoint_.c_str());
                mq_ = boost::make_shared<message_queue>(
                    create_only, endpoint_.c_str(), msg_num_, sizeof(message));
                HPX_IPC_RESET_EC(ec);
            }
            else {
                using namespace boost::interprocess;
                mq_ = boost::make_shared<message_queue>(
                    open_only, endpoint_.c_str());
                HPX_IPC_RESET_EC(ec);
            }
        }

        void bind(std::string const& endpoint, boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                endpoint_ = endpoint;
                boost::interprocess::message_queue::remove(endpoint_.c_str());
                HPX_IPC_RESET_EC(ec);
            }
        }

        void close(boost::system::error_code &ec)
        {
            if (!mq_) {
                HPX_IPC_RESET_EC(ec);
                return;
            }

            close_operation_.store(true);
            BOOST_SCOPE_EXIT(&close_operation_) {
                close_operation_.store(false);
            } BOOST_SCOPE_EXIT_END

//             std::cout << "acceptor: " << endpoint_ << ": close" << std::endl;

            // wait for pending operations to return
            while (executing_operation_.load())
                ;

            if (mq_) {
                mq_.reset();
                boost::interprocess::message_queue::remove(endpoint_.c_str());
                HPX_IPC_RESET_EC(ec);
            }
            else {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
        }

        void destroy()
        {
            aborted_.store(true);
            BOOST_SCOPE_EXIT(&aborted_) {
                aborted_.store(false);
            } BOOST_SCOPE_EXIT_END

//             std::cout << "acceptor: " << endpoint_ << ": destroy" << std::endl;

            // cancel operation
            while (executing_operation_.load())
                ;
        }

        template <typename Service>
        bool try_accept(basic_data_window<Service>& window,
            boost::system::error_code &ec)
        {
            if (close_operation_.load() || !mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
            }
            else {
                message msg;
                if (!try_receive_command(msg, ec))
                    return false;

                // verify that the received command was 'connect'
                if (!ec) {
                    if (msg.command_ == message::shutdown) {
                        close(ec);
                        HPX_IPC_THROWS_IF(ec, boost::asio::error::eof);
                    }
                    else if (msg.command_ != message::connect) {
                        HPX_IPC_THROWS_IF(ec, boost::asio::error::not_connected);
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
        void set_option(basic_acceptor_options::msg_num const& opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                msg_num_ = opt.val_;
                HPX_IPC_RESET_EC(ec);
            }
        }

        void set_option(basic_acceptor_options::manage const& opt,
            boost::system::error_code &ec)
        {
            if (mq_) {
                HPX_IPC_THROWS_IF(ec, boost::asio::error::already_connected);
            }
            else {
                manage_lifetime_ = opt.val_;
                HPX_IPC_RESET_EC(ec);
            }
        }

    protected:
        bool try_receive_command(message& msg, boost::system::error_code &ec)
        {
            ++executing_operation_;
            BOOST_SCOPE_EXIT(&executing_operation_) {
                --executing_operation_;
            } BOOST_SCOPE_EXIT_END

            if (close_operation_.load() || !mq_) {
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
                return mq_->try_receive(&msg, sizeof(msg), recvd_size, priority);
            }
            catch (boost::interprocess::interprocess_exception const& e) {
                aborted_.store(false);
                HPX_IPC_THROWS_IF(ec, make_error_code(e.get_error_code()));
            }
            return false;
        }

    private:
        std::size_t msg_num_;
        std::string endpoint_;
        boost::shared_ptr<boost::interprocess::message_queue> mq_;
        boost::atomic<boost::uint16_t> executing_operation_;
        boost::atomic<bool> aborted_;
        boost::atomic<bool> close_operation_;
        bool manage_lifetime_;
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef basic_acceptor<
        basic_acceptor_service<acceptor_impl>
    > acceptor;
}}}}

#endif


