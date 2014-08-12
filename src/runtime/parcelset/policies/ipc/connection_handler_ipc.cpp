//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_PARCELPORT_IPC)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/ipc/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ipc/acceptor.hpp>
#include <hpx/runtime/parcelset/policies/ipc/sender.hpp>
#include <hpx/runtime/parcelset/policies/ipc/receiver.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/asio/placeholders.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines +=
            "data_buffer_cache_size = ${HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE:512}",
            "zero_copy_optimization = 0",
            "async_serialization = 0"
            ;

        return lines;
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , acceptor_(0)
      , connection_count_(0)
      , data_buffer_cache_(ini.get_ipc_data_buffer_cache_size())
    {
        // we never do zero copy optimization for this parcelport
        allow_zero_copy_optimizations_ = false;
    }

    connection_handler::~connection_handler()
    {
        if (NULL != acceptor_) {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
            acceptor_ = 0;
        }
    }

    bool connection_handler::do_run()
    {
        if (NULL == acceptor_)
            acceptor_ = new acceptor(io_service_pool_.get_io_service(0));

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it =
                accept_begin(here_, io_service_pool_.get_io_service(0));
             it != end; ++it, ++tried)
        {
            try {
                boost::shared_ptr<receiver> conn(
                    new receiver(
                        io_service_pool_.get_io_service(), here(), *this));

                boost::asio::ip::tcp::endpoint ep = *it;

                std::string fullname(ep.address().to_string() + "." +
                    boost::lexical_cast<std::string>(ep.port()));

                acceptor_->set_option(acceptor::msg_num(10));
                acceptor_->set_option(acceptor::manage(true));
                acceptor_->bind(fullname);
                acceptor_->open();

                acceptor_->async_accept(conn->window(),
                    boost::bind(&connection_handler::handle_accept,
                        this,
                        boost::asio::placeholders::error, conn));
            }
            catch (boost::system::system_error const&) {
                errors.add(boost::current_exception());
                continue;
            }
        }

        if (errors.size() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "ipc::connection_handler::run", errors.get_message());
            return false;
        }
        return true;
    }

    void connection_handler::do_stop()
    {
        {
            // cancel all pending read operations, close those sockets
            lcos::local::spinlock::scoped_lock l(mtx_);
            BOOST_FOREACH(boost::shared_ptr<receiver> c, accepted_connections_)
            {
                boost::system::error_code ec;
                data_window& w = c->window();
                w.shutdown(ec); // shut down connection
                w.close(ec);    // close the data window to give it back to the OS
            }
            accepted_connections_.clear();
        }

        data_buffer_cache_.clear();

        // cancel all pending accept operations
        if (NULL != acceptor_)
        {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
            acceptor_ = NULL;
        }
    }

    std::string connection_handler::get_locality_name() const
    {
        return "ipc";
    }

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        boost::shared_ptr<sender> sender_connection(new sender(io_service,
                here_, l, data_buffer_cache_, parcels_sent_, ++connection_count_));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try {
                naming::locality::iterator_type end = connect_end(l);
                for (naming::locality::iterator_type it =
                        connect_begin(l, io_service);
                      it != end; ++it)
                {
                    boost::asio::ip::tcp::endpoint const& ep = *it;
                    std::string fullname(ep.address().to_string() + "." +
                        boost::lexical_cast<std::string>(ep.port()));

                    data_window& w = sender_connection->window();
                    w.close();
                    w.connect(fullname, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                this_thread::suspend(hpx::threads::pending,
                    "connection_handler(ipc)::create_connection");
            }
            catch (boost::system::system_error const& e) {
                sender_connection->window().close();
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "ipc::parcelport::get_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->window().close();
            sender_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";
            HPX_THROWS_IF(ec, network_error,
                "ipc::parcelport::get_connection",
                hpx::util::osstream_get_string(strm));
            return sender_connection;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
    }

    // accepted new incoming connection
    void connection_handler::handle_accept(boost::system::error_code const & e,
        boost::shared_ptr<receiver> receiver_conn)
    {
        if (!e) {
            // handle this incoming parcel
            boost::shared_ptr<receiver> c(receiver_conn); // hold on to receiver_conn

            // create new connection waiting for next incoming parcel
            receiver_conn.reset(new receiver(
                io_service_pool_.get_io_service(), here(), *this));

            acceptor_->async_accept(receiver_conn->window(),
                boost::bind(&connection_handler::handle_accept,
                    this,
                    boost::asio::placeholders::error, receiver_conn));

            {
                // keep track of all the accepted connections
                lcos::local::spinlock::scoped_lock l(mtx_);
                accepted_connections_.insert(c);
            }

            // now accept the incoming connection by starting to read from the
            // data window
            c->async_read(boost::bind(&connection_handler::handle_read_completion,
                this, boost::asio::placeholders::error, c));
        }
        else {
            // remove this connection from the list of known connections
            lcos::local::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(receiver_conn);
        }
    }

    // Handle completion of a read operation.
    void connection_handler::handle_read_completion(
        boost::system::error_code const& e,
        boost::shared_ptr<receiver> receiver_conn)
    {
        if (!e) return;

        if (e != boost::asio::error::operation_aborted &&
            e != boost::asio::error::eof)
        {
            LPT_(error)
                << "handle read operation completion: error: "
                << e.message();

        }
//         if (e != boost::asio::error::eof)
        {
            // remove this connection from the list of known connections
            lcos::local::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(receiver_conn);
        }
    }
}}}}

#endif
