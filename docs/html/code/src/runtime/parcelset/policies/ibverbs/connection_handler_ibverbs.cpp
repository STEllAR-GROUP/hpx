//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/acceptor.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/sender.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/receiver.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines +=
            "buffer_size = ${HPX_PARCEL_IBVERBS_BUFFER_SIZE:65536}"
            ;

        return lines;
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , acceptor_(0)
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

    bool connection_handler::run()
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
                        io_service_pool_.get_io_service(), *this));
                conn->get_buffer();

                boost::asio::ip::tcp::endpoint ep = *it;

                acceptor_->bind(ep);

                acceptor_->async_accept(conn->context(),
                    boost::bind(&connection_handler::handle_accept,
                        this,
                        boost::asio::placeholders::error, conn));
            }
            catch (boost::system::system_error const& e) {
                errors.add(e);   // store all errors
                continue;
            }
        }

        if (errors.get_error_count() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "ibverbs::connection_handler::run", errors.get_message());
            return false;
        }
        return true;
    }

    void connection_handler::stop()
    {
        {
            // cancel all pending read operations, close those sockets
            lcos::local::spinlock::scoped_lock l(mtx_);
            BOOST_FOREACH(boost::shared_ptr<receiver> c, accepted_connections_)
            {
                boost::system::error_code ec;
                server_context& ctx = c->context();
                ctx.shutdown(ec); // shut down connection
                ctx.close(ec);    // close the data window to give it back to the OS
            }
            accepted_connections_.clear();
        }

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
        return "ibverbs";
    }

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        boost::shared_ptr<sender> sender_connection(new sender(
            io_service, l, parcels_sent_));

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

                    client_context& ctx = sender_connection->context();
                    sender_connection->get_buffer(parcel(), 0);
                    ctx.close();
                    ctx.connect(ep, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend(hpx::threads::pending,
                        "connection_handler(ibverbs)::create_connection");
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                sender_connection->context().close();
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "ibverbs::parcelport::get_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->context().close();
            sender_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "ibverbs::parcelport::get_connection",
                hpx::util::osstream_get_string(strm));
            return sender_connection;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
    }
    
    // accepted new incoming connection
    void connection_handler::handle_accept(boost::system::error_code const & e,
        boost::shared_ptr<receiver> conn)
    {
        if (!e) {
            // handle this incoming parcel
            boost::shared_ptr<receiver> c(conn); // hold on to receiver_conn

            // create new connection waiting for next incoming parcel
            conn.reset(new receiver(
                io_service_pool_.get_io_service(), *this));
            conn->get_buffer();

            acceptor_->async_accept(conn->context(),
                boost::bind(&connection_handler::handle_accept,
                    this,
                    boost::asio::placeholders::error, conn));

            {
                // keep track of all the accepted connections
                lcos::local::spinlock::scoped_lock l(mtx_);
                accepted_connections_.insert(c);
            }

            // now accept the incoming connection by starting to read from the
            // context
            c->async_read(boost::bind(&connection_handler::handle_read_completion,
                this, boost::asio::placeholders::error, c));
        }
        else {
            // remove this connection from the list of known connections
            lcos::local::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(conn);
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
