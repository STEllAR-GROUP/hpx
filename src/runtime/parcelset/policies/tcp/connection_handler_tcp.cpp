//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_PARCELPORT_TCP)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/tcp/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/tcp/sender.hpp>
#include <hpx/runtime/parcelset/policies/tcp/receiver.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/shared_ptr.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace tcp
{
    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , acceptor_(NULL)
    {
        /*
        if (here_.get_type() != connection_tcp) {
            HPX_THROW_EXCEPTION(network_error, "tcp::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
        */
    }

    connection_handler::~connection_handler()
    {
        if(acceptor_ != NULL)
        {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
            acceptor_ = NULL;
        }
    }

    bool connection_handler::do_run()
    {
        using boost::asio::ip::tcp;
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        if (NULL == acceptor_)
            acceptor_ = new tcp::acceptor(io_service);

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it = accept_begin(here_, io_service);
             it != end; ++it, ++tried)
        {
            try {
                boost::shared_ptr<receiver> receiver_conn(
                    new receiver(io_service, *this));

                tcp::endpoint ep = *it;
                acceptor_->open(ep.protocol());
                acceptor_->set_option(tcp::acceptor::reuse_address(true));
                acceptor_->bind(ep);
                acceptor_->listen();
                acceptor_->async_accept(receiver_conn->socket(),
                    boost::bind(&connection_handler::handle_accept,
                        this,
                        boost::asio::placeholders::error, receiver_conn));
            }
            catch (boost::system::system_error const&) {
                errors.add(boost::current_exception());
                continue;
            }
        }

        if (errors.size() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "tcp::parcelport::run", errors.get_message());
            return false;
        }
        return true;
    }

    void connection_handler::do_stop()
    {
        {
            // cancel all pending read operations, close those sockets
            lcos::local::spinlock::scoped_lock l(connections_mtx_);
            BOOST_FOREACH(boost::shared_ptr<receiver> c,
                accepted_connections_)
            {
                boost::system::error_code ec;
                boost::asio::ip::tcp::socket& s = c->socket();
                if (s.is_open()) {
                    s.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                    s.close(ec);    // close the socket to give it back to the OS
                }
            }

            accepted_connections_.clear();
#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
            write_connections_.clear();
#endif
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

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();

        // The parcel gets serialized inside the connection constructor, no
        // need to keep the original parcel alive after this call returned.
        boost::shared_ptr<sender> sender_connection(new sender(
            io_service, l, this->parcels_sent_));

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
                    boost::asio::ip::tcp::socket& s = sender_connection->socket();
                    s.close();
                    s.connect(*it, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend(hpx::threads::pending,
                        "connection_handler(tcp)::create_connection");
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                sender_connection->socket().close();
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "tcp::connection_handler::get_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->socket().close();
            sender_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "tcp::connection_handler::get_connection",
                hpx::util::osstream_get_string(strm));
            return sender_connection;
        }

        // make sure the Nagle algorithm is disabled for this socket,
        // disable lingering on close
        boost::asio::ip::tcp::socket& s = sender_connection->socket();

        s.set_option(boost::asio::ip::tcp::no_delay(true));
        s.set_option(boost::asio::socket_base::linger(true, 0));

#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
        {
            lcos::local::spinlock::scoped_lock lock(connections_mtx_);
            write_connections_.insert(sender_connection);
        }
#endif
#if defined(HPX_DEBUG)
        HPX_ASSERT(l == sender_connection->destination());

        std::string connection_addr = s.remote_endpoint().address().to_string();
        boost::uint16_t connection_port = s.remote_endpoint().port();
        HPX_ASSERT(l.get_address() == connection_addr);
        HPX_ASSERT(l.get_port() == connection_port);
#endif

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
    }

    // accepted new incoming connection
    void connection_handler::handle_accept(boost::system::error_code const & e,
        boost::shared_ptr<receiver> receiver_conn)
    {
        if(!e)
        {
            // handle this incoming connection
            boost::shared_ptr<receiver> c(receiver_conn);

            boost::asio::io_service& io_service = io_service_pool_.get_io_service();
            receiver_conn.reset(new receiver(io_service, *this));
            acceptor_->async_accept(receiver_conn->socket(),
                boost::bind(&connection_handler::handle_accept,
                    this,
                    boost::asio::placeholders::error, receiver_conn));

            {
                // keep track of all accepted connections
                lcos::local::spinlock::scoped_lock l(connections_mtx_);
                accepted_connections_.insert(c);
            }

            // disable Nagle algorithm, disable lingering on close
            boost::asio::ip::tcp::socket& s = c->socket();
            s.set_option(boost::asio::ip::tcp::no_delay(true));
            s.set_option(boost::asio::socket_base::linger(true, 0));

            // now accept the incoming connection by starting to read from the
            // socket
            c->async_read(
                boost::bind(&connection_handler::handle_read_completion,
                    this,
                    boost::asio::placeholders::error, c));
        }
        else
        {
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
            lcos::local::spinlock::scoped_lock l(connections_mtx_);
            accepted_connections_.erase(receiver_conn);
        }
    }
}}}}

#endif
