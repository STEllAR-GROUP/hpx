//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/system/error_code.hpp
// hpxinspect:nodeprecatedname:boost::system::error_code
// hpxinspect:nodeprecatedinclude:boost/system/system_error.hpp
// hpxinspect:nodeprecatedname:boost::system::system_error

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/config/detail/compat_error_code.hpp>
#include <hpx/asio/asio_util.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/plugins/parcelport/tcp/connection_handler.hpp>
#include <hpx/plugins/parcelport/tcp/receiver.hpp>
#include <hpx/plugins/parcelport/tcp/sender.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/util/get_entry_as.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>

#include <thread>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>

namespace hpx { namespace parcelset { namespace policies { namespace tcp
{
    parcelset::locality parcelport_address(util::runtime_configuration const & ini)
    {
        // load all components as described in the configuration information
        if (ini.has_section("hpx.parcel")) {
            util::section const* sec = ini.get_section("hpx.parcel");
            if (nullptr != sec) {
                return parcelset::locality(
                    locality(
                        sec->get_entry("address", HPX_INITIAL_IP_ADDRESS)
                      , hpx::util::get_entry_as<std::uint16_t>(
                            *sec, "port", HPX_INITIAL_IP_PORT)
                    )
                );
            }
        }
        return
            parcelset::locality(
                locality(
                    HPX_INITIAL_IP_ADDRESS
                  , HPX_INITIAL_IP_PORT
                )
            );
    }

    connection_handler::connection_handler(
        util::runtime_configuration const& ini,
        threads::policies::callback_notifier const& notifier)
      : base_type(ini, parcelport_address(ini), notifier)
      , acceptor_(nullptr)
    {
        if (here_.type() != std::string("tcp")) {
            HPX_THROW_EXCEPTION(network_error, "tcp::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + std::string(here_.type()));
        }
    }

    connection_handler::~connection_handler()
    {
        HPX_ASSERT(acceptor_ == nullptr);
    }

    bool connection_handler::do_run()
    {
        using boost::asio::ip::tcp;
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        if (nullptr == acceptor_)
            acceptor_ = new tcp::acceptor(io_service);

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        util::endpoint_iterator_type end = util::accept_end();
        for (util::endpoint_iterator_type it =
                util::accept_begin(here_.get<locality>(), io_service);
             it != end; ++it, ++tried)
        {
            try {
                std::shared_ptr<receiver> receiver_conn(
                    new receiver(io_service, get_max_inbound_message_size(), *this));

                tcp::endpoint ep = *it;
                acceptor_->open(ep.protocol());
                acceptor_->set_option(tcp::acceptor::reuse_address(true));
                acceptor_->bind(ep);
                acceptor_->listen();
                acceptor_->async_accept(receiver_conn->socket(),
                    util::bind(&connection_handler::handle_accept,
                        this,
                        boost::asio::placeholders::error, receiver_conn));
            }
            catch (boost::system::system_error const&) {
                errors.add(std::current_exception());
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
            std::lock_guard<lcos::local::spinlock> l(connections_mtx_);
            for (std::shared_ptr<receiver> const& c : accepted_connections_)
            {
                c->shutdown();
            }

            accepted_connections_.clear();
#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
            write_connections_.clear();
#endif
        }
        if(acceptor_ != nullptr)
        {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
            acceptor_ = nullptr;
        }
    }

    std::shared_ptr<sender> connection_handler::create_connection(
        parcelset::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();

        // The parcel gets serialized inside the connection constructor, no
        // need to keep the original parcel alive after this call returned.
        std::shared_ptr<sender> sender_connection(new sender(io_service, l, this));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            // The acceptor is only nullptr when the parcelport has been stopped.
            // An exit here, avoids hangs when late parcels are in flight (those are
            // mainly decref requests).
            if(acceptor_ == nullptr)
                return std::shared_ptr<sender>();
            try {
                util::endpoint_iterator_type end = util::connect_end();
                for (util::endpoint_iterator_type it =
                        util::connect_begin(l.get<locality>(), io_service);
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
                    this_thread::suspend(
                        hpx::threads::thread_schedule_state::pending,
                        "connection_handler(tcp)::create_connection");
                }
                else {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
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

            std::ostringstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            if (tolerate_node_faults())
                return sender_connection;

            HPX_THROWS_IF(ec, network_error,
                "tcp::connection_handler::get_connection",
                strm.str());
            return sender_connection;
        }

        // make sure the Nagle algorithm is disabled for this socket,
        // disable lingering on close
        boost::asio::ip::tcp::socket& s = sender_connection->socket();

        s.set_option(boost::asio::ip::tcp::no_delay(true));
        s.set_option(boost::asio::socket_base::linger(true, 0));

#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
        {
            std::lock_guard<lcos::local::spinlock> lock(connections_mtx_);
            write_connections_.insert(sender_connection);
        }
#endif
#if defined(HPX_DEBUG)
        HPX_ASSERT(l == sender_connection->destination());

        std::string connection_addr = s.remote_endpoint().address().to_string();
        std::uint16_t connection_port = s.remote_endpoint().port();
        HPX_ASSERT(hpx::util::cleanup_ip_address(l.get<locality>().address())
            == hpx::util::cleanup_ip_address(connection_addr));
        HPX_ASSERT(l.get<locality>().port() == connection_port);
#endif

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
    }

    parcelset::locality connection_handler::agas_locality(
        util::runtime_configuration const & ini) const
    {
        // load all components as described in the configuration information
        if (ini.has_section("hpx.agas")) {
            util::section const* sec = ini.get_section("hpx.agas");
            if (nullptr != sec) {
                return
                    parcelset::locality(
                        locality(
                            sec->get_entry("address", HPX_INITIAL_IP_ADDRESS)
                          , hpx::util::get_entry_as<std::uint16_t>(
                                *sec, "port", HPX_INITIAL_IP_PORT)
                        )
                    );
            }
        }
        return
            parcelset::locality(
                locality(
                    HPX_INITIAL_IP_ADDRESS
                  , HPX_INITIAL_IP_PORT
                )
            );
    }

    parcelset::locality connection_handler::create_locality() const
    {
        return parcelset::locality(locality());
    }

    // accepted new incoming connection
    void connection_handler::handle_accept(std::error_code const & e,
        std::shared_ptr<receiver> receiver_conn)
    {
        if(!e)
        {
            // handle this incoming connection
            std::shared_ptr<receiver> c(receiver_conn);

            boost::asio::io_service& io_service = io_service_pool_.get_io_service();
            receiver_conn.reset(new receiver(io_service, get_max_inbound_message_size(),
                *this));
            acceptor_->async_accept(receiver_conn->socket(),
                util::bind(&connection_handler::handle_accept,
                    this,
                    boost::asio::placeholders::error, receiver_conn));

            {
                // keep track of all accepted connections
                std::lock_guard<lcos::local::spinlock> l(connections_mtx_);
                accepted_connections_.insert(c);
            }

            // disable Nagle algorithm, disable lingering on close
            boost::asio::ip::tcp::socket& s = c->socket();
            s.set_option(boost::asio::ip::tcp::no_delay(true));
            s.set_option(boost::asio::socket_base::linger(true, 0));

            // now accept the incoming connection by starting to read from the
            // socket
            c->async_read(
                util::bind(&connection_handler::handle_read_completion,
                    this,
                    boost::asio::placeholders::error, c));
        }
        else
        {
            // remove this connection from the list of known connections
            std::lock_guard<lcos::local::spinlock> l(mtx_);
            accepted_connections_.erase(receiver_conn);
        }
    }

    // Handle completion of a read operation.
    void connection_handler::handle_read_completion(
        std::error_code const& e,
        std::shared_ptr<receiver> receiver_conn)
    {
        if (!e) return;

        if (!compat_error_code::equal(
                e, boost::asio::error::operation_aborted) &&
            !compat_error_code::equal(e, boost::asio::error::eof))
        {
            LPT_(error)
                << "handle read operation completion: error: "
                << e.message();
        }

        //if (!compat_error_code::equal(e, boost::asio::error::eof))
        {
            // remove this connection from the list of known connections
            std::lock_guard<lcos::local::spinlock> l(connections_mtx_);
            accepted_connections_.erase(receiver_conn);
        }
    }
}}}}

#endif
