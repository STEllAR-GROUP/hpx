//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/integer/endian.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/naming/resolver_client_connection.hpp>
#include <hpx/runtime/naming/server/reply.hpp>
#include <hpx/runtime/naming/server/request.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/find_msb.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    resolver_client::resolver_client(util::io_service_pool& io_service_pool, 
            std::string const& address, unsigned short port,
            bool start_asynchronously) 
      : io_service_pool_(io_service_pool), 
        socket_(io_service_pool.get_io_service())
    {
        try {
            if (start_asynchronously)
                io_service_pool.run(false);

            using namespace boost::asio::ip;

            // try to convert the address string to an IP address directly
            boost::system::error_code error = boost::asio::error::try_again;
            tcp::endpoint ep;
            if (util::get_endpoint(address, port, ep)) {
                socket_.connect(ep, error);
                there_ = locality(ep);
                if (!error) 
                    return;
            }

            // resolve the given address
            tcp::resolver resolver(io_service_pool.get_io_service());
            tcp::resolver::query query(address, 
                boost::lexical_cast<std::string>(port));

            // Try each endpoint until we successfully establish a connection.
            tcp::resolver::iterator end;
            tcp::resolver::iterator it = resolver.resolve(query);
            for (/**/; it != end; ++it)
            {
                socket_.close();
                socket_.connect(*it, error);
                if (!error) {
                    there_ = locality(*it);
                    return;
                }
            }
            if (error) {
                socket_.close();
                boost::throw_exception(
                    hpx::exception(network_error, error.message()));
            }
        }
        catch (boost::system::error_code const& e) {
            boost::throw_exception(hpx::exception(network_error, e.message()));
        }
    }

    resolver_client::resolver_client(util::io_service_pool& io_service_pool, 
            locality l, bool start_asynchronously) 
      : there_(l), io_service_pool_(io_service_pool), 
        socket_(io_service_pool.get_io_service())
    {
        if (start_asynchronously)
            io_service_pool.run(false);

        boost::system::error_code error = boost::asio::error::try_again;
        socket_.connect(l.get_endpoint(), error);
        if (error) {
            socket_.close();
            boost::throw_exception(
                hpx::exception(network_error, error.message()));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // synchronous functionality
    bool resolver_client::get_prefix(locality const& l, id_type& prefix) const
    {
        // send request
        server::request req (server::command_getprefix, l);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != repeated_request)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        prefix = rep.get_prefix();
        return s == success;
    }

    bool resolver_client::get_prefixes(std::vector<id_type>& prefixes) const
    {
        // send request
        server::request req (server::command_getprefixes);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        typedef std::vector<boost::uint32_t>::const_iterator iterator;
        iterator end = rep.get_prefixes().end();
        for (iterator it = rep.get_prefixes().begin(); it != end; ++it)
            prefixes.push_back(get_id_from_prefix(*it));

        return s == success;
    }

    bool resolver_client::get_id_range(locality const& l, std::size_t count, 
        id_type& lower_bound, id_type& upper_bound) const
    {
        // send request
        server::request req (server::command_getidrange, l, count);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != repeated_request)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        lower_bound = rep.get_lower_bound();
        upper_bound = rep.get_upper_bound();
        return s == success;
    }

    bool resolver_client::bind_range(id_type const& id, std::size_t count, 
        address const& addr, std::ptrdiff_t offset,
        std::size_t gids_per_object) const
    {
        // make sure gids_per_object is dividable by 2
        BOOST_ASSERT(gids_per_object == util::find_msb_value(gids_per_object));

        // send request
        server::request req (server::command_bind_range, id, count, addr, 
            offset, gids_per_object);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        return s == success;
    }

    bool resolver_client::unbind_range(id_type const& id, std::size_t count, 
        address& addr) const
    {
        // send request
        server::request req (server::command_unbind_range, id, count);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        addr = rep.get_address();
        return s == success;
    }

    bool resolver_client::resolve(id_type const& id, address& addr) const
    {
        // send request
        server::request req (server::command_resolve, id);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        addr = rep.get_address();
        return s == success;
    }

    bool resolver_client::registerid(std::string const& ns_name, 
        id_type const& id) const
    {
        // send request
        server::request req (server::command_registerid, ns_name, id);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        return s == success;
    }

    bool resolver_client::unregisterid(std::string const& ns_name) const
    {
        // send request
        server::request req (server::command_unregisterid, ns_name);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        return s == success;
    }

    bool resolver_client::queryid(std::string const& ns_name, id_type& id) const
    {
        // send request
        server::request req (server::command_queryid, ns_name);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        id = rep.get_id();
        return s == success;
    }

    bool resolver_client::get_statistics_count(std::vector<std::size_t>& counts) const
    {
        // send request
        server::request req (server::command_statistics_count);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        counts.clear();
        for (std::size_t i = 0; i < server::command_lastcommand; ++i)
            counts.push_back(std::size_t(rep.get_statictics(i)));

        return s == success;
    }

    bool resolver_client::get_statistics_mean(std::vector<double>& timings) const
    {
        // send request
        server::request req (server::command_statistics_mean);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        std::swap(timings, rep.get_statictics());
        return s == success;
    }

    bool resolver_client::get_statistics_moment2(std::vector<double>& timings) const
    {
        // send request
        server::request req (server::command_statistics_moment2);
        server::reply rep;
        execute(req, rep);

        hpx::error s = (hpx::error) rep.get_status();
        if (s != success && s != no_success)
            boost::throw_exception(hpx::exception((error)s, rep.get_error()));

        std::swap(timings, rep.get_statictics());
        return s == success;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool resolver_client::read_completed(boost::system::error_code const& err, 
        std::size_t bytes_transferred, boost::uint32_t size)
    {
        return bytes_transferred >= size;
    }

    bool resolver_client::write_completed(boost::system::error_code const& err, 
        std::size_t bytes_transferred, boost::uint32_t size)
    {
        return bytes_transferred >= size;
    }

    void resolver_client::execute(server::request const &req, 
        server::reply& rep) const
    {
        typedef util::container_device<std::vector<char> > io_device_type;

        try {
            // connect socket
            std::vector<char> buffer;
            {
                // serialize the request
                boost::iostreams::stream<io_device_type> io(buffer);
                util::portable_binary_oarchive archive(io);
                archive << req;
            }

            boost::system::error_code err = boost::asio::error::fault;

            // send the data
            boost::integer::ulittle32_t size = buffer.size();
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&size, sizeof(size)));
            buffers.push_back(boost::asio::buffer(buffer));
            std::size_t written_bytes = boost::asio::write(socket_, buffers,
                boost::bind(&resolver_client::write_completed, _1, _2, 
                    buffer.size() + sizeof(size)),
                err);
            if (err) {
                boost::throw_exception(
                    hpx::exception(network_error, err.message()));
            }
            if (buffer.size() + sizeof(size) != written_bytes) {
                boost::throw_exception(
                    hpx::exception(network_error, "network write failed"));
            }

            // wait for response
            // first read the size of the message 
            std::size_t reply_length = boost::asio::read(socket_,
                boost::asio::buffer(&size, sizeof(size)),
                boost::bind(&resolver_client::read_completed, _1, _2, sizeof(size)),
                err);
            if (err) {
                boost::throw_exception(
                    hpx::exception(network_error, err.message()));
            }
            if (reply_length != sizeof(size)) {
                boost::throw_exception(
                    hpx::exception(network_error, "network read failed"));
            }

            // now read the rest of the message
            boost::uint32_t native_size = size;
            buffer.resize(native_size);
            reply_length = boost::asio::read(socket_, boost::asio::buffer(buffer), 
                boost::bind(&resolver_client::read_completed, _1, _2, native_size), 
                err);

            if (err) {
                boost::throw_exception(
                    hpx::exception(network_error, err.message()));
            }
            if (reply_length != native_size) {
                boost::throw_exception(
                    hpx::exception(network_error, "network read failed"));
            }

            // De-serialize the data
            {
                boost::iostreams::stream<io_device_type> io(buffer);
                util::portable_binary_iarchive archive(io);
                archive >> rep;
            }
        }
        catch (boost::system::error_code const& e) {
            boost::throw_exception(hpx::exception(network_error, e.message()));
        }        
        catch (std::exception const& e) {
            boost::throw_exception(hpx::exception(network_error, e.what()));
        }
        catch(...) {
            boost::throw_exception(hpx::exception(no_success, 
                "unexpected error"));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // asynchronous API
    util::unique_future<bool> 
    resolver_client::bind_range_async(id_type const& lower_id, std::size_t count, 
        address const& addr, std::ptrdiff_t offset, std::size_t gids_per_object)
    {
        typedef resolver_client_connection<bool> connection_type;

        // make sure gids_per_object is dividable by 2
        BOOST_ASSERT(gids_per_object == util::find_msb_value(gids_per_object));

        // prepare request
        connection_type* conn = new connection_type(socket_, 
            server::command_bind_range, lower_id, count, addr, offset, 
            gids_per_object);
        boost::shared_ptr<connection_type> client_conn(conn);

        conn->execute();
        return conn->get_future();
    }

    util::unique_future<bool> 
    resolver_client::unbind_range_async(id_type const& lower_id, std::size_t count)
    {
        typedef resolver_client_connection<bool> connection_type;

        // prepare request
        connection_type* conn = new connection_type(socket_, 
            server::command_unbind_range, lower_id, count);
        boost::shared_ptr<connection_type> client_conn(conn);

        conn->execute();
        return conn->get_future();
    }

    util::unique_future<std::pair<bool, address> >  
    resolver_client::resolve_async(id_type const& id)
    {
        typedef 
            resolver_client_connection<std::pair<bool, address> > 
        connection_type;

        // prepare request
        connection_type* conn = new connection_type(socket_, 
            server::command_resolve, id);
        boost::shared_ptr<connection_type> client_conn(conn);

        conn->execute();
        return conn->get_future();
    }

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming

