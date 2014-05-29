//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/asio_util.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/host_name.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    locality::iterator_type accept_begin(locality const& loc,
        boost::asio::io_service& io_service, bool ibverbs)
    {
        using boost::asio::ip::tcp;

        // collect errors here
        exception_list errors;

        std::string port_str(boost::lexical_cast<std::string>(loc.get_port()));

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        std::string address = ibverbs ? loc.get_ibverbs_address() : loc.get_address();
#else
        std::string address = loc.get_address();
#endif

        // try to directly create an endpoint from the address
        try {
            tcp::endpoint ep;
            if (util::get_endpoint(address, loc.get_port(), ep))
            {
                return locality::iterator_type(
                    tcp::resolver::iterator::create(ep, address, port_str));
            }
        }
        catch (boost::system::system_error const&) {
            errors.add(boost::current_exception());
        }

        // it's not an address, try to treat it as a host name
        try {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(address, port_str);

            return locality::iterator_type(resolver.resolve(query));
        }
        catch (boost::system::system_error const&) {
            errors.add(boost::current_exception());
        }

        // it's not a host name either, create a custom iterator allowing to
        // filter the returned endpoints, for this we use "localhost" as the
        // address to enumerate endpoints
        try {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(boost::asio::ip::host_name(), port_str);

            return locality::iterator_type(detail::is_valid_endpoint(
                address), resolver.resolve(query));
        }
        catch (boost::system::system_error const&) {
            errors.add(boost::current_exception());
        }

        // report errors
        hpx::util::osstream strm;
        strm << errors.get_message() << " (while trying to resolve: "
             << address << ":" << loc.get_port() << ")";

        HPX_THROW_EXCEPTION(network_error, "accept_begin",
            hpx::util::osstream_get_string(strm));
        return locality::iterator_type();
    }

    //////////////////////////////////////////////////////////////////////////
    locality::iterator_type connect_begin(locality const& loc,
        boost::asio::io_service& io_service, bool ibverbs)
    {
        using boost::asio::ip::tcp;

        // collect errors here
        exception_list errors;

        std::string port_str(boost::lexical_cast<std::string>(loc.get_port()));
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        std::string address = ibverbs ? loc.get_ibverbs_address() : loc.get_address();
#else
        std::string address = loc.get_address();
#endif

        // try to directly create an endpoint from the address
        try {
            tcp::endpoint ep;
            if (util::get_endpoint(address, loc.get_port(), ep))
            {
                return locality::iterator_type(tcp::resolver::iterator::create(
                    ep, address, port_str));
            }
        }
        catch (boost::system::system_error const&) {
            errors.add(boost::current_exception());
        }

        // it's not an address, try to treat it as a host name
        try {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(
                !address.empty() ?
                    address :
                    boost::asio::ip::host_name(),
                port_str);

            return locality::iterator_type(resolver.resolve(query));
        }
        catch (boost::system::system_error const&) {
            errors.add(boost::current_exception());
        }

        // report errors
        hpx::util::osstream strm;
        strm << errors.get_message() << " (while trying to connect to: "
             << address << ":" << loc.get_port() << ")";

        HPX_THROW_EXCEPTION(network_error, "connect_begin",
            hpx::util::osstream_get_string(strm));

        return locality::iterator_type();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void locality::save(Archive& ar, const unsigned int version) const
    {
        ar.save(address_);
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        ar.save(ibverbs_address_);
#endif
        ar.save(port_);

#if defined(HPX_HAVE_PARCELPORT_MPI)
        HPX_ASSERT(HPX_LOCALITY_VERSION_MPI == version);
        ar.save(rank_);
#endif
    }

    template <typename Archive>
    void locality::load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_LOCALITY_VERSION_MPI)
        {
            HPX_THROW_EXCEPTION(version_too_new,
                "locality::load",
                "trying to load locality with unknown version");
            return;
        }

        ar.load(address_);
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        ar.load(ibverbs_address_);
#endif
        ar.load(port_);

#if defined(HPX_HAVE_PARCELPORT_MPI)
        // try to read rank only if the sender knows about MPI
        if (version > HPX_LOCALITY_VERSION_NO_MPI)
            ar.load(rank_);
#else
        // account for the additional rank
        if (version > HPX_LOCALITY_VERSION_NO_MPI)
        {
            int rank = -1;
            ar.load(rank);

            if (rank != -1) {
            // FIXME: we might have received a locality which is of
            // no use to us as our locality is not configured to
            // support MPI.
                HPX_THROW_EXCEPTION(version_unknown,
                    "locality::load",
                    "load locality with valid rank while MPI was "
                        "not configured");
            }
            return;
        }
#endif
    }

    template HPX_EXPORT
    void locality::save(util::portable_binary_oarchive&, const unsigned int) const;

    template HPX_EXPORT
    void locality::load(util::portable_binary_iarchive&, const unsigned int);
}}
