//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_TCP_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_TCP_CONNECTION_HANDLER_HPP

#include <hpx/config/warnings_prefix.hpp>

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/runtime/parcelset/policies/tcp/locality.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/host_name.hpp>

namespace hpx { namespace parcelset
{
    namespace policies { namespace tcp
    {
        class receiver;
        class sender;
        class HPX_EXPORT connection_handler;
    }}

    template <>
    struct connection_handler_traits<policies::tcp::connection_handler>
    {
        typedef policies::tcp::sender connection_type;
        typedef boost::mpl::true_  send_early_parcel;
        typedef boost::mpl::false_ do_background_work;
        typedef boost::mpl::false_ do_enable_parcel_handling;

        static const char * name()
        {
            return "tcp";
        }

        static const char * pool_name()
        {
            return "parcel_pool_tcp";
        }

        static const char * pool_name_postfix()
        {
            return "-tcp";
        }
    };

    namespace policies { namespace tcp
    {
        parcelset::locality parcelport_address(util::runtime_configuration const & ini);

        class HPX_EXPORT connection_handler
          : public parcelport_impl<connection_handler>
        {
            typedef parcelport_impl<connection_handler> base_type;
        public:

            static std::vector<std::string> runtime_configuration()
            {
                std::vector<std::string> lines;

                return lines;
            }

            connection_handler(util::runtime_configuration const& ini,
                HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
                HPX_STD_FUNCTION<void()> const& on_stop_thread);

            ~connection_handler();

            /// Start the handling of connections.
            bool do_run();

            /// Stop the handling of connectons.
            void do_stop();

            /// Retrieve the type of the locality represented by this parcelport
            connection_type get_type() const
            {
                return connection_tcp;
            }

            /// Return the name of this locality
            std::string get_locality_name() const
            {
                return boost::asio::ip::host_name();
            }

            boost::shared_ptr<sender> create_connection(
                parcelset::locality const& l, error_code& ec);

            parcelset::locality agas_locality(util::runtime_configuration const & ini) const;

            parcelset::locality create_locality() const;

        private:
            void handle_accept(boost::system::error_code const & e,
                boost::shared_ptr<receiver> receiver_conn);
            void handle_read_completion(boost::system::error_code const& e,
                boost::shared_ptr<receiver> receiver_conn);

            /// Acceptor used to listen for incoming connections.
            boost::asio::ip::tcp::acceptor* acceptor_;

            /// The list of accepted connections
            mutable lcos::local::spinlock connections_mtx_;

            typedef std::set<boost::shared_ptr<receiver> > accepted_connections_set;
            accepted_connections_set accepted_connections_;

#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
            typedef std::set<boost::weak_ptr<sender> > write_connections_set;
            write_connections_set write_connections_;
#endif
        };
    }}
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
