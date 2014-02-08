//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_CONNECTION_HANDLER_HPP

#include <hpx/config/warnings_prefix.hpp>

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/acceptor.hpp>


namespace hpx { namespace parcelset {
    namespace policies { namespace ibverbs
    {
        class receiver;
        class sender;
        class HPX_EXPORT connection_handler;
    }}

    template <>
    struct connection_handler_traits<policies::ibverbs::connection_handler>
    {
        typedef policies::ibverbs::sender connection_type;
        typedef boost::mpl::false_  send_early_parcel;
        typedef boost::mpl::false_ do_background_work;
        typedef boost::mpl::false_ do_enable_parcel_handling;

        static const char * name()
        {
            return "ibverbs";
        }

        static const char * pool_name()
        {
            return "parcel_pool_ibverbs";
        }

        static const char * pool_name_postfix()
        {
            return "-ibverbs";
        }
    };

    namespace policies { namespace ibverbs
    {
        class HPX_EXPORT connection_handler
          : public parcelport_impl<connection_handler>
        {
            typedef parcelport_impl<connection_handler> base_type;

        public:
            static std::vector<std::string> runtime_configuration();

            connection_handler(util::runtime_configuration const& ini,
                HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
                HPX_STD_FUNCTION<void()> const& on_stop_thread);

            ~connection_handler();

            /// Start the handling of connections.
            bool do_run();

            /// Stop the handling of connections.
            void do_stop();

            /// Retrieve the type of the locality represented by this parcelport
            connection_type get_type() const
            {
                return connection_ibverbs;
            }

            /// Return the name of this locality
            std::string get_locality_name() const;

            boost::shared_ptr<sender> create_connection(
                naming::locality const& l, error_code& ec);

        private:
            // helper functions for receiving parcels
            void handle_accept(boost::system::error_code const& e,
                boost::shared_ptr<receiver>);
            void handle_read_completion(boost::system::error_code const& e,
                boost::shared_ptr<receiver>);

            /// Acceptor used to listen for incoming connections.
            acceptor* acceptor_;

            /// The list of accepted connections
            typedef std::set<boost::shared_ptr<receiver> > accepted_connections_set;
            accepted_connections_set accepted_connections_;
        };
    }}
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
