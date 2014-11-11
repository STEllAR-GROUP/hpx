//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_MPI_CONNECTION_HANDLER_HPP

#include <hpx/config/warnings_prefix.hpp>

#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/runtime/parcelset/policies/mpi/acceptor.hpp>
#include <hpx/runtime/parcelset/policies/mpi/locality.hpp>

namespace hpx { namespace parcelset {
    namespace policies { namespace mpi
    {
        class receiver;
        class sender;
        class HPX_EXPORT connection_handler;
    }}

    template <>
    struct connection_handler_traits<policies::mpi::connection_handler>
    {
        typedef policies::mpi::sender connection_type;
        typedef boost::mpl::true_  send_early_parcel;
        typedef boost::mpl::true_ do_background_work;
        typedef boost::mpl::true_ do_enable_parcel_handling;

        static const char * name()
        {
            return "mpi";
        }

        static const char * pool_name()
        {
            return "parcel_pool_mpi";
        }

        static const char * pool_name_postfix()
        {
            return "-mpi";
        }
    };

    namespace policies { namespace mpi
    {
        parcelset::locality parcelport_address(util::runtime_configuration const & ini);

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

            void background_work();

            /// Retrieve the type of the locality represented by this parcelport
            connection_type get_type() const
            {
                return connection_mpi;
            }

            /// Return the name of this locality
            std::string get_locality_name() const;

            boost::shared_ptr<sender> create_connection(
                parcelset::locality const& l, error_code& ec);

            void enable_parcel_handling(bool new_state);

            void add_sender(boost::shared_ptr<sender> const& sender_connection);
            void close_sender_connection(int tag, int rank);

            parcelset::locality agas_locality(util::runtime_configuration const & ini) const;
            parcelset::locality create_locality() const;

        private:
            int get_next_tag();
            void handle_messages();

            MPI_Comm communicator_;
            // handle messages
            acceptor acceptor_;

            hpx::lcos::local::spinlock close_mtx_;
            std::vector<std::pair<int, int> > pending_close_requests_;

            typedef std::map<int, boost::shared_ptr<receiver> > receivers_tag_map_type;
            typedef std::map<int, receivers_tag_map_type> receivers_rank_map_type;
            receivers_rank_map_type receivers_map_;

            typedef std::list<boost::shared_ptr<receiver> > receivers_type;
            receivers_type receivers_;

            hpx::lcos::local::spinlock senders_mtx_;
            typedef std::list<boost::shared_ptr<sender> > senders_type;
            senders_type senders_;

            boost::atomic<bool> stopped_;
            boost::atomic<bool> handling_messages_;

            hpx::lcos::local::spinlock tag_mtx_;
            std::size_t next_tag_;
            std::deque<int> free_tags_;

            bool use_io_pool_;
        };
    }}
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
