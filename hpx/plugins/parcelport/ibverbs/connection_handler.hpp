//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_CONNECTION_HANDLER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_CONNECTION_HANDLER_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/plugins/parcelport/ibverbs/acceptor.hpp>
#include <hpx/plugins/parcelport/ibverbs/locality.hpp>

#include <hpx/util/cache/local_cache.hpp>
#include <hpx/util/cache/entries/lru_entry.hpp>
#include <hpx/util/memory_chunk_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/atomic.hpp>

#include <map>
#include <memory>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

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
        typedef boost::mpl::true_ do_background_work;
        typedef boost::mpl::false_ do_enable_parcel_handling;

        static const char * type()
        {
            return "ibverbs";
        }

        static const char * pool_name()
        {
            return "parcel-pool-ibverbs";
        }

        static const char * pool_name_postfix()
        {
            return "-ibverbs";
        }
    };

    namespace policies { namespace ibverbs
    {
        parcelset::locality parcelport_address(util::runtime_configuration const & ini);

        class HPX_EXPORT connection_handler
          : public parcelport_impl<connection_handler>
        {
            typedef parcelport_impl<connection_handler> base_type;

        public:
            static std::size_t memory_chunk_size(util::runtime_configuration const& ini);
            static std::size_t max_memory_chunks(util::runtime_configuration const& ini);

            connection_handler(util::runtime_configuration const& ini,
                util::function_nonser<void(std::size_t, char const*)>
                    const& on_start_thread,
                util::function_nonser<void()> const& on_stop_thread);

            ~connection_handler();

            /// Start the handling of connections.
            bool do_run();

            /// Stop the handling of connections.
            void do_stop();

            void background_work();

            std::shared_ptr<sender> create_connection(
                parcelset::locality const& l, error_code& ec);

            parcelset::locality agas_locality(util::runtime_configuration const & ini)
                const;

            parcelset::locality create_locality() const;

            void add_sender(std::shared_ptr<sender> const& sender_connection);

            ibv_pd *get_pd(ibv_context *context, boost::system::error_code & ec);

            ibverbs_mr register_buffer(ibv_pd * pd, char * buffer, std::size_t size,
                int access);

        private:
            // helper functions for receiving parcels
            void handle_messages();
            bool do_sends();
            bool do_receives();
            void handle_accepts();

            util::memory_chunk_pool memory_pool_;

            typedef std::map<ibv_context *, ibv_pd *> pd_map_type;
            hpx::lcos::local::spinlock pd_map_mtx_;
            pd_map_type pd_map_;

            std::size_t mr_cache_size_;

            typedef std::pair<char *, util::memory_chunk_pool::size_type> chunk_pair;
            typedef hpx::util::cache::entries::lru_entry<ibverbs_mr> mr_cache_entry_type;
            typedef hpx::util::cache::local_cache<chunk_pair, mr_cache_entry_type>
                mr_cache_type;
            /*
            typedef
                std::map<chunk_pair, ibverbs_mr>
                mr_cache_type;
            */
            typedef
                std::map<ibv_pd *, mr_cache_type>
                mr_map_type;

            hpx::lcos::local::spinlock mr_map_mtx_;
            mr_map_type mr_map_;

            ibv_device **device_list_;
            std::vector<ibv_context *> context_list_;

            /// Acceptor used to listen for incoming connections.
            acceptor acceptor_;

            hpx::lcos::local::spinlock receivers_mtx_;
            typedef std::list<std::shared_ptr<receiver> > receivers_type;
            receivers_type receivers_;

            hpx::lcos::local::spinlock senders_mtx_;
            typedef std::list<std::shared_ptr<sender> > senders_type;
            senders_type senders_;

            boost::atomic<bool> stopped_;
            boost::atomic<bool> handling_messages_;
            boost::atomic<bool> handling_accepts_;

            bool use_io_pool_;
        };
    }}
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

#endif
