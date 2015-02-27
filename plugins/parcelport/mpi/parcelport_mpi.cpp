//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>
#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/util/command_line_handling.hpp>

// parcelport
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/condition_variable.hpp>

#include <hpx/plugins/parcelport/mpi/locality.hpp>
#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/plugins/parcelport/mpi/sender.hpp>
#include <hpx/plugins/parcelport/mpi/receiver.hpp>

#include <hpx/util/memory_chunk_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class parcelport
      : public parcelset::parcelport
    {
    private:
        static parcelset::locality here()
        {
            return
                parcelset::locality(
                    locality(
                        util::mpi_environment::rank()
                    )
                );
        }

        static std::size_t max_connections(util::runtime_configuration const& ini)
        {
            return hpx::util::get_entry_as<std::size_t>(
                ini, "hpx.parcel.mpi.max_connections", HPX_PARCEL_MAX_CONNECTIONS);
        }
    public:
        parcelport(util::runtime_configuration const& ini,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread)
          : parcelset::parcelport(ini, here(), "mpi")
          , archive_flags_(boost::archive::no_header)
          , stopped_(false)
          , bootstrapping_(true)
          , max_connections_(max_connections(ini))
          , chunk_pool_(4096, max_connections_)
          , sender_(stopped_, max_connections_)
          , receiver_(*this, chunk_pool_, stopped_, max_connections_)
        {
#ifdef BOOST_BIG_ENDIAN
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
            if (endian_out == "little")
                archive_flags_ |= util::endian_little;
            else if (endian_out == "big")
                archive_flags_ |= util::endian_big;
            else {
                HPX_ASSERT(endian_out =="little" || endian_out == "big");
            }

            if (!this->allow_array_optimizations()) {
                archive_flags_ |= util::disable_array_optimization;
                archive_flags_ |= util::disable_data_chunking;
            }
            else {
                if (!this->allow_zero_copy_optimizations())
                    archive_flags_ |= util::disable_data_chunking;
            }
        }

        ~parcelport()
        {
            receive_early_parcels_thread_.join();
            util::mpi_environment::finalize();
        }

        bool can_bootstrap() const
        {
            return true;
        }

        /// Return the name of this locality
        std::string get_locality_name() const
        {
            return util::mpi_environment::get_processor_name();
        }

        parcelset::locality
        agas_locality(util::runtime_configuration const & ini) const
        {
            return
                parcelset::locality(
                    locality(
                        util::mpi_environment::enabled() ? 0 : -1
                    )
                );
        }

        parcelset::locality create_locality() const
        {
            return parcelset::locality(locality());
        }

        void put_parcels(std::vector<parcelset::locality> dests,
            std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers)
        {
            HPX_ASSERT(dests.size() == parcels.size());
            HPX_ASSERT(dests.size() == handlers.size());
            for(std::size_t i = 0; i != dests.size(); ++i)
            {
                put_parcel(dests[i], parcels[i], handlers[i]);
            }
        }

        void send_early_parcel(parcelset::locality const & dest, parcel& p)
        {
            put_parcel(dest, p
              , boost::bind(
                    &parcelport::early_write_handler
                  , this
                  , ::_1
                  , p
                )
            );
        }

        util::io_service_pool* get_thread_pool(char const* name)
        {
            return 0;
        }

        // This parcelport doesn't maintain a connection cache
        boost::int64_t get_connection_cache_statistics(
            connection_cache_statistics_type, bool reset)
        {
            return 0;
        }

        void remove_from_connection_cache(parcelset::locality const& loc)
        {}

        bool run(bool blocking = true)
        {
            receiver_.run();
            receive_early_parcels_thread_ =
                boost::thread(&parcelport::receive_early_parcels, this,
                    hpx::get_runtime_ptr());
            return true;
        }

        void stop(bool blocking = true)
        {
            stopped_ = true;
            sender_.stop();
            receiver_.stop();
            if(blocking)
            {
                MPI_Barrier(util::mpi_environment::communicator());
            }
        }

        void enable(bool new_state)
        {
        }

        void put_parcel(parcelset::locality const & dest, parcel p,
            write_handler_type f)
        {
            if(stopped_) return;

            util::high_resolution_timer timer;

            allocator_type alloc(chunk_pool_);
            snd_buffer_type buffer(alloc);
            encode_parcels(&p, std::size_t(-1), buffer, archive_flags_, this->get_max_outbound_message_size());

            buffer.data_point_.time_ = timer.elapsed_nanoseconds();

            int dest_rank = dest.get<locality>().rank();
            HPX_ASSERT(dest_rank != util::mpi_environment::rank());

            sender_.send(
                dest_rank
              , buffer
              , hpx::util::bind(&receiver::receive, boost::ref(receiver_), false)
            );

            error_code ec;
            f(ec, p);
            buffer.data_point_.time_ =
                timer.elapsed_nanoseconds() - buffer.data_point_.time_;
            parcels_sent_.add_data(buffer.data_point_);

            //do_background_work();
        }

        bool do_background_work(std::size_t num_thread)
        {
            if (stopped_)
                return false;
            return receiver_.receive();
        }

    private:
        typedef util::memory_chunk_pool<> memory_pool_type;
        typedef util::detail::memory_chunk_pool_allocator<char> allocator_type;
        typedef
            std::vector<char, allocator_type>
            data_type;
        typedef parcel_buffer<data_type> snd_buffer_type;
        typedef parcel_buffer<data_type, data_type> rcv_buffer_type;
        typedef lcos::local::spinlock mutex_type;

        int archive_flags_;

        boost::atomic<bool> stopped_;
        boost::atomic<bool> bootstrapping_;
        std::size_t const max_connections_;

        memory_pool_type chunk_pool_;

        mutex_type connections_mtx_;
        lcos::local::detail::condition_variable connections_cond_;
        sender sender_;
        receiver receiver_;

        boost::thread receive_early_parcels_thread_;

        void receive_early_parcels(hpx::runtime * rt)
        {
            rt->register_thread("receive_early_parcel");
            try
            {
                while(rt->get_state() <= runtime::state_startup)
                {
                    do_background_work(0);
                }
            }
            catch(...)
            {
                rt->unregister_thread();
                throw;
            }
            rt->unregister_thread();
        }

        void early_write_handler(
            boost::system::error_code const& ec, parcel const & p)
        {
            if (ec) {
                // all errors during early parcel handling are fatal
                try {
                    HPX_THROW_EXCEPTION(network_error, "early_write_handler",
                        "error while handling early parcel: " +
                            ec.message() + "(" +
                            boost::lexical_cast<std::string>(ec.value())+ ")");
                }
                catch (hpx::exception const& e) {
                    hpx::detail::report_exception_and_terminate(e);
                }
                return;
            }
        }
    };
}}}}

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.mpi]
    //      ...
    //      priority = 100
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::mpi::parcelport>
    {
        static char const* priority()
        {
            return "100";
        }
        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
            util::mpi_environment::init(argc, argv, cfg);
        }

        static char const* call()
        {
            return
#if defined(HPX_PARCELPORT_MPI_ENV)
                "env = ${HPX_PARCELPORT_MPI_ENV:" HPX_PARCELPORT_MPI_ENV "}\n"
#else
                "env = ${HPX_PARCELPORT_MPI_ENV:MV2_COMM_WORLD_RANK,PMI_RANK,OMPI_COMM_WORLD_SIZE,ALPS_APP_PE}\n"
#endif
                "multithreaded = ${HPX_PARCELPORT_MPI_MULTITHREADED:1}\n"
                "max_connections = ${HPX_PARCELPORT_MPI_MAX_CONNECTIONS:8192}\n"
                ;
        }
    };
}}

HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::mpi::parcelport,
    mpi);
