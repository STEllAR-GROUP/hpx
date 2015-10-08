//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>
#include <hpx/config/warnings_prefix.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#endif

#include <hpx/hpx_fwd.hpp>

#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>
#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/util/command_line_handling.hpp>

// parcelport
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/condition_variable.hpp>

#include <hpx/plugins/parcelport/mpi/locality.hpp>
#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/plugins/parcelport/mpi/sender.hpp>
#include <hpx/plugins/parcelport/mpi/receiver.hpp>

#include <hpx/util/memory_chunk_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/archive/basic_archive.hpp>

namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset
{
    namespace policies { namespace mpi
    {
        class HPX_EXPORT parcelport;
    }}

    template <>
    struct connection_handler_traits<policies::mpi::parcelport>
    {
        typedef policies::mpi::sender_connection connection_type;
        typedef boost::mpl::true_  send_early_parcel;
        typedef boost::mpl::true_ do_background_work;

        static const char * type()
        {
            return "mpi";
        }

        static const char * pool_name()
        {
            return "parcel-pool-mpi";
        }

        static const char * pool_name_postfix()
        {
            return "-mpi";
        }
    };

    namespace policies { namespace mpi
    {
        int acquire_tag(sender * s)
        {
            return s->acquire_tag();
        }

        void add_connection(sender * s, boost::shared_ptr<sender_connection> const &ptr)
        {
            s->add(ptr);
        }

        class HPX_EXPORT parcelport
          : public parcelport_impl<parcelport>
        {
            typedef parcelport_impl<parcelport> base_type;

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
                util::function_nonser<void(std::size_t, char const*)> const& on_start,
                util::function_nonser<void()> const& on_stop)
              : base_type(ini, here(), on_start, on_stop)
              , stopped_(false)
              , chunk_pool_(4096, max_connections(ini))
              , sender_(chunk_pool_)
              , receiver_(*this, chunk_pool_)
              , handles_parcels_(0)
            {}

            ~parcelport()
            {
                if(receive_early_parcels_thread_.joinable())
                    receive_early_parcels_thread_.join();
                util::mpi_environment::finalize();
            }

            /// Start the handling of connections.
            bool do_run()
            {
                receiver_.run();
                sender_.run();
                receive_early_parcels_thread_ =
                    boost::thread(&parcelport::receive_early_parcels, this,
                        hpx::get_runtime_ptr());
                return true;
            }

            /// Stop the handling of connectons.
            void do_stop()
            {
                while(do_background_work(0))
                {
                    if(threads::get_self_ptr())
                        hpx::this_thread::suspend(hpx::threads::pending,
                            "mpi::parcelport::do_stop");
                }
                stopped_ = true;
                while(handles_parcels_ != 0)
                {
                    if(threads::get_self_ptr())
                        hpx::this_thread::suspend(hpx::threads::pending,
                            "mpi::parcelport::do_stop");
                }
                MPI_Barrier(util::mpi_environment::communicator());
            }

            /// Return the name of this locality
            std::string get_locality_name() const
            {
                return util::mpi_environment::get_processor_name();
            }

            boost::shared_ptr<sender_connection> create_connection(
                parcelset::locality const& l, error_code& ec)
            {
                int dest_rank = l.get<locality>().rank();
                return sender_.create_connection(
                    dest_rank, parcels_sent_);
            }

            parcelset::locality agas_locality(
                util::runtime_configuration const & ini) const
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

            bool background_work(std::size_t num_thread)
            {
                if (stopped_)
                    return false;

                handles_parcels h(this);

                bool has_work = sender_.background_work(num_thread);
                has_work = receiver_.background_work(num_thread) || has_work;
                return has_work;
            }

        private:
            typedef util::memory_chunk_pool<> memory_pool_type;
            typedef
                util::detail::memory_chunk_pool_allocator<
                    char, util::memory_chunk_pool<>
                > allocator_type;
            typedef
                std::vector<char, allocator_type>
                data_type;
            typedef lcos::local::spinlock mutex_type;

            boost::atomic<bool> stopped_;

            memory_pool_type chunk_pool_;

            sender sender_;
            receiver receiver_;

            boost::thread receive_early_parcels_thread_;

            boost::atomic<std::size_t> handles_parcels_;

            struct handles_parcels
            {
                handles_parcels(parcelport *pp)
                  : this_(pp)
                {
                    ++this_->handles_parcels_;
                }

                ~handles_parcels()
                {
                    --this_->handles_parcels_;
                }

                parcelport *this_;
            };

            void receive_early_parcels(hpx::runtime * rt)
            {
                rt->register_thread("receive_early_parcel");
                try
                {
                    while(rt->get_state() <= state_startup)
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
                    boost::exception_ptr exception =
                        hpx::detail::get_exception(hpx::exception(ec),
                            "mpi::early_write_handler", __FILE__, __LINE__,
                            "error while handling early parcel: " +
                                ec.message() + "(" +
                                boost::lexical_cast<std::string>(ec.value()) +
                                ")" + parcelset::dump_parcel(p));

                    hpx::report_error(exception);
                }
            }

        };
    }}
}}

#include <hpx/config/warnings_suffix.hpp>

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
#if defined(HPX_HAVE_PARCELPORT_MPI_ENV)
                "env = ${HPX_HAVE_PARCELPORT_MPI_ENV:" HPX_HAVE_PARCELPORT_MPI_ENV "}\n"
#else
                "env = ${HPX_HAVE_PARCELPORT_MPI_ENV:"
                        "MV2_COMM_WORLD_RANK,PMI_RANK,OMPI_COMM_WORLD_SIZE,ALPS_APP_PE"
                    "}\n"
#endif
                "multithreaded = ${HPX_HAVE_PARCELPORT_MPI_MULTITHREADED:0}\n"
                "max_connections = ${HPX_HAVE_PARCELPORT_MPI_MAX_CONNECTIONS:8192}\n"
                ;
        }
    };
}}

HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::mpi::parcelport,
    mpi);
