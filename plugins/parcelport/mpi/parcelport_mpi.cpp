//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/traits/plugin_config_data.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#endif

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

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/atomic.hpp>
#include <boost/archive/basic_archive.hpp>
#include <boost/exception_ptr.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

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
        typedef std::true_type  send_early_parcel;
        typedef std::true_type  do_background_work;
        typedef std::true_type  use_connection_cache;

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

        void add_connection(sender * s, std::shared_ptr<sender_connection> const &ptr)
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
              , receiver_(*this)
            {}

            ~parcelport()
            {
                util::mpi_environment::finalize();
            }

            /// Start the handling of connections.
            bool do_run()
            {
                receiver_.run();
                sender_.run();
                for(std::size_t i = 0; i != io_service_pool_.size(); ++i)
                {
                    io_service_pool_.get_io_service(int(i)).post(
                        hpx::util::bind(
                            &parcelport::io_service_work, this
                        )
                    );
                }
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
                MPI_Barrier(util::mpi_environment::communicator());
            }

            /// Return the name of this locality
            std::string get_locality_name() const
            {
                return util::mpi_environment::get_processor_name();
            }

            std::shared_ptr<sender_connection> create_connection(
                parcelset::locality const& l, error_code& ec)
            {
                int dest_rank = l.get<locality>().rank();
                return sender_.create_connection(dest_rank, this);
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

                bool has_work = false;
                has_work = sender_.background_work();
                has_work = receiver_.background_work() || has_work;
                return has_work;
            }

        private:
            typedef lcos::local::spinlock mutex_type;

            boost::atomic<bool> stopped_;

            sender sender_;
            receiver<parcelport> receiver_;

            void io_service_work()
            {
                std::size_t k = 0;
                // We only execute work on the IO service while HPX is starting
                while(hpx::is_starting())
                {
                    bool has_work = sender_.background_work();
                    has_work = receiver_.background_work() || has_work;
                    if(has_work)
                    {
                        k = 0;
                    }
                    else
                    {
                        ++k;
                        hpx::lcos::local::spinlock::yield(k);
                    }
                }
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
                                std::to_string(ec.value()) +
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
