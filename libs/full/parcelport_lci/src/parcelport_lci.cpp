//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/util.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>

#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/receiver.hpp>
#include <hpx/parcelport_lci/sender.hpp>
#include <hpx/parcelset/parcelport_impl.hpp>
#include <hpx/parcelset_base/locality.hpp>
#include <hpx/plugin_factories/parcelport_factory.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::parcelset {

    namespace policies::lci {
        class HPX_EXPORT parcelport;
    }    // namespace policies::lci

    template <>
    struct connection_handler_traits<policies::lci::parcelport>
    {
        using connection_type = policies::lci::sender_connection;
        using send_early_parcel = std::true_type;
        using do_background_work = std::true_type;
        using send_immediate_parcels = std::false_type;

        static constexpr const char* type() noexcept
        {
            return "lci";
        }

        static constexpr const char* pool_name() noexcept
        {
            return "parcel-pool-lci";
        }

        static constexpr const char* pool_name_postfix() noexcept
        {
            return "-lci";
        }
    };

    namespace policies::lci {
        int acquire_tag(sender* s) noexcept
        {
            return s->acquire_tag();
        }

        void add_connection(
            sender* s, std::shared_ptr<sender_connection> const& ptr)
        {
            s->add(ptr);
        }

        class HPX_EXPORT parcelport : public parcelport_impl<parcelport>
        {
            using base_type = parcelport_impl<parcelport>;

            static parcelset::locality here()
            {
                return parcelset::locality(
                    locality(util::lci_environment::enabled() ?
                            util::lci_environment::rank() :
                            -1));
            }

            static std::size_t max_connections(
                util::runtime_configuration const& ini)
            {
                return hpx::util::get_entry_as<std::size_t>(ini,
                    "hpx.parcel.lci.max_connections",
                    HPX_PARCEL_MAX_CONNECTIONS);
            }

        public:
            parcelport(util::runtime_configuration const& ini,
                threads::policies::callback_notifier const& notifier)
              : base_type(ini, here(), notifier)
              , stopped_(false)
              , receiver_(*this)
            {
            }

            ~parcelport()
            {
                util::lci_environment::finalize();
            }

            // Start the handling of connections.
            bool do_run()
            {
                receiver_.run();
                sender_.run();
                for (std::size_t i = 0; i != io_service_pool_.size(); ++i)
                {
                    io_service_pool_.get_io_service(int(i)).post(
                        hpx::bind(&parcelport::io_service_work, this));
                }
                return true;
            }

            // Stop the handling of connections.
            void do_stop()
            {
                while (do_background_work(0, parcelport_background_mode_all))
                {
                    if (threads::get_self_ptr())
                        hpx::this_thread::suspend(
                            hpx::threads::thread_schedule_state::pending,
                            "lci::parcelport::do_stop");
                }
                stopped_ = true;
                LCI_barrier();
            }

            /// Return the name of this locality
            std::string get_locality_name() const override
            {
                // hostname-rank
                return util::lci_environment::get_processor_name() + "-" +
                    std::to_string(util::lci_environment::rank());
            }

            std::shared_ptr<sender_connection> create_connection(
                parcelset::locality const& l, error_code&)
            {
                int dest_rank = l.get<locality>().rank();
                return sender_.create_connection(dest_rank, this);
            }

            parcelset::locality agas_locality(
                util::runtime_configuration const&) const override
            {
                return parcelset::locality(
                    locality(util::lci_environment::enabled() ? 0 : -1));
            }

            parcelset::locality create_locality() const override
            {
                return parcelset::locality(locality());
            }

            bool background_work(
                std::size_t /* num_thread */, parcelport_background_mode mode)
            {
                if (stopped_)
                    return false;

                static thread_local int do_lci_progress = -1;
                if (do_lci_progress == -1)
                {
                    if (enable_lci_progress_pool &&
                        hpx::threads::get_self_id() !=
                            hpx::threads::invalid_thread_id &&
                        hpx::this_thread::get_pool() ==
                            &hpx::resource::get_thread_pool(
                                "lci-progress-pool"))
                    {
                        do_lci_progress = 1;
                    }
                    else
                    {
                        do_lci_progress = 0;
                    }
                }

                bool has_work = false;
                if (do_lci_progress)
                {
                    util::lci_environment::join_prg_thread_if_running();
                    // magic number
                    int max_idle_loop_count = 1000;
                    int idle_loop_count = 0;
                    while (idle_loop_count < max_idle_loop_count)
                    {
                        while (util::lci_environment::do_progress())
                        {
                            has_work = true;
                            idle_loop_count = 0;
                        }
                        ++idle_loop_count;
                    }
                }
                else
                {
                    if (mode & parcelport_background_mode_send)
                    {
                        has_work = sender_.background_work();
                    }
                    if (mode & parcelport_background_mode_receive)
                    {
                        has_work = receiver_.background_work() || has_work;
                    }
                }
                return has_work;
            }

            static bool enable_lci_progress_pool;

        private:
            using mutex_type = hpx::spinlock;

            std::atomic<bool> stopped_;

            sender sender_;
            receiver<parcelport> receiver_;

            void io_service_work()
            {
                std::size_t k = 0;
                // We only execute work on the IO service while HPX is starting
                while (hpx::is_starting())
                {
                    bool has_work = sender_.background_work();
                    has_work = receiver_.background_work() || has_work;
                    if (has_work)
                    {
                        k = 0;
                    }
                    else
                    {
                        ++k;
                        util::detail::yield_k(k,
                            "hpx::parcelset::policies::lci::parcelport::"
                            "io_service_work");
                    }
                }
            }

            void early_write_handler(std::error_code const& ec, parcel const& p)
            {
                if (ec)
                {
                    // all errors during early parcel handling are fatal
                    std::exception_ptr exception = hpx::detail::get_exception(
                        hpx::exception(ec), "lci::early_write_handler",
                        __FILE__, __LINE__,
                        "error while handling early parcel: " + ec.message() +
                            "(" + std::to_string(ec.value()) + ")" +
                            parcelset::dump_parcel(p));

                    hpx::report_error(exception);
                }
            }
        };
        bool parcelport::enable_lci_progress_pool = false;
    }    // namespace policies::lci
}    // namespace hpx::parcelset

#include <hpx/config/warnings_suffix.hpp>

namespace hpx::traits {
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.lci]
    //      ...
    //      priority = 200
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::lci::parcelport>
    {
        static constexpr char const* priority() noexcept
        {
            return "50";
        }

        static void init(
            int* argc, char*** argv, util::command_line_handling& cfg)
        {
            util::lci_environment::init(argc, argv, cfg.rtcfg_);
            cfg.num_localities_ =
                static_cast<std::size_t>(util::lci_environment::size());
            cfg.node_ = static_cast<std::size_t>(util::lci_environment::rank());
            hpx::parcelset::policies::lci::parcelport::
                enable_lci_progress_pool = hpx::util::get_entry_as<bool>(
                    cfg.rtcfg_, "hpx.parcel.lci.rp_prg_pool",
                    false /* Does not matter*/);
        }

        // TODO: implement creation of custom thread pool here
        static void init(hpx::resource::partitioner& rp) noexcept
        {
            if (util::lci_environment::enabled() &&
                hpx::parcelset::policies::lci::parcelport::
                    enable_lci_progress_pool)
            {
                rp.create_thread_pool("lci-progress-pool",
                    hpx::resource::scheduling_policy::local,
                    hpx::threads::policies::scheduler_mode::do_background_work);
                rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0],
                    "lci-progress-pool");
            }
        }

        static void destroy()
        {
            util::lci_environment::finalize();
        }

        static constexpr char const* call() noexcept
        {
            return
            // TODO: change these for LCI
#if defined(HPX_HAVE_PARCELPORT_LCI_ENV)
                "env = "
                "${HPX_HAVE_PARCELPORT_LCI_ENV:" HPX_HAVE_PARCELPORT_LCI_ENV
                "}\n"
#else
                "env = ${HPX_HAVE_PARCELPORT_LCI_ENV:"
                "MV2_COMM_WORLD_RANK,PMIX_RANK,PMI_RANK,LCI_COMM_WORLD_SIZE,"
                "ALPS_APP_PE,PALS_NODEID"
                "}\n"
#endif
                "max_connections = "
                "${HPX_HAVE_PARCELPORT_LCI_MAX_CONNECTIONS:8192}\n"
                "rp_prg_pool = 0\n";
        }
    };
}    // namespace hpx::traits

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::lci::parcelport, lci)

#endif
