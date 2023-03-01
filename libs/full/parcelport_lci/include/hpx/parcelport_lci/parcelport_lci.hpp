//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/config/warnings_prefix.hpp>

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/receiver.hpp>
#include <hpx/parcelport_lci/sender.hpp>
#include <hpx/parcelport_lci/sender_connection.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

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
        using send_immediate_parcels = std::true_type;
        using is_connectionless = std::true_type;

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
        class HPX_EXPORT parcelport : public parcelport_impl<parcelport>
        {
            using base_type = parcelport_impl<parcelport>;
            static parcelset::locality here();
            static std::size_t max_connections(
                util::runtime_configuration const& ini);

        public:
            using sender_type = sender;
            parcelport(util::runtime_configuration const& ini,
                threads::policies::callback_notifier const& notifier);

            ~parcelport();

            void initialized() override;
            // Start the handling of connections.
            bool do_run();

            // Stop the handling of connections.
            void do_stop();

            /// Return the name of this locality
            std::string get_locality_name() const override;

            std::shared_ptr<sender_connection> create_connection(
                parcelset::locality const& l, error_code&);

            parcelset::locality agas_locality(
                util::runtime_configuration const&) const override;

            parcelset::locality create_locality() const override;

            void send_early_parcel(
                hpx::parcelset::locality const& dest, parcel p) override;

            bool do_background_work(std::size_t num_thread,
                parcelport_background_mode mode) override;

            bool background_work(
                std::size_t num_thread, parcelport_background_mode mode);

            bool can_send_immediate();

            // whether the parcelport is sending early parcels
            static bool is_sending_early_parcel;

        private:
            using mutex_type = hpx::spinlock;

            std::atomic<bool> stopped_;

            sender sender_;
            receiver<parcelport> receiver_;

            void io_service_work();
        };
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

        static void init(int*, char***, util::command_line_handling& cfg)
        {
            if (util::lci_environment::enabled())
            {
                util::lci_environment::setup(cfg.rtcfg_);
                cfg.num_localities_ =
                    static_cast<std::size_t>(util::lci_environment::size());
                cfg.node_ =
                    static_cast<std::size_t>(util::lci_environment::rank());
            }
        }

        static void init(hpx::resource::partitioner& rp) noexcept
        {
            if (util::lci_environment::enabled() &&
                util::lci_environment::enable_lci_progress_pool)
            {
                rp.create_thread_pool("lci-progress-pool-eager",
                    hpx::resource::scheduling_policy::static_,
                    hpx::threads::policies::scheduler_mode::
                        do_background_work_only);
                if (util::lci_environment::use_two_device)
                    rp.create_thread_pool("lci-progress-pool-iovec",
                        hpx::resource::scheduling_policy::static_,
                        hpx::threads::policies::scheduler_mode::
                            do_background_work_only);
                rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0],
                    "lci-progress-pool-eager");
                if (util::lci_environment::use_two_device)
                    rp.add_resource(rp.numa_domains()[0].cores()[1].pus()[0],
                        "lci-progress-pool-iovec");
            }
        }

        static void destroy()
        {
            util::lci_environment::finalize();
        }

        static constexpr char const* call() noexcept
        {
            return
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
                "rp_prg_pool = 1\n"
                "backlog_queue = 0\n"
                "use_two_device = 0\n"
                "prg_thread_core = -1\n";
        }
    };
}    // namespace hpx::traits

#endif
