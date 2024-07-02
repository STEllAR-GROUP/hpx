//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/config/warnings_prefix.hpp>

#include <hpx/parcelport_lci/config.hpp>
#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/completion_manager_base.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sender_base.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx::parcelset {
    namespace policies::lci {
        class HPX_EXPORT parcelport;
    }    // namespace policies::lci

    template <>
    struct connection_handler_traits<policies::lci::parcelport>
    {
        using connection_type = policies::lci::sender_connection_base;
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
            using sender_type = sender_base;
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

            std::shared_ptr<sender_connection_base> create_connection(
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

            bool send_immediate(parcelset::parcelport* pp,
                parcelset::locality const& dest,
                sender_base::parcel_buffer_type buffer,
                sender_base::callback_fn_type&& callbackFn);

        private:
            using mutex_type = hpx::spinlock;
            std::atomic<bool> stopped_;
            std::shared_ptr<sender_base> sender_p;
            std::shared_ptr<receiver_base> receiver_p;

            void io_service_work();

        public:
            // States
            // whether the parcelport has been initialized
            // (starting to execute the background works)
            std::atomic<bool> is_initialized = false;

            // LCI objects
            struct completion_manager_t;
            struct device_t
            {
                // These are all pointers to the real data structure allocated
                // by LCI. They would not be modified once initialized.
                // So we should not have false sharing here.
                int idx;
                LCI_device_t device;
                LCI_endpoint_t endpoint_new;
                LCI_endpoint_t endpoint_followup;
                completion_manager_t* completion_manager_p;
            };
            std::vector<device_t> devices;

            // Parcelport objects
            static std::atomic<bool> prg_thread_flag;
            std::unique_ptr<std::thread> prg_thread_p;
            struct completion_manager_t
            {
                std::shared_ptr<completion_manager_base> send;
                std::shared_ptr<completion_manager_base> recv_new;
                std::shared_ptr<completion_manager_base> recv_followup;
            };
            std::vector<completion_manager_t> completion_managers;

            bool do_progress_local();
            device_t& get_tls_device();

        private:
            static void progress_thread_fn(
                const std::vector<device_t>& devices);

            void setup(util::runtime_configuration const& rtcfg);
            void cleanup();

            void join_prg_thread_if_running();
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
                parcelset::policies::lci::config_t::init_config(cfg.rtcfg_);
                cfg.num_localities_ =
                    static_cast<std::size_t>(util::lci_environment::size());
                cfg.node_ =
                    static_cast<std::size_t>(util::lci_environment::rank());
            }
        }

        static void init(hpx::resource::partitioner& rp) noexcept
        {
            if (util::lci_environment::enabled() &&
                parcelset::policies::lci::config_t::progress_type ==
                    parcelset::policies::lci::config_t::progress_type_t::rp)
            {
                if (!parcelset::policies::lci::config_t::is_initialized)
                {
                    fprintf(stderr,
                        "init_config hasn't been called! Something is "
                        "wrong!\n");
                    exit(1);
                }
                rp.create_thread_pool("lci-progress-pool",
                    hpx::resource::scheduling_policy::static_,
                    hpx::threads::policies::scheduler_mode::
                        do_background_work_only);

                size_t npus_to_add =
                    parcelset::policies::lci::config_t::progress_thread_num;
                std::vector<const hpx::resource::pu*> pus;
                for (auto& numa_domain : rp.numa_domains())
                {
                    for (auto& core : numa_domain.cores())
                    {
                        for (auto& pu : core.pus())
                            pus.push_back(&pu);
                    }
                }
                if (pus.size() <= 1)
                {
                    fprintf(stderr, "We don't have enough pus!\n");
                    exit(1);
                }
                if ((size_t) npus_to_add > pus.size() / 2)
                {
                    npus_to_add = pus.size() / 2;
                }
                for (size_t i = 0; i < npus_to_add; ++i)
                {
                    size_t next_pu = i * pus.size() / npus_to_add;
                    rp.add_resource(*pus[next_pu], "lci-progress-pool");
                }
            }
        }

        static void destroy() {}

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
                "log_level = none\n"
                "log_outfile = stderr\n"
                "sendimm = 1\n"
                "backlog_queue = 0\n"
                "prg_thread_num = 1\n"
                "protocol = putsendrecv\n"
                "comp_type_header = queue\n"
                "comp_type_followup = queue\n"
                "progress_type = rp\n"
                "prepost_recv_num = 1\n"
                "reg_mem = 1\n"
                "ndevices = 1\n"
                "ncomps = 1\n"
                "enable_in_buffer_assembly = 1\n"
                "send_nb_max_retry = 32\n"
                "mbuffer_alloc_max_retry = 32\n"
                "bg_work_max_count = 32\n"
                "bg_work_when_send = 0\n"
                "enable_sendmc = 0\n"
                "comp_type = deprecated\n";
        }
    };
}    // namespace hpx::traits

#endif
