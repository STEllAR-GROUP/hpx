//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelset_base/parcelport.hpp>

#include <hpx/parcelport_lci/config.hpp>
#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/completion_manager/completion_manager_queue.hpp>
#include <hpx/parcelport_lci/completion_manager/completion_manager_sync.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/putva/receiver_putva.hpp>
#include <hpx/parcelport_lci/putva/sender_putva.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sender_base.hpp>
#include <hpx/parcelport_lci/sender_connection_base.hpp>
#include <hpx/parcelport_lci/sendrecv/receiver_sendrecv.hpp>
#include <hpx/parcelport_lci/sendrecv/sender_sendrecv.hpp>

#include <hpx/assert.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace hpx::parcelset::policies::lci {
    parcelset::locality parcelport::here()
    {
        return parcelset::locality(locality(util::lci_environment::enabled() ?
                util::lci_environment::rank() :
                -1));
    }

    std::size_t parcelport::max_connections(
        util::runtime_configuration const& ini)
    {
        return hpx::util::get_entry_as<std::size_t>(
            ini, "hpx.parcel.lci.max_connections", HPX_PARCEL_MAX_CONNECTIONS);
    }

    parcelport::parcelport(util::runtime_configuration const& ini,
        threads::policies::callback_notifier const& notifier)
      : parcelport::base_type(ini, here(), notifier)
      , stopped_(false)
    {
        if (!util::lci_environment::enabled())
            return;
        if (!parcelset::policies::lci::config_t::is_initialized)
        {
            fprintf(stderr,
                "init_config hasn't been called! Something is wrong!\n");
            exit(1);
        }
        setup(ini);
    }

    parcelport::~parcelport()
    {
        if (!util::lci_environment::enabled())
            return;
        cleanup();
        util::lci_environment::finalize();
    }

    void parcelport::initialized()
    {
        if (util::lci_environment::enabled() &&
            config_t::progress_type != config_t::progress_type_t::pthread)
        {
            join_prg_thread_if_running();
        }
        is_initialized = true;
    }

    // Start the handling of connections.
    bool parcelport::do_run()
    {
        receiver_p->run();
        sender_p->run();
        for (std::size_t i = 0; i != io_service_pool_.size(); ++i)
        {
            io_service_pool_.get_io_service(int(i)).post(
                hpx::bind(&parcelport::io_service_work, this));
        }
        return true;
    }

    // Stop the handling of connections.
    void parcelport::do_stop()
    {
        while (do_background_work(0, parcelport_background_mode_all))
        {
            if (threads::get_self_ptr())
                hpx::this_thread::suspend(
                    hpx::threads::thread_schedule_state::pending,
                    "lci::parcelport::do_stop");
        }
        stopped_ = true;
    }

    /// Return the name of this locality
    std::string parcelport::get_locality_name() const
    {
        // hostname-rank
        return util::lci_environment::get_processor_name() + "-" +
            std::to_string(util::lci_environment::rank());
    }

    std::shared_ptr<sender_connection_base> parcelport::create_connection(
        parcelset::locality const& l, error_code&)
    {
        int dest_rank = l.get<locality>().rank();
        return sender_p->create_connection(dest_rank, this);
    }

    parcelset::locality parcelport::agas_locality(
        util::runtime_configuration const&) const
    {
        return parcelset::locality(
            locality(util::lci_environment::enabled() ? 0 : -1));
    }

    parcelset::locality parcelport::create_locality() const
    {
        return parcelset::locality(locality());
    }

    void parcelport::send_early_parcel(
        hpx::parcelset::locality const& dest, parcel p)
    {
        is_sending_early_parcel = true;
        base_type::send_early_parcel(dest, HPX_MOVE(p));
        is_sending_early_parcel = false;
    }

    bool parcelport::do_background_work(
        std::size_t num_thread, parcelport_background_mode mode)
    {
        static thread_local int do_lci_progress = -1;
        if (do_lci_progress == -1)
        {
            do_lci_progress = 0;
            if (config_t::progress_type == config_t::progress_type_t::rp &&
                hpx::threads::get_self_id() != hpx::threads::invalid_thread_id)
            {
                if (hpx::this_thread::get_pool() ==
                    &hpx::resource::get_thread_pool("lci-progress-pool-eager"))
                    do_lci_progress = 1;
                else if (config_t::use_two_device &&
                    hpx::this_thread::get_pool() ==
                        &hpx::resource::get_thread_pool(
                            "lci-progress-pool-iovec"))
                    do_lci_progress = 2;
            }
        }

        bool has_work = false;
        // magic number
        const int max_idle_loop_count = 1000;
        if (do_lci_progress == 1)
        {
            int idle_loop_count = 0;
            while (idle_loop_count < max_idle_loop_count)
            {
                while (util::lci_environment::do_progress(device_eager))
                {
                    has_work = true;
                    idle_loop_count = 0;
                }
                ++idle_loop_count;
            }
        }
        else if (do_lci_progress == 2)
        {
            int idle_loop_count = 0;
            while (idle_loop_count < max_idle_loop_count)
            {
                while (util::lci_environment::do_progress(device_iovec))
                {
                    has_work = true;
                    idle_loop_count = 0;
                }
                ++idle_loop_count;
            }
        }
        else
        {
            has_work = base_type::do_background_work(num_thread, mode);
        }
        return has_work;
    }

    bool parcelport::background_work(
        std::size_t num_thread, parcelport_background_mode mode)
    {
        if (stopped_)
            return false;

        bool has_work = false;
        if (mode & parcelport_background_mode_send)
        {
            has_work = sender_p->background_work(num_thread);
            if (config_t::progress_type == config_t::progress_type_t::worker)
                do_progress();
            if (config_t::enable_lci_backlog_queue)
                // try to send pending messages
                has_work = backlog_queue::background_work(
                               send_completion_manager.get(), num_thread) ||
                    has_work;
        }
        if (mode & parcelport_background_mode_receive)
        {
            has_work = receiver_p->background_work() || has_work;
            if (config_t::progress_type == config_t::progress_type_t::worker)
                do_progress();
        }
        return has_work;
    }

    bool parcelport::can_send_immediate()
    {
        return config_t::enable_send_immediate;
    }

    bool parcelport::send_immediate(parcelset::parcelport* pp,
        parcelset::locality const& dest, sender_base::parcel_buffer_type buffer,
        sender_base::callback_fn_type&& callbackFn)
    {
        return sender_p->send_immediate(pp, dest,
            HPX_FORWARD(sender_base::parcel_buffer_type, buffer),
            HPX_FORWARD(sender_base::callback_fn_type, callbackFn));
    }

    void parcelport::io_service_work()
    {
        std::size_t k = 0;
        // We only execute work on the IO service while HPX is starting
        // and stop io service work when worker threads start to execute the
        // background function
        while (hpx::is_starting() && !is_initialized)
        {
            bool has_work = sender_p->background_work(0);
            has_work = receiver_p->background_work() || has_work;
            if (config_t::progress_type == config_t::progress_type_t::worker)
                while (do_progress())
                    continue;
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

    std::atomic<bool> parcelport::prg_thread_flag = false;
    void parcelport::progress_thread_fn(LCI_device_t device)
    {
        while (prg_thread_flag)
        {
            util::lci_environment::do_progress(device);
        }
    }

    void set_affinity(pthread_t pthread_handler, size_t target)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target, &cpuset);
        int rv =
            pthread_setaffinity_np(pthread_handler, sizeof(cpuset), &cpuset);
        if (rv != 0)
        {
            fprintf(stderr, "ERROR %d thread affinity didn't work.\n", rv);
            exit(1);
        }
    }

    void parcelport::setup(util::runtime_configuration const& rtcfg)
    {
        HPX_UNUSED(rtcfg);
        // Create device
        device_eager = LCI_UR_DEVICE;
        if (config_t::use_two_device)
            LCI_device_init(&device_iovec);
        else
            device_iovec = LCI_UR_DEVICE;

        // Create completion objects
        if (config_t::protocol == config_t::protocol_t::sendrecv &&
            config_t::completion_type == LCI_COMPLETION_SYNC)
        {
            recv_new_completion_manager =
                std::make_shared<completion_manager_sync>();
        }
        else
        {
            recv_new_completion_manager =
                std::make_shared<completion_manager_queue>();
        }
        switch (config_t::completion_type)
        {
        case LCI_COMPLETION_QUEUE:
            send_completion_manager =
                std::make_shared<completion_manager_queue>();
            recv_followup_completion_manager =
                std::make_shared<completion_manager_queue>();
            break;
        case LCI_COMPLETION_SYNC:
            send_completion_manager =
                std::make_shared<completion_manager_sync>();
            recv_followup_completion_manager =
                std::make_shared<completion_manager_sync>();
            break;
        default:
            throw std::runtime_error("Unknown completion type!");
        }

        // Create endpoints
        LCI_plist_t plist_;
        LCI_plist_create(&plist_);
        LCI_plist_set_comp_type(
            plist_, LCI_PORT_COMMAND, config_t::completion_type);
        LCI_plist_set_comp_type(
            plist_, LCI_PORT_MESSAGE, config_t::completion_type);
        LCI_endpoint_init(&endpoint_followup, device_eager, plist_);
        LCI_plist_set_default_comp(
            plist_, recv_new_completion_manager->get_completion_object());
        if (config_t::protocol == config_t::protocol_t::sendrecv &&
            config_t::completion_type == LCI_COMPLETION_SYNC)
            LCI_plist_set_comp_type(
                plist_, LCI_PORT_MESSAGE, LCI_COMPLETION_SYNC);
        else
        {
            LCI_plist_set_comp_type(
                plist_, LCI_PORT_MESSAGE, LCI_COMPLETION_QUEUE);
        }
        if (config_t::protocol == config_t::protocol_t::sendrecv)
            LCI_plist_set_match_type(plist_, LCI_MATCH_TAG);
        LCI_endpoint_init(&endpoint_new_eager, device_eager, plist_);
        LCI_endpoint_init(&endpoint_new_iovec, device_iovec, plist_);
        LCI_plist_free(&plist_);

        // Create progress threads
        HPX_ASSERT(prg_thread_flag == false);
        HPX_ASSERT(prg_thread_eager_p == nullptr);
        HPX_ASSERT(prg_thread_iovec_p == nullptr);
        prg_thread_flag = true;
        prg_thread_eager_p =
            std::make_unique<std::thread>(progress_thread_fn, device_eager);
        if (config_t::use_two_device)
            prg_thread_iovec_p =
                std::make_unique<std::thread>(progress_thread_fn, device_iovec);
        if (config_t::progress_thread_core >= 0)
        {
            set_affinity(prg_thread_eager_p->native_handle(),
                config_t::progress_thread_core);
            if (config_t::use_two_device)
                set_affinity(prg_thread_iovec_p->native_handle(),
                    config_t::progress_thread_core + 1);
        }

        // Create the sender and receiver
        switch (config_t::protocol)
        {
        case config_t::protocol_t::putva:
            sender_p = std::make_shared<sender_putva>(this);
            receiver_p = std::make_shared<receiver_putva>(this);
            break;
        case config_t::protocol_t::sendrecv:
        case config_t::protocol_t::putsendrecv:
            sender_p = std::make_shared<sender_sendrecv>(this);
            receiver_p = std::make_shared<receiver_sendrecv>(this);
            break;
        default:
            throw std::runtime_error("Unknown Protocol!");
        }
    }

    void parcelport::cleanup()
    {
        join_prg_thread_if_running();
        // free ep, rcq
        LCI_endpoint_free(&endpoint_new_iovec);
        LCI_endpoint_free(&endpoint_followup);
        LCI_endpoint_free(&endpoint_new_eager);
        if (config_t::use_two_device)
            LCI_device_free(&device_iovec);
    }

    void parcelport::join_prg_thread_if_running()
    {
        if (prg_thread_eager_p || prg_thread_iovec_p)
        {
            prg_thread_flag = false;
            if (prg_thread_eager_p)
            {
                prg_thread_eager_p->join();
                prg_thread_eager_p.reset();
            }
            if (prg_thread_iovec_p)
            {
                prg_thread_iovec_p->join();
                prg_thread_iovec_p.reset();
            }
        }
    }

    bool parcelport::do_progress()
    {
        bool ret = false;
        ret = util::lci_environment::do_progress(device_eager) || ret;
        if (config_t::use_two_device)
            ret = util::lci_environment::do_progress(device_iovec) || ret;
        return ret;
    }
}    // namespace hpx::parcelset::policies::lci

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::lci::parcelport, lci)

#endif
