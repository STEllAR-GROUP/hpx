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

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver.hpp>
#include <hpx/parcelport_lci/sender.hpp>
#include <hpx/parcelport_lci/sender_connection.hpp>

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
      , receiver_(*this)
    {
    }

    parcelport::~parcelport()
    {
        util::lci_environment::finalize();
    }

    void parcelport::initialized()
    {
        if (util::lci_environment::enabled() &&
            util::lci_environment::enable_lci_progress_pool)
        {
            util::lci_environment::join_prg_thread_if_running();
        }
    }

    // Start the handling of connections.
    bool parcelport::do_run()
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
        LCI_barrier();
    }

    /// Return the name of this locality
    std::string parcelport::get_locality_name() const
    {
        // hostname-rank
        return util::lci_environment::get_processor_name() + "-" +
            std::to_string(util::lci_environment::rank());
    }

    std::shared_ptr<sender_connection> parcelport::create_connection(
        parcelset::locality const& l, error_code&)
    {
        int dest_rank = l.get<locality>().rank();
        return sender_.create_connection(dest_rank, this);
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
            if (util::lci_environment::enable_lci_progress_pool &&
                hpx::threads::get_self_id() != hpx::threads::invalid_thread_id)
            {
                if (hpx::this_thread::get_pool() ==
                    &hpx::resource::get_thread_pool("lci-progress-pool-eager"))
                    do_lci_progress = 1;
                else if (util::lci_environment::use_two_device &&
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
                while (util::lci_environment::do_progress(
                    util::lci_environment::get_device_eager()))
                {
                    has_work = true;
                    idle_loop_count = 0;
                }
                ++idle_loop_count;
            }
        }
        else if (do_lci_progress == 2)
        {
            // magic number
            int idle_loop_count = 0;
            while (idle_loop_count < max_idle_loop_count)
            {
                while (util::lci_environment::do_progress(
                    util::lci_environment::get_device_iovec()))
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

        bool has_work;
        if (mode & parcelport_background_mode_send)
        {
            has_work = sender_.background_work(num_thread);
            // try to send pending messages
            has_work = backlog_queue::background_work(num_thread) || has_work;
        }
        if (mode & parcelport_background_mode_receive)
        {
            has_work = receiver_.background_work() || has_work;
        }
        return has_work;
    }

    bool parcelport::can_send_immediate()
    {
        return util::lci_environment::enable_send_immediate;
    }

    void parcelport::io_service_work()
    {
        std::size_t k = 0;
        // We only execute work on the IO service while HPX is starting
        // TODO: stop io service work when worker threads start to execute the
        //  background function
        while (hpx::is_starting())
        {
            bool has_work = sender_.background_work(0);
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
    bool parcelport::is_sending_early_parcel = false;
}    // namespace hpx::parcelset::policies::lci

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::lci::parcelport, lci)

#endif
