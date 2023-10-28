//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2020 Google
//  Copyright (c)      2022 Patrick Diehl
//  Copyright (c)      2023 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/openshmem_base/openshmem.hpp>
#include <hpx/modules/openshmem_base.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/util.hpp>

#include <mpp/shmem-def.h>
#include <mpp/shmem.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    namespace detail {

        bool detect_openshmem_environment(
            util::runtime_configuration const& cfg, char const* default_env)
        {
            std::string openshmem_environment_strings =
                cfg.get_entry("hpx.parcel.openshmem.env", default_env);

            hpx::string_util::char_separator<char> sep(";,: ");
            hpx::string_util::tokenizer tokens(
                openshmem_environment_strings, sep);
            for (auto const& tok : tokens)
            {
                char* env = std::getenv(tok.c_str());
                if (env)
                {
                    LBT_(debug)
                        << "Found OPENSHMEM environment variable: " << tok
                        << "=" << std::string(env)
                        << ", enabling OPENSHMEM support\n";
                    return true;
                }
            }

            LBT_(info)
                << "No known OPENSHMEM environment variable found, disabling "
                   "OPENSHMEM support\n";
            return false;
        }
    }    // namespace detail

    bool openshmem_environment::check_openshmem_environment(
        [[maybe_unused]] util::runtime_configuration const& cfg)
    {
#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_MODULE_OPENSHMEM_BASE)
        // We disable the OPENSHMEM parcelport if any of these hold:
        //
        // - The parcelport is explicitly disabled
        // - The application is not run in an OPENSHMEM environment
        // - The TCP parcelport is enabled and has higher priority
        //
        if (get_entry_as(cfg, "hpx.parcel.openshmem.enable", 1) == 0 ||
            (get_entry_as(cfg, "hpx.parcel.tcp.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.tcp.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.openshmem.priority", 0))) ||
            (get_entry_as(cfg, "hpx.parcel.openshmem.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.openshmem.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.mpi.priority", 0))))
        {
            LBT_(info)
                << "OpenSHMEM support disabled via configuration settings\n";
            return false;
        }

        return true;
#else
        return false;
#endif
    }
}    // namespace hpx::util

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_MODULE_OPENSHMEM_BASE))

namespace hpx::util {

    hpx::spinlock openshmem_environment::pollingLock{};
    hpx::mutex openshmem_environment::dshm_mut{};
    hpx::mutex openshmem_environment::mtx_{};
    bool openshmem_environment::enabled_ = false;
    bool openshmem_environment::has_called_init_ = false;
    int openshmem_environment::provided_threading_flag_ = 0;
    int openshmem_environment::is_initialized_ = -1;
    int openshmem_environment::init_val_ = 0;
    hpx::mutex* openshmem_environment::segment_mutex = nullptr;
    openshmem_seginfo_t* openshmem_environment::segments = nullptr;
    std::uint8_t* hpx::util::openshmem_environment::shmem_buffer = nullptr;
    unsigned int openshmem_environment::rcv = 0;
    unsigned int openshmem_environment::xmt = 0;

    ///////////////////////////////////////////////////////////////////////////
    int openshmem_environment::init([[maybe_unused]] int* argc,
        [[maybe_unused]] char*** argv, [[maybe_unused]] int& provided)
    {
        if (!has_called_init_)
        {
            shmem_init();
            openshmem_environment::init_val_ = 1;
            has_called_init_ = true;
        }

        if (openshmem_environment::init_val_ == 0)
        {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::init",
                "OPENSHMEM initialization error");
        }
        else if (openshmem_environment::init_val_ == 0)
        {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::init",
                "OPENSHMEM resource error");
        }
        else if (openshmem_environment::init_val_ == 0)
        {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::init",
                "OPENSHMEM bad argument error");
        }
        else if (openshmem_environment::init_val_ == 0)
        {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::init",
                "OPENSHMEM not ready error");
        }

        // create a number of segments equal to the number of hardware
        // threads per machine (locality)
        //
        //segments.resize(hpx::threads::hardware_concurrency() * size());
        //
        openshmem_environment::segments = new openshmem_seginfo_t[size()];
        openshmem_environment::segment_mutex = new hpx::mutex[size()];

        hpx::util::openshmem_environment::shmem_buffer =
            static_cast<std::uint8_t*>(shmem_calloc(
                OPENSHMEM_PER_RANK_PAGESIZE, sizeof(std::uint8_t)));

        for (int i = 0; i < size(); ++i)
        {
            segments[i].addr = hpx::util::openshmem_environment::shmem_buffer +
                (i * OPENSHMEM_PER_RANK_PAGESIZE);
            segments[i].size = static_cast<std::uint8_t*>(segments[i].addr) +
                OPENSHMEM_PER_RANK_PAGESIZE;
        }

        shmem_barrier_all();

        return openshmem_environment::init_val_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void openshmem_environment::init([[maybe_unused]] int* argc,
        [[maybe_unused]] char*** argv,
        [[maybe_unused]] util::runtime_configuration& rtcfg)
    {
        if (enabled_)
            return;    // don't call twice

        int this_rank = -1;
        has_called_init_ = false;

        // We assume to use the OpenSHMEM parcelport if it is not explicitly disabled
        enabled_ = check_openshmem_environment(rtcfg);
        if (!enabled_)
        {
            rtcfg.add_entry("hpx.parcel.openshmem.enable", "0");
            return;
        }

        rtcfg.add_entry("hpx.parcel.bootstrap", "openshmem");

        int retval = init(argc, argv, provided_threading_flag_);
        if (1 != retval)
        {
            // explicitly disable openshmem if not run by openshmemrun
            rtcfg.add_entry("hpx.parcel.openshmem.enable", "0");

            enabled_ = false;

            std::string msg(
                "openshmem_environment::init: openshmem_init failed");
            throw std::runtime_error(msg.c_str());
        }

        if (provided_threading_flag_ != 1)
        {
            // explicitly disable openshmem if not run by openshmemrun
            rtcfg.add_entry("hpx.parcel.openshmem.multithreaded", "0");
        }

        this_rank = rank();

#if defined(HPX_HAVE_NETWORKING)
        if (this_rank == 0)
        {
            rtcfg.mode_ = hpx::runtime_mode::console;
        }
        else
        {
            rtcfg.mode_ = hpx::runtime_mode::worker;
        }
#elif defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        rtcfg.mode_ = hpx::runtime_mode::console;
#else
        rtcfg.mode_ = hpx::runtime_mode::local;
#endif

        rtcfg.add_entry("hpx.parcel.openshmem.rank", std::to_string(this_rank));
        rtcfg.add_entry(
            "hpx.parcel.openshmem.processorname", get_processor_name());
    }

    std::string openshmem_environment::get_processor_name()
    {
        char name[1024 + 1] = {'\0'};
        const std::string rnkstr = std::to_string(rank());
        const int len = rnkstr.size();
        if (1025 < len)
        {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::get_processor_name",
                "openshmem processor name is larger than 1025");
        }
        std::copy(std::begin(rnkstr), std::end(rnkstr), name);
        return name;
    }

    void openshmem_environment::put_signal(const std::uint8_t* addr,
        const int node, std::uint8_t* raddr, const std::size_t size,
        unsigned int* sigaddr)
    {
        if (rank() == node)
        {
            const std::lock_guard<hpx::mutex> lk(segment_mutex[node]);
            std::memmove(raddr, addr, size);
        }
        else
        {
            const std::lock_guard<hpx::mutex> lk(segment_mutex[node]);

            shmem_uint8_put_signal(raddr, addr, size,
                reinterpret_cast<std::uint64_t*>(sigaddr), 1, SHMEM_SIGNAL_SET,
                node);
        }
    }

    void openshmem_environment::wait_until(
        const unsigned int value, unsigned int* sigaddr)
    {
        shmem_uint_wait_until(sigaddr, SHMEM_CMP_EQ, value);
    }

    void openshmem_environment::get(std::uint8_t* addr, const int node,
        const std::uint8_t* raddr, const std::size_t size)
    {
        if (rank() == node)
        {
            std::memmove(addr, raddr, size);
        }
        else
        {
            shmem_uint8_get(
                addr, raddr, size, node);    // dest, node, src, size
        }
    }

    void openshmem_environment::global_barrier()
    {
        shmem_barrier_all();
    }

    void openshmem_environment::finalize()
    {
        if (enabled() && has_called_init())
        {
            shmem_finalize();

            delete segments;
            delete segment_mutex;
            shmem_free(hpx::util::openshmem_environment::shmem_buffer);
            segments = nullptr;
            segment_mutex = nullptr;
            shmem_buffer = nullptr;
        }
    }

    bool openshmem_environment::enabled()
    {
        return enabled_;
    }

    bool openshmem_environment::multi_threaded()
    {
        return provided_threading_flag_ != 0;
    }

    bool openshmem_environment::has_called_init()
    {
        return has_called_init_;
    }

    int openshmem_environment::size()
    {
        int res(-1);
        if (enabled())
            res = static_cast<int>(shmem_n_pes());
        return res;
    }

    int openshmem_environment::rank()
    {
        int res(-1);
        if (enabled())
            res = static_cast<int>(shmem_my_pe());
        return res;
    }

    openshmem_environment::scoped_lock::scoped_lock()
    {
        if (!multi_threaded())
            mtx_.lock();
    }

    openshmem_environment::scoped_lock::~scoped_lock()
    {
        if (!multi_threaded())
            mtx_.unlock();
    }

    void openshmem_environment::scoped_lock::unlock()
    {
        if (!multi_threaded())
            mtx_.unlock();
    }

    openshmem_environment::scoped_try_lock::scoped_try_lock()
      : locked(true)
    {
        if (!multi_threaded())
        {
            locked = mtx_.try_lock();
        }
    }

    openshmem_environment::scoped_try_lock::~scoped_try_lock()
    {
        if (!multi_threaded() && locked)
            mtx_.unlock();
    }

    void openshmem_environment::scoped_try_lock::unlock()
    {
        if (!multi_threaded() && locked)
        {
            locked = false;
            mtx_.unlock();
        }
    }
}    // namespace hpx::util

#endif
