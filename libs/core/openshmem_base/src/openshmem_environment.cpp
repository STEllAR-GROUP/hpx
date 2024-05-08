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
#include <hpx/modules/openshmem_base.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/util.hpp>

#include <shmemx.h>

#include <atomic>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    bool openshmem_environment::check_openshmem_environment(
        util::runtime_configuration const& cfg)
    {
#if !defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_OPENSHMEM)
        return false;
#elif defined(HPX_HAVE_MODULE_OPENSHMEM_BASE)
        const std::string default_env{"SHMEM_VERSION;SHMEM_INFO;SHMEM_SYMMETRIC_SIZE;MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID"};
        std::string openshmem_environment_strings =
            cfg.get_entry("hpx.parcel.openshmem.env", default_env);

        hpx::string_util::char_separator sep(";,: ");
        hpx::string_util::tokenizer tokens(openshmem_environment_strings, sep);
        for (auto const& tok : tokens)
        {
            char* env = std::getenv(tok.c_str());
            if (env)
            {
                LBT_(debug)
                    << "Found OpenSHMEM environment variable: " << tok << "="
                    << std::string(env) << ", enabling OpenSHMEM support\n";
                return true;
            }
        }

        LBT_(info) << "No known OpenSHMEM environment variable found, disabling "
                      "OpenSHMEM support\n";

        return false;
#else
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

            cfg.add_entry("hpx.parcel.openshmem.enable", "0");
            return false;
        }

        const std::string default_env{"SHMEM_VERSION;SHMEM_INFO;SHMEM_SYMMETRIC_SIZE;MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID"};
        std::string openshmem_environment_strings =
            cfg.get_entry("hpx.parcel.openshmem.env", default_env);

        hpx::string_util::char_separator sep(";,: ");
        hpx::string_util::tokenizer tokens(openshmem_environment_strings, sep);
        for (auto const& tok : tokens)
        {
            char* env = std::getenv(tok.c_str());
            if (env)
            {
                LBT_(debug)
                    << "Found OpenSHMEM environment variable: " << tok << "="
                    << std::string(env) << ", enabling OpenSHMEM support\n";
                return true;
            }
        }

        LBT_(info) << "No known OpenSHMEM environment variable found, disabling "
                      "OpenSHMEM support\n";

        return false;
#endif
    }
}    // namespace hpx::util

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_MODULE_OPENSHMEM_BASE))

namespace hpx::util {

    hpx::mutex openshmem_environment::mtx_{};
    bool openshmem_environment::enabled_ = false;
    bool openshmem_environment::has_called_init_ = false;
    int openshmem_environment::provided_threading_flag_ = 0;
    int openshmem_environment::is_initialized_ = -1;
    int openshmem_environment::init_val_ = 0;
    std::size_t openshmem_environment::this_rank = -1;
    //std::vector<std::shared_ptr<hpx::spinlock>> openshmem_environment::segment_mutex{};
    std::vector<openshmem_seginfo_t> openshmem_environment::segments{};
    std::uint8_t* hpx::util::openshmem_environment::shmem_buffer = nullptr;

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

        // get system page size
        //
        const std::size_t sys_pgsz = sysconf(_SC_PAGESIZE);

        // compute number of pages pages
        //
        // page_count = num_localities * number of threads
        //
        const std::size_t page_count = size();
	if(page_count < 1) {
            HPX_THROW_EXCEPTION(error::invalid_status,
                "hpx::util::openshmem_environment::init",
                "OPENSHMEM not ready error");
	}

        // symmetric allocation for number of pages total + number of signals
        //
        // (allocate page_size * number of PEs * number of threads) + (number of PEs * number of threads * 2 [for signaling])
        // 
        const std::size_t byte_count = (sys_pgsz*page_count)+(sizeof(unsigned int)*page_count*2);

        // allocate symmetric memory
        //
        hpx::util::openshmem_environment::shmem_buffer =
            static_cast<std::uint8_t*>(shmem_calloc(
                byte_count, sizeof(std::uint8_t)));

        // compute the base address for signals
        //
        const std::size_t beg_signal = (sys_pgsz*page_count);

        // allocate the page cache
        //
        openshmem_environment::segments.resize(page_count);

        // initialize the page cache
        //
        for (std::size_t i = 0; i < page_count; ++i)
        {
            segments[i].beg_addr = hpx::util::openshmem_environment::shmem_buffer +
                (i * sys_pgsz);
            segments[i].end_addr = static_cast<std::uint8_t*>(segments[i].beg_addr) +
                sys_pgsz;

            // all of the rcv signals are linearly arranged before the rcv signals
            //
            segments[i].rcv = hpx::util::openshmem_environment::shmem_buffer + beg_signal + i;

            // all of the xmt signals are linearly arranged after the rcv signals
            //
            segments[i].xmt = hpx::util::openshmem_environment::shmem_buffer + beg_signal + page_count + i;

            //segments[i].mut = &(openshmem_environment::segment_mutex[i]);
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

        if (enabled()) {
            scoped_lock l;
            this_rank = static_cast<int>(shmem_my_pe());
        }

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
        scoped_lock l;

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
        std::uint8_t * sigaddr)
    {
        if (rank() == node)
        {
            std::memmove(raddr, addr, size);
        }
        else
        {
#if !defined(SHMEM_SIGNAL_SET)
            shmem_uint8_put(raddr, addr, size, node);
            shmem_uint8_put(reinterpret_cast<std::uint8_t*>(sigaddr),
                reinterpret_cast<std::uint8_t*>(sigaddr), sizeof(sigaddr),
                node);
            shmem_fence();
#else
            shmem_uint8_put_signal(raddr, addr, size,
                reinterpret_cast<std::uint64_t*>(sigaddr), 1, SHMEM_SIGNAL_SET,
                node);
#endif
        }
    }

    // template metaprogramming for `openshmem_environment::wait_until`
    //
    template<typename Sig>
    struct signature;

    template<typename R, typename ...Args>
    struct signature<R(Args...)>
    {
        using type = std::tuple<Args...>;
    };

    void openshmem_environment::wait_until(
        const std::uint8_t value, std::uint8_t * sigaddr)
    {
        // some openshmem implementations place a `volatile` in the argument list
        // this section compile-time section detects this situation and enables
        // the right option
        //
        using arg_type = std::conditional<
            std::is_same<typename signature<decltype(shmem_int_wait_until)>::type, 
                         std::tuple<volatile int*, int, int>>::value,
                volatile unsigned int *,
                unsigned int *
            >::type;

        union {
            std::uint8_t * uaddr;
            arg_type iaddr;
        } tmp;
        tmp.uaddr = sigaddr;

	shmem_uint_wait_until(tmp.iaddr, SHMEM_CMP_EQ, static_cast<int>(value));
    }

    constexpr bool vendor_strings_equal(char const * a, char const * b) {
        return *a == *b && (*a == '\0' || vendor_strings_equal(a + 1, b + 1));
    }

    struct ucx {};
    struct mpi {};
    struct err {};

    template<typename tag>
    struct wait_until_any_wrapper {
        std::size_t operator()(unsigned int * addr, const std::size_t count, const std::uint8_t value) { return -1; }
    };

#if defined(OSHMEM_SHMEMX_H)
    template<>
    struct wait_until_any_wrapper<mpi> {
        std::size_t operator()(unsigned int * addr, const std::size_t count, const std::uint8_t value) {
            int rc = 0;
            for(std::size_t i = 0; i < count; ++i) {
                rc = shmem_uint_test(addr+i, SHMEM_CMP_EQ, value); 
                if(rc) { return i; }
            }

            return -1;
        }	    
    };
#endif

#if defined(_SHMEM_H)
    template<>
    struct wait_until_any_wrapper<ucx> {
        std::size_t operator()(unsigned int * addr, const std::size_t count, const std::uint8_t value) {
	    const std::size_t sig_idx =
#if defined(_SHMEMX_H) && _SHMEMX_H == 1
                shmemx_uint_wait_until_any
#else
                shmem_uint_wait_until_any
#endif
            (
                addr,
                count,
                nullptr,
                SHMEM_CMP_EQ,
                value
            );

            return sig_idx;
        }	    
    };
#endif

    std::size_t openshmem_environment::wait_until_any(const std::uint8_t value, std::uint8_t * sigaddr, const std::size_t count) {
#if defined(SHMEM_VENDOR_STRING)
	using tag = std::conditional< std::integral_constant<bool, vendor_strings_equal(SHMEM_VENDOR_STRING,"osss-ucx")>::value, ucx, 
            std::conditional< std::integral_constant<bool, vendor_strings_equal(SHMEM_VENDOR_STRING,"http://www.open-mpi.org/")>::value, mpi, err>::type
        >::type;
	using wait_until_any_type = wait_until_any_wrapper<tag>;
#else
        #define SHMEM_VENDOR_STRING "SHMEM_VENDOR_STRING not defined"
	using tag = err;
	using wait_until_any_type = wait_until_any_wrapper<tag>;
#endif
        wait_until_any_type t{};
	return t(reinterpret_cast<unsigned int *>(sigaddr), count, value);
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
            scoped_lock l;
            shmem_free(hpx::util::openshmem_environment::shmem_buffer);
            shmem_finalize();
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
        int res(0);
        if (enabled()) {
            scoped_lock l;
            res = static_cast<int>(shmem_n_pes());
        }
        return res;
    }

    int openshmem_environment::rank()
    {
        return this_rank;
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
