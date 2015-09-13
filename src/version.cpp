////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config/defines.hpp>
#include <hpx/version.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#endif

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/find_prefix.hpp>

#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <sstream>

#if defined(HPX_HAVE_HWLOC)
#include <hwloc.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    boost::uint8_t major_version()
    {
        return HPX_VERSION_MAJOR;
    }

    boost::uint8_t minor_version()
    {
        return HPX_VERSION_MINOR;
    }

    boost::uint8_t subminor_version()
    {
        return HPX_VERSION_SUBMINOR;
    }

    boost::uint32_t full_version()
    {
        return HPX_VERSION_FULL;
    }

    std::string full_version_as_string()
    {
        return boost::str(
            boost::format("%d.%d.%d") % //-V609
            HPX_VERSION_MAJOR % HPX_VERSION_MINOR %
            HPX_VERSION_SUBMINOR);
    }

    boost::uint8_t agas_version()
    {
        return HPX_AGAS_VERSION;
    }

    std::string tag()
    {
        return HPX_VERSION_TAG;
    }

#if defined(HPX_HAVE_PARCELPORT_MPI)
    std::string mpi_version()
    {
        std::ostringstream strm;

        // add type and library version
#if defined(OPEN_MPI)
        strm << "OpenMPI V" << OMPI_MAJOR_VERSION << "."
             << OMPI_MINOR_VERSION << "." << OMPI_RELEASE_VERSION;
#elif defined(MPICH)
        strm << "MPICH V" << MPICH_VERSION;
#elif defined(MVAPICH2_VERSION)
        strm << "MVAPICH2 V" << MVAPICH2_VERSION;
#else
        strm << "Unknown MPI";
#endif
        // add general MPI version
        strm << ", MPI V" << MPI_VERSION << "." << MPI_SUBVERSION;
        return strm.str();
    }
#endif

    std::string copyright()
    {
        char const* const copyright =
            "HPX - High Performance ParalleX\n"
            "A general purpose parallel C++ runtime system for\
             distributed applications\n"
            "of any scale.\n\n"
            "Copyright (c) 2007-2015, The STE||AR Group,\n"
            "http://stellar-group.org, email:hpx-users@stellar.cct.lsu.edu\n\n"
            "Distributed under the Boost Software License, \
             Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n";
        return copyright;
    }

    // Returns the HPX full build information string.
    std::string full_build_string()
    {
        std::ostringstream strm;
        strm << "{config}:\n" << configuration_string()
             << "{version}: " << build_string() << "\n"
             << "{boost}: " << boost_version() << "\n"
             << "{build-type}: " << build_type() << "\n"
             << "{date}: " << build_date_time() << "\n"
             << "{platform}: " << boost_platform() << "\n"
             << "{compiler}: " << boost_compiler() << "\n"
             << "{stdlib}: " << boost_stdlib() << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    //  HPX_HAVE_THREAD_PARENT_REFERENCE=1
    //  HPX_HAVE_THREAD_PHASE_INFORMATION=1
    //  HPX_HAVE_THREAD_DESCRIPTION=1
    //  HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION=1
    //  HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH=5
    //  HPX_HAVE_THREAD_TARGET_ADDRESS=1
    //  HPX_HAVE_THREAD_QUEUE_WAITTIME=0

    std::string configuration_string()
    {
        std::ostringstream strm;

#if defined(HPX_HAVE_NATIVE_TLS)
        strm << "  HPX_HAVE_NATIVE_TLS=ON\n";
#else
        strm << "  HPX_HAVE_NATIVE_TLS=OFF\n";
#endif
#if defined(HPX_HAVE_STACKTRACES)
        strm << "  HPX_HAVE_STACKTRACES=ON\n";
#else
        strm << "  HPX_HAVE_STACKTRACES=OFF\n";
#endif
#if defined(HPX_HAVE_COMPRESSION_BZIP2)
        strm << "  HPX_HAVE_COMPRESSION_BZIP2=ON\n";
#else
        strm << "  HPX_HAVE_COMPRESSION_BZIP2=OFF\n";
#endif
#if defined(HPX_HAVE_COMPRESSION_SNAPPY)
        strm << "  HPX_HAVE_COMPRESSION_SNAPPY=ON\n";
#else
        strm << "  HPX_HAVE_COMPRESSION_SNAPPY=OFF\n";
#endif
#if defined(HPX_HAVE_COMPRESSION_ZLIB)
        strm << "  HPX_HAVE_COMPRESSION_ZLIB=ON\n";
#else
        strm << "  HPX_HAVE_COMPRESSION_ZLIB=OFF\n";
#endif
#if defined(HPX_HAVE_PARCEL_COALESCING)
        strm << "  HPX_HAVE_PARCEL_COALESCING=ON\n";
#else
        strm << "  HPX_HAVE_PARCEL_COALESCING=OFF\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_TCP)
        strm << "  HPX_HAVE_PARCELPORT_TCP=ON\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_TCP=OFF\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
        strm << "  HPX_HAVE_PARCELPORT_MPI=ON (" << mpi_version() << ")\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_MPI=OFF\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_IPC)
        strm << "  HPX_HAVE_PARCELPORT_IPC=ON\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_IPC=OFF\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        strm << "  HPX_HAVE_PARCELPORT_IBVERBS=ON\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_IBVERBS=OFF\n";
#endif
#if defined(HPX_HAVE_VERIFY_LOCKS)
        strm << "  HPX_HAVE_VERIFY_LOCKS=ON\n";
#else
        strm << "  HPX_HAVE_VERIFY_LOCKS=OFF\n";
#endif
#if defined(HPX_HAVE_HWLOC)
        strm << "  HPX_HAVE_HWLOC=ON\n";
#else
        strm << "  HPX_HAVE_HWLOC=OFF\n";
#endif
#if defined(HPX_HAVE_ITTNOTIFY)
        strm << "  HPX_HAVE_ITTNOTIFY=ON\n";
#else
        strm << "  HPX_HAVE_ITTNOTIFY=OFF\n";
#endif
#if defined(BOOST_MSVC)
#if defined(HPX_HAVE_FIBER_BASED_COROUTINES)
        strm << "  HPX_HAVE_FIBER_BASED_COROUTINES=ON\n";
#else
        strm << "  HPX_HAVE_FIBER_BASED_COROUTINES=OFF\n";
#endif
#if defined(HPX_HAVE_SWAP_CONTEXT_EMULATION)
        strm << "  HPX_HAVE_SWAP_CONTEXT_EMULATION=ON\n";
#else
        strm << "  HPX_HAVE_SWAP_CONTEXT_EMULATION=OFF\n";
#endif
#endif
#if defined(HPX_HAVE_RUN_MAIN_EVERYWHERE)
        strm << "  HPX_HAVE_RUN_MAIN_EVERYWHERE=ON\n";
#else
        strm << "  HPX_HAVE_RUN_MAIN_EVERYWHERE=OFF\n";
#endif

#if defined(HPX_LIMIT)
        strm << "  HPX_LIMIT=" << HPX_LIMIT << "\n";
#endif
#if defined(HPX_PARCEL_MAX_CONNECTIONS)
        strm << "  HPX_PARCEL_MAX_CONNECTIONS="
             << HPX_PARCEL_MAX_CONNECTIONS << "\n";
#endif
#if defined(HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY)
        strm << "  HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY="
             << HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY << "\n";
#endif
#if defined(HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE)
        strm << "  HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE="
             << HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE << "\n";
#endif
#if defined(HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD)
        strm << "  HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD="
             << HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD << "\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_IPC) && defined(HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE)
        strm << "  HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE="
             << HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE << "\n";
#endif
#if defined(HPX_HAVE_MALLOC)
        strm << "  HPX_HAVE_MALLOC=" << HPX_HAVE_MALLOC << "\n";
#endif

        strm << "  HPX_PREFIX (configured)=" << util::hpx_prefix() << "\n";
#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__MIC)
        strm << "  HPX_PREFIX=" << util::find_prefix() << "\n";
#endif

        return strm.str();
    }

    std::string build_string()
    {
        return boost::str(
            boost::format("V%s%s (AGAS: V%d.%d), Git: %.10s") % //-V609
                full_version_as_string() % HPX_VERSION_TAG %
                (HPX_AGAS_VERSION / 0x10) % (HPX_AGAS_VERSION % 0x10) %
                HPX_HAVE_GIT_COMMIT);
    }

    std::string boost_version()
    {
        // BOOST_VERSION: 105400
        return boost::str(boost::format("V%d.%d.%d") %
            (BOOST_VERSION / 100000) % (BOOST_VERSION / 100 % 1000) %
            (BOOST_VERSION % 100));
    }

#if defined(HPX_HAVE_HWLOC)
    std::string hwloc_version()
    {
        // HWLOC_API_VERSION: 0x00010700
        return boost::str(boost::format("V%d.%d.%d") %
            (HWLOC_API_VERSION / 0x10000) % (HWLOC_API_VERSION / 0x100 % 0x100) %
            (HWLOC_API_VERSION % 0x100));
    }
#endif

#if defined(HPX_HAVE_MALLOC)
    std::string malloc_version()
    {
        return HPX_HAVE_MALLOC;
    }
#endif

    std::string boost_platform()
    {
        return BOOST_PLATFORM;
    }

    std::string boost_compiler()
    {
        return BOOST_COMPILER;
    }

    std::string boost_stdlib()
    {
        return BOOST_STDLIB;
    }

    std::string complete_version()
    {
        boost::format logo(
            "Versions:\n"
            "  HPX: %s\n"
            "  Boost: %s\n"
#if defined(HPX_HAVE_HWLOC)
            "  Hwloc: %s\n"
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
            "  MPI: %s\n"
#endif
            "\n"
            "Build:\n"
            "  Type: %s\n"
            "  Date: %s\n"
            "  Platform: %s\n"
            "  Compiler: %s\n"
            "  Standard Library: %s\n");

        std::string version = boost::str(logo %
            build_string() %
            boost_version() %
#if defined(HPX_HAVE_HWLOC)
            hwloc_version() %
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
            mpi_version() %
#endif
            build_type() %
            build_date_time() %
            boost_platform() %
            boost_compiler() %
            boost_stdlib());

#if defined(HPX_HAVE_MALLOC)
            version += "  Allocator: " + malloc_version() + "\n";
#endif

            return version;
    }

    std::string build_type()
    {
        return BOOST_PP_STRINGIZE(HPX_BUILD_TYPE);
    }

    std::string build_date_time()
    {
        return std::string(__DATE__)  + " " + __TIME__;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string runtime_configuration_string(
        util::command_line_handling const& cfg)
    {
        std::ostringstream strm;

        // runtime mode
        strm << "  {mode}: " << get_runtime_mode_name(cfg.mode_) << "\n";

        if (cfg.num_localities_ != 1)
            strm << "  {localities}: " << cfg.num_localities_ << "\n";

        // default scheduler used for this run
        strm << "  {scheduler}: " << cfg.queuing_ << "\n";

        // amount of threads and cores configured for this run
        strm << "  {os-threads}: " << cfg.num_threads_ << "\n";
        strm << "  {cores}: " << cfg.num_cores_ << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////////
    char const HPX_CHECK_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_VERSION);
    char const HPX_CHECK_BOOST_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_BOOST_VERSION);
}

