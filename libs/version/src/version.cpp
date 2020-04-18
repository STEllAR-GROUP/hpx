////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/config/config_strings.hpp>
#include <hpx/config/version.hpp>
#include <hpx/format.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/version.hpp>

#include <boost/config.hpp>
#include <boost/version.hpp>

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_LIB_MPI)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

// Intel MPI does not like to be included after stdio.h. As such, we include mpi.h
// as soon as possible.
#include <mpi.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif

#include <hwloc.h>

#include <cstdint>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    std::uint8_t major_version()
    {
        return HPX_VERSION_MAJOR;
    }

    std::uint8_t minor_version()
    {
        return HPX_VERSION_MINOR;
    }

    std::uint8_t subminor_version()
    {
        return HPX_VERSION_SUBMINOR;
    }

    std::uint32_t full_version()
    {
        return HPX_VERSION_FULL;
    }

    std::string full_version_as_string()
    {
        return hpx::util::format("{}.{}.{}",    //-V609
            HPX_VERSION_MAJOR, HPX_VERSION_MINOR, HPX_VERSION_SUBMINOR);
    }

    std::uint8_t agas_version()
    {
        return HPX_AGAS_VERSION;
    }

    std::string tag()
    {
        return HPX_VERSION_TAG;
    }

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_LIB_MPI)
    std::string mpi_version()
    {
        std::ostringstream strm;

        // add type and library version
#if defined(OPEN_MPI)
        strm << "OpenMPI V" << OMPI_MAJOR_VERSION << "." << OMPI_MINOR_VERSION
             << "." << OMPI_RELEASE_VERSION;
#elif defined(MPICH)
        strm << "MPICH V" << MPICH_VERSION;
#elif defined(MVAPICH2_VERSION)
        strm << "MVAPICH2 V" << MVAPICH2_VERSION;
#else
        strm << "Unknown MPI";
#endif
        // add general MPI version
#if defined(MPI_VERSION) && defined(MPI_SUBVERSION)
        strm << ", MPI V" << MPI_VERSION << "." << MPI_SUBVERSION;
#else
        strm << ", unknown MPI version";
#endif
        return strm.str();
    }
#endif

    std::string copyright()
    {
        char const* const copyright =
            "HPX - The C++ Standard Library for Parallelism and Concurrency\n"
            "(A general purpose parallel C++ runtime system for distributed "
            "applications\n"
            "of any scale).\n\n"
            "Copyright (c) 2007-2020, The STE||AR Group,\n"
            "http://stellar-group.org, email:hpx-users@stellar.cct.lsu.edu\n\n"
            "Distributed under the Boost Software License, "
            "Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at "
            "http://www.boost.org/LICENSE_1_0.txt)\n";
        return copyright;
    }

    // Returns the HPX full build information string.
    std::string full_build_string()
    {
        std::ostringstream strm;
        strm << "{config}:\n"
             << configuration_string() << "{version}: " << build_string()
             << "\n"
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

        strm << "Core library:\n";

        char const* const* p = hpx::config_strings;
        while (*p)
            strm << "  " << *p++ << "\n";
        strm << "\n";

#if defined(HPX_PARCEL_MAX_CONNECTIONS)
        strm << "  HPX_PARCEL_MAX_CONNECTIONS=" << HPX_PARCEL_MAX_CONNECTIONS
             << "\n";
#endif
#if defined(HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY)
        strm << "  HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY="
             << HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY << "\n";
#endif
#if defined(HPX_AGAS_LOCAL_CACHE_SIZE)
        strm << "  HPX_AGAS_LOCAL_CACHE_SIZE=" << HPX_AGAS_LOCAL_CACHE_SIZE
             << "\n";
#endif
#if defined(HPX_HAVE_MALLOC)
        strm << "  HPX_HAVE_MALLOC=" << HPX_HAVE_MALLOC << "\n";
#endif

        const char* prefix = util::hpx_prefix();
        if (prefix == nullptr)
        {
            strm << "  HPX_PREFIX (configured)=unknown\n";
#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__MIC)
            strm << "  HPX_PREFIX=unknown\n";
#endif
        }
        else
        {
            strm << "  HPX_PREFIX (configured)=" << prefix << "\n";
#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__MIC)
            strm << "  HPX_PREFIX=" << util::find_prefix() << "\n";
#endif
        }
        strm << "\n";

        // print module configurations
        module_config_strings const* ms = hpx::config_strings_modules;
        while (ms->name)
        {
            strm << "Module " << ms->name << ":\n";
            char const* const* p = ms->config_strings;
            while (*p)
                strm << "  " << *p++ << "\n";
            strm << "\n";
            ++ms;
        }

        return strm.str();
    }

    std::string build_string()
    {
        return hpx::util::format("V{}{} (AGAS: V{}.{}), Git: {:.10}",    //-V609
            full_version_as_string(), HPX_VERSION_TAG, HPX_AGAS_VERSION / 0x10,
            HPX_AGAS_VERSION % 0x10, HPX_HAVE_GIT_COMMIT);
    }

    std::string boost_version()
    {
        // BOOST_VERSION: 105800
        return hpx::util::format("V{}.{}.{}", BOOST_VERSION / 100000,
            BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100);
    }

    std::string hwloc_version()
    {
        // HWLOC_API_VERSION: 0x00010700
        return hpx::util::format("V{}.{}.{}", HWLOC_API_VERSION / 0x10000,
            HWLOC_API_VERSION / 0x100 % 0x100, HWLOC_API_VERSION % 0x100);
    }

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
        std::string version = hpx::util::format("Versions:\n"
                                                "  HPX: {}\n"
                                                "  Boost: {}\n"
                                                "  Hwloc: {}\n"
#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_LIB_MPI)
                                                "  MPI: {}\n"
#endif
                                                "\n"
                                                "Build:\n"
                                                "  Type: {}\n"
                                                "  Date: {}\n"
                                                "  Platform: {}\n"
                                                "  Compiler: {}\n"
                                                "  Standard Library: {}\n",
            build_string(), boost_version(), hwloc_version(),
#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_LIB_MPI)
            mpi_version(),
#endif
            build_type(), build_date_time(), boost_platform(), boost_compiler(),
            boost_stdlib());

#if defined(HPX_HAVE_MALLOC)
        version += "  Allocator: " + malloc_version() + "\n";
#endif

        return version;
    }

    std::string build_type()
    {
        return HPX_PP_STRINGIZE(HPX_BUILD_TYPE);
    }

    std::string build_date_time()
    {
        return std::string(__DATE__) + " " + __TIME__;
    }
}    // namespace hpx
