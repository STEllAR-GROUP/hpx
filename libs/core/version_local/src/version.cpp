////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/local/config.hpp>
#include <hpx/local/config/config_strings.hpp>
#include <hpx/local/config/version.hpp>
#include <hpx/local/version.hpp>
#include <hpx/modules/config_registry.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <boost/config.hpp>
#include <boost/version.hpp>

#if defined(HPX_HAVE_MODULE_MPI_BASE)
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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::local {
    std::uint8_t major_version()
    {
        return HPX_LOCAL_VERSION_MAJOR;
    }

    std::uint8_t minor_version()
    {
        return HPX_LOCAL_VERSION_MINOR;
    }

    std::uint8_t subminor_version()
    {
        return HPX_LOCAL_VERSION_SUBMINOR;
    }

    std::uint32_t full_version()
    {
        return HPX_LOCAL_VERSION_FULL;
    }

    std::string full_version_as_string()
    {
        return hpx::util::format("{}.{}.{}",    //-V609
            HPX_LOCAL_VERSION_MAJOR, HPX_LOCAL_VERSION_MINOR,
            HPX_LOCAL_VERSION_SUBMINOR);
    }

    std::string tag()
    {
        return HPX_LOCAL_VERSION_TAG;
    }

#if defined(HPX_HAVE_MODULE_MPI_BASE)
    std::string mpi_version()
    {
        std::ostringstream strm;

        // add type and library version
#if defined(OPEN_MPI)
        hpx::util::format_to(strm, "OpenMPI V{}.{}.{}", OMPI_MAJOR_VERSION,
            OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#elif defined(MPICH)
        hpx::util::format_to(strm, "MPICH V{}", MPICH_VERSION);
#elif defined(MVAPICH2_VERSION)
        hpx::util::format_to(strm, "MVAPICH2 V{}", MVAPICH2_VERSION);
#else
        strm << "Unknown MPI";
#endif
        // add general MPI version
#if defined(MPI_VERSION) && defined(MPI_SUBVERSION)
        hpx::util::format_to(strm, ", MPI V{}.{}", MPI_VERSION, MPI_SUBVERSION);
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
            "Copyright (c) 2007-2021, The STE||AR Group,\n"
            "http://stellar-group.org, email:hpx-users@stellar-group.org\n\n"
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
    std::string configuration_string()
    {
        std::ostringstream strm;

        strm << "HPXLocal:\n";

#if defined(HPX_HAVE_MALLOC)
        hpx::util::format_to(strm, "  HPX_HAVE_MALLOC={}\n", HPX_HAVE_MALLOC);
#endif

        char const* const* p = hpx::local::config_strings;
        while (*p)
            strm << "  " << *p++ << "\n";
        strm << "\n";

        // print module configurations
        auto configs = hpx::config_registry::get_module_configs();
        std::sort(configs.begin(), configs.end(),
            [](auto& a, auto& b) { return a.module_name < b.module_name; });
        for (auto& c : configs)
        {
            if (!c.config_entries.empty())
            {
                strm << "Module " << c.module_name << ":\n";

                for (auto const& e : c.config_entries)
                {
                    strm << "  " << e << std::endl;
                }

                strm << "\n";
            }
        }

        return strm.str();
    }

    std::string build_string()
    {
        return hpx::util::format("V{}{}, Git: {:.10}",    //-V609
            full_version_as_string(), HPX_LOCAL_VERSION_TAG,
            HPX_LOCAL_HAVE_GIT_COMMIT);
    }

    std::string boost_version()
    {
        // BOOST_VERSION: 107100
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
                                                "  HPXLocal: {}\n"
                                                "  Boost: {}\n"
                                                "  Hwloc: {}\n"
#if defined(HPX_HAVE_MODULE_MPI_BASE)
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
#if defined(HPX_HAVE_MODULE_MPI_BASE)
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
        return HPX_PP_STRINGIZE(HPX_LOCAL_BUILD_TYPE);
    }

    std::string build_date_time()
    {
        return std::string(__DATE__) + " " + __TIME__;
    }
}    // namespace hpx::local
