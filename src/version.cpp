////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>
#include <boost/preprocessor/stringize.hpp>

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

    boost::uint8_t agas_version()
    {
        return HPX_AGAS_VERSION;
    }

    std::string tag()
    {
        return HPX_VERSION_TAG;
    }

    std::string copyright()
    {
        char const* const copyright =
            "HPX - High Performance ParalleX\n"
            "A general purpose parallel C++ runtime system for distributed applications\n"
            "of any scale.\n\n"
            "Copyright (c) 2007-2013 The STE||AR Group, Louisiana State University,\n"
            "http://stellar.cct.lsu.edu, email:hpx-users@stellar.cct.lsu.edu\n\n"
            "Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n";
        return copyright;
    }

    // Returns the HPX full build information string.
    std::string full_build_string()
    {
        hpx::util::osstream strm;
        strm << "{config}:\n" << configuration_string()
             << "{version}: " << build_string() << "\n"
             << "{boost}: " << boost_version() << "\n"
             << "{build-type}: " << build_type() << "\n"
             << "{date}: " << build_date_time() << "\n"
             << "{platform}: " << boost_platform() << "\n"
             << "{compiler}: " << boost_compiler() << "\n"
             << "{stdlib}: " << boost_stdlib() << "\n";

        return util::osstream_get_string(strm);
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    //  HPX_THREAD_MAINTAIN_PARENT_REFERENCE=1
    //  HPX_THREAD_MAINTAIN_PHASE_INFORMATION=1
    //  HPX_THREAD_MAINTAIN_DESCRIPTION=1
    //  HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION=1
    //  HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH=5
    //  HPX_THREAD_MAINTAIN_TARGET_ADDRESS=1
    //  HPX_THREAD_MAINTAIN_QUEUE_WAITTIME=0
    //  HPX_UTIL_BIND
    //  HPX_UTIL_FUNCTION
    //  HPX_UTIL_TUPLE
    //  HPX_HAVE_CXX11_RVALUE_REFERENCES
    //  HPX_HAVE_CXX11_LAMBDAS
    //  HPX_HAVE_CXX11_AUTO
    //  HPX_HAVE_CXX11_DECLTYPE
    //  HPX_HAVE_CXX11_STD_UNIQUE_PTR
    //  HPX_ACTION_ARGUMENT_LIMIT=4
    //  HPX_FUNCTION_ARGUMENT_LIMIT=7

    std::string configuration_string()
    {
        hpx::util::osstream strm;

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
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
        strm << "  HPX_HAVE_PARCELPORT_SHMEM=ON\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_SHMEM=OFF\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        strm << "  HPX_HAVE_PARCELPORT_IBVERBS=ON\n";
#else
        strm << "  HPX_HAVE_PARCELPORT_IBVERBS=OFF\n";
#endif
#if defined(HPX_HAVE_VERIFY_LOCKS) && HPX_HAVE_VERIFY_LOCKS
        strm << "  HPX_HAVE_VERIFY_LOCKS=ON\n";
#else
        strm << "  HPX_HAVE_VERIFY_LOCKS=OFF\n";
#endif
#if defined(HPX_HAVE_HWLOC)
        strm << "  HPX_HAVE_HWLOC=ON\n";
#else
        strm << "  HPX_HAVE_HWLOC=OFF\n";
#endif
#if defined(HPX_HAVE_ITTNOTIFY) && HPX_HAVE_ITTNOTIFY
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

#if defined(HPX_LIMIT)
        strm << "  HPX_LIMIT=" << HPX_LIMIT << "\n";
#endif
#if defined(HPX_ACTION_ARGUMENT_LIMIT)
        strm << "  HPX_ACTION_ARGUMENT_LIMIT="
             << HPX_ACTION_ARGUMENT_LIMIT << "\n";
#endif
#if defined(HPX_COMPONENT_CREATE_ARGUMENT_LIMIT)
        strm << "  HPX_COMPONENT_CREATE_ARGUMENT_LIMIT="
             << HPX_COMPONENT_CREATE_ARGUMENT_LIMIT << "\n";
#endif
#if defined(HPX_FUNCTION_ARGUMENT_LIMIT)
        strm << "  HPX_FUNCTION_ARGUMENT_LIMIT="
             << HPX_FUNCTION_ARGUMENT_LIMIT << "\n";
#endif
#if defined(HPX_LOCK_LIMIT)
        strm << "  HPX_LOCK_LIMIT=" << HPX_LOCK_LIMIT << "\n";
#endif
#if defined(HPX_TUPLE_LIMIT)
        strm << "  HPX_TUPLE_LIMIT=" << HPX_TUPLE_LIMIT << "\n";
#endif
#if defined(HPX_WAIT_ARGUMENT_LIMIT)
        strm << "  HPX_WAIT_ARGUMENT_LIMIT="
             << HPX_WAIT_ARGUMENT_LIMIT << "\n";
#endif
#if defined(HPX_MAX_PARCEL_CONNECTIONS)
        strm << "  HPX_MAX_PARCEL_CONNECTIONS="
             << HPX_MAX_PARCEL_CONNECTIONS << "\n";
#endif
#if defined(HPX_MAX_PARCEL_CONNECTIONS_PER_LOCALITY)
        strm << "  HPX_MAX_PARCEL_CONNECTIONS_PER_LOCALITY="
             << HPX_MAX_PARCEL_CONNECTIONS_PER_LOCALITY << "\n";
#endif
#if defined(HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE)
        strm << "  HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE="
             << HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE << "\n";
#endif
#if defined(HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD)
        strm << "  HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD="
             << HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD << "\n";
#endif
#if defined(HPX_HAVE_PARCELPORT_SHMEM) && defined(HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE)
        strm << "  HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE="
             << HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE << "\n";
#endif

        strm << "  HPX_PREFIX=" << HPX_PREFIX << "\n";

        return util::osstream_get_string(strm);
    }

    std::string build_string()
    {
        return boost::str(
            boost::format("V%d.%d.%d%s (AGAS: V%d.%d), Git: %s") % //-V609
                HPX_VERSION_MAJOR % HPX_VERSION_MINOR %
                HPX_VERSION_SUBMINOR % HPX_VERSION_TAG %
                (HPX_AGAS_VERSION / 0x10) % (HPX_AGAS_VERSION % 0x10) %
                HPX_GIT_COMMIT);
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
            "\n"
            "Build:\n"
            "  Type: %s\n"
            "  Date: %s\n"
            "  Platform: %s\n"
            "  Compiler: %s\n"
            "  Standard Library: %s");

        return boost::str(logo %
            build_string() %
            boost_version() %
#if defined(HPX_HAVE_HWLOC)
            hwloc_version() %
#endif
            build_type() %
            build_date_time() %
            boost_platform() %
            boost_compiler() %
            boost_stdlib());
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
    char const HPX_CHECK_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_VERSION);
    char const HPX_CHECK_BOOST_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_BOOST_VERSION);
}

