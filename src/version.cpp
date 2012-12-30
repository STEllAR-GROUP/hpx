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

#include <boost/version.hpp>
#include <boost/format.hpp>
#include <boost/preprocessor/stringize.hpp>

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
            "An distributed and parallel runtime system for conventional machines\n"
            "implementing (parts of) the ParalleX execution model.\n\n"
            "Copyright (C) 1998-2012 The STE||AR Group, Louisiana State University, http://stellar.cct.lsu.edu\n\n"
            "Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n";

        return copyright;
    }

    // Returns the HPX full build information string.
    std::string full_build_string()
    {
        hpx::util::osstream strm;
        strm << "{version}: " << build_string() << "\n"
             << "{boost}: " << boost_version() << "\n"
             << "{build-type}: " << build_type() << "\n"
             << "{date}: " << build_date_time() << "\n"
             << "{platform}: " << boost_platform() << "\n"
             << "{compiler}: " << boost_compiler() << "\n"
             << "{stdlib}: " << boost_stdlib() << "\n";
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
        return boost::str(boost::format("V%d.%d.%d") %
            (BOOST_VERSION / 100000) % (BOOST_VERSION / 100 % 1000) %
            (BOOST_VERSION % 100));
    }

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
            "\n"
            "Build:\n"
            "  Type: %s\n"
            "  Date: %s\n"
            "  Platform: %s\n"
            "  Compiler: %s\n"
            "  Standard Library: %s\n");

        return boost::str(logo %
            build_string() %
            boost_version() %
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

