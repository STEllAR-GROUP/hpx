////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

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
            "An experimental runtime system for conventional machines implementing\n"
            "(parts of) the ParalleX execution model.\n\n"
            "Copyright (C) 1998-2012 Hartmut Kaiser, Bryce Lelbach and others\n"
            "The STE||AR Group, http://stellar.cct.lsu.edu\n\n"
            "Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n";

        return copyright;
    }

    std::string build_string()
    {
        return boost::str(boost::format("V%d.%d.%d%s (AGAS: V%d.%d), SVN: %s") %
            HPX_VERSION_MAJOR % HPX_VERSION_MINOR %
            HPX_VERSION_SUBMINOR % HPX_VERSION_TAG %
            (HPX_AGAS_VERSION / 0x10) % (HPX_AGAS_VERSION % 0x10) %
            HPX_SVN_REVISION);
    }

    std::string boost_version()
    {
        return boost::str(boost::format("V%d.%d.%d") %
            (BOOST_VERSION / 100000) % (BOOST_VERSION / 100 % 1000) %
            (BOOST_VERSION % 100));
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
            "  Date: %s %s\n"
            "  Platform: %s\n"
            "  Compiler: %s\n"
            "  Standard Library: %s\n");

        return boost::str(logo %
            build_string() %
            boost_version() %
            HPX_BUILD_TYPE %
            __DATE__ % __TIME__ %
            BOOST_PLATFORM %
            BOOST_COMPILER %
            BOOST_STDLIB);
    }

    ///////////////////////////////////////////////////////////////////////////
    char const HPX_CHECK_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_VERSION);
    char const HPX_CHECK_BOOST_VERSION[] = BOOST_PP_STRINGIZE(HPX_CHECK_BOOST_VERSION);
}

