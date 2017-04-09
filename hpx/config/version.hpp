//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2013 Adrian Serio
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Make HPX inspect tool happy: hpxinspect:nounnamed

#ifndef HPX_CONFIG_VERSION_HPP
#define HPX_CONFIG_VERSION_HPP

#include <hpx/config.hpp>
#include <hpx/config/export_definitions.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/version.hpp>

///////////////////////////////////////////////////////////////////////////////
//  The version of HPX
//
//  HPX_VERSION_FULL & 0x0000FF is the sub-minor version
//  HPX_VERSION_FULL & 0x00FF00 is the minor version
//  HPX_VERSION_FULL & 0xFF0000 is the major version
//
//  HPX_VERSION_DATE   YYYYMMDD is the date of the release
//                               (estimated release date for master branch)
//
#define HPX_VERSION_FULL         0x001000

#define HPX_VERSION_MAJOR        1
#define HPX_VERSION_MINOR        0
#define HPX_VERSION_SUBMINOR     0

#define HPX_VERSION_DATE         20170424

#if !defined(HPX_AGAS_VERSION)
    #define HPX_AGAS_VERSION 0x30
#endif

#define HPX_VERSION_TAG          "-rc1"

#if !defined(HPX_HAVE_GIT_COMMIT)
    #define HPX_HAVE_GIT_COMMIT  "unknown"
#endif

///////////////////////////////////////////////////////////////////////////////
// The version check enforces the major and minor version numbers to match for
// every compilation unit to be compiled.
#define HPX_CHECK_VERSION                                                     \
    BOOST_PP_CAT(hpx_check_version_,                                          \
        BOOST_PP_CAT(HPX_VERSION_MAJOR,                                       \
            BOOST_PP_CAT(_, HPX_VERSION_MINOR)))                              \
    /**/

// The version check enforces the major and minor version numbers to match for
// every compilation unit to be compiled.
#define HPX_CHECK_BOOST_VERSION                                               \
    BOOST_PP_CAT(hpx_check_boost_version_, BOOST_VERSION)                     \
    /**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // Helper data structures allowing to automatically detect version problems
    // between applications and the core libraries.
    HPX_EXPORT extern char const HPX_CHECK_VERSION[];
    HPX_EXPORT extern char const HPX_CHECK_BOOST_VERSION[];
}

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_EXPORTS) && !defined(HPX_NO_VERSION_CHECK)
    // This is instantiated for each translation unit outside of the HPX core
    // library, forcing to resolve the variable HPX_CHECK_VERSION.
    namespace
    {
        // Note: this function is never executed.
#if defined(__GNUG__)
        __attribute__ ((unused))
#endif
        char const* check_hpx_version()
        {
            char const* versions[] = {
                hpx::HPX_CHECK_VERSION, hpx::HPX_CHECK_BOOST_VERSION
            };
            return versions[0];
        }
    }
#endif

#endif
