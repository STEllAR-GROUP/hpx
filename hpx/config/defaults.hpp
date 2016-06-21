//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_DEFAULTS_SEP_26_2008_0352PM)
#define HPX_CONFIG_DEFAULTS_SEP_26_2008_0352PM

#include <hpx/util/find_prefix.hpp>
#include <hpx/version.hpp>

#include <boost/preprocessor/stringize.hpp>

// The HPX runtime needs to know where to look for the HPX ini files if no ini
// path is specified by the user (default in $HPX_LOCATION/share/hpx-1.0.0/ini).
// Also, the default component path is set within the same prefix

#define HPX_BASE_DIR_NAME             "hpx-"                                  \
        BOOST_PP_STRINGIZE(HPX_VERSION_MAJOR) "."                             \
        BOOST_PP_STRINGIZE(HPX_VERSION_MINOR) "."                             \
        BOOST_PP_STRINGIZE(HPX_VERSION_SUBMINOR)                              \
    /**/

#if !defined(HPX_DEFAULT_INI_PATH)
#define HPX_DEFAULT_INI_PATH                                                  \
        hpx::util::find_prefixes("/share/" HPX_BASE_DIR_NAME "/ini")          \
    /**/
#endif
#if !defined(HPX_DEFAULT_INI_FILE)
#define HPX_DEFAULT_INI_FILE                                                  \
        hpx::util::find_prefixes("/share/" HPX_BASE_DIR_NAME "/hpx.ini")      \
    /**/
#endif
#if !defined(HPX_DEFAULT_COMPONENT_PATH)
#define HPX_DEFAULT_COMPONENT_PATH                                            \
        hpx::util::find_prefixes("/hpx")                                      \
    /**/
#endif

#endif

