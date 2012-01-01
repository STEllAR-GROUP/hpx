//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file relies on a number of defines to be set, usually by
// configure or the build environment used.  These defines are:
//
// HPX_PREFIX: the target directory for this HPX installation.

#if !defined(HPX_CONFIG_DEFAULTS_SEP_26_2008_0352PM)
#define HPX_CONFIG_DEFAULTS_SEP_26_2008_0352PM

#if !defined(HPX_PREFIX)
#error "Do not include this file without defining HPX_PREFIX."
#endif

// The HPX runtime needs to know where to look for the HPX ini files if no ini
// path is specified by the user (default in $HPX_LOCATION/share/saga/). Also,
// the default component path is set within the same prefix

#if !defined(HPX_DEFAULT_INI_PATH)
#define HPX_DEFAULT_INI_PATH          HPX_PREFIX "/share/hpx"
#endif
#if !defined(HPX_DEFAULT_INI_FILE)
#define HPX_DEFAULT_INI_FILE          HPX_PREFIX "/share/hpx/hpx.ini"
#endif
#if !defined(HPX_DEFAULT_COMPONENT_PATH)
#define HPX_DEFAULT_COMPONENT_PATH    HPX_PREFIX "/lib/hpx"
#endif

#endif

