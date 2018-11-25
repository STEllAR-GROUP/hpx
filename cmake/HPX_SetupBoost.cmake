# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We first try to find the required minimum set of Boost libraries. This will
# also give us the version of the found boost installation
if(HPX_WITH_STATIC_LINKING)
  set(Boost_USE_STATIC_LIBS ON)
endif()

# Add additional version to recognize
set(Boost_ADDITIONAL_VERSIONS
    ${Boost_ADDITIONAL_VERSIONS}
    "1.66.0" "1.66"
    "1.65.0" "1.65"
    "1.64.0" "1.64"
    "1.63.0" "1.63"
    "1.62.0" "1.62"
    "1.61.0" "1.61"
    "1.60.0" "1.60"
    "1.59.0" "1.59"
    "1.58.0" "1.58"
    "1.57.0" "1.57")

find_package(Boost 1.55 REQUIRED
                        COMPONENTS
                          filesystem
                          iostreams
                          program_options
                          system
                        OPTIONAL_COMPONENTS
                          context
                          thread
                          log
                          regex)

if(NOT Boost_FOUND)
  hpx_error("Could not find Boost. Please set BOOST_ROOT to point to your Boost installation.")
endif()

# Boost preprocessor definitions
hpx_add_config_cond_define(BOOST_PARAMETER_MAX_ARITY 7)
if(MSVC)
  hpx_option(HPX_WITH_BOOST_ALL_DYNAMIC_LINK BOOL
    "Add BOOST_ALL_DYN_LINK to compile flags (default: OFF)"
    OFF ADVANCED)
  if (HPX_WITH_BOOST_ALL_DYNAMIC_LINK)
    hpx_add_config_cond_define(BOOST_ALL_DYN_LINK)
  endif()
else()
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_cond_define(BOOST_BIGINT_HAS_NATIVE_INT64)
