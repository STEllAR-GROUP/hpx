// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROGRAM_OPTIONS_VERSION_HPP_VP_2004_04_05
#define PROGRAM_OPTIONS_VERSION_HPP_VP_2004_04_05

#include <hpx/program_options/config.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/program_options/version.hpp
#include <boost/program_options/version.hpp>
#else

/** The version of the source interface.
    The value will be incremented whenever a change is made which might
    cause compilation errors for existing code.
*/
#ifdef HPX_PROGRAM_OPTIONS_VERSION
#error HPX_PROGRAM_OPTIONS_VERSION already defined
#endif
#define HPX_PROGRAM_OPTIONS_VERSION 2

// Signal that implicit options will use values from next
// token, if available.
#define HPX_PROGRAM_OPTIONS_IMPLICIT_VALUE_NEXT_TOKEN 1

#endif
#endif
