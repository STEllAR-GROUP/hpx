//    Copyright (c) 2004 Hartmut Kaiser
//
//    SPDX-License-Identifier: BSL-1.0
//    Use, modification and distribution is subject to the Boost Software
//    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//    http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/program_options/config/defines.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/any.hpp
// hpxinspect:nodeprecatedname:boost::any
// hpxinspect:nodeprecatedinclude:boost/optional.hpp
// hpxinspect:nodeprecatedname:boost::optional
// hpxinspect:nodeprecatedinclude:boost/program_options/config.hpp

#include <boost/any.hpp>
#include <boost/optional.hpp>
#include <boost/program_options/config.hpp>

namespace hpx { namespace program_options {

    using any = boost::any;
    using boost::any_cast;
    template <typename T>
    using optional = boost::optional<T>;

#define PROGRAM_OPTIONS_DEPRECATED_MESSAGE                                     \
    "The Boost.ProgramOptions was replaced by an equivalent "                  \
    "HPX ProgramOptions module that exposes a similar API."                    \
    "Please consider changing your code to use that module instead."           \
    "The easiest way to achieve this is to switch your includes "              \
    "to #include <hpx/program_options/*> and the related types to the "        \
    "namespace hpx::program_options."

}}    // namespace hpx::program_options
#else
#include <hpx/datastructures/any.hpp>
#include <hpx/datastructures/optional.hpp>

namespace hpx { namespace program_options {

    using any = hpx::any_nonser;
    using hpx::any_cast;
    template <typename T>
    using optional = hpx::util::optional<T>;
}}    // namespace hpx::program_options
#endif
