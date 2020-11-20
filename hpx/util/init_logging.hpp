//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime_configuration/ini.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    /// The init_logging type will be used for initialization purposes only as
    /// well.
    HPX_EXPORT void init_logging(runtime_configuration& ini, bool isconsole);
}}}

