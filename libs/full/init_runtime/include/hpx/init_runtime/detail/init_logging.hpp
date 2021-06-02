//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/init_runtime_local/detail/init_logging.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/runtime_configuration/runtime_configuration.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail {

    HPX_EXPORT void init_logging_full(runtime_configuration&);
}}}    // namespace hpx::util::detail

#endif
