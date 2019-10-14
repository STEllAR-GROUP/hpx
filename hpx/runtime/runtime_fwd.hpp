//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/runtime_fwd.hpp

#ifndef HPX_RUNTIME_RUNTIME_FWD_HPP
#define HPX_RUNTIME_RUNTIME_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>

namespace hpx
{
    class HPX_API_EXPORT runtime;

    ///////////////////////////////////////////////////////////////////////////
    class HPX_API_EXPORT runtime_impl;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    HPX_API_EXPORT runtime& get_runtime();
    HPX_API_EXPORT runtime*& get_runtime_ptr();

    /// Return true if networking is enabled.
    ///
    /// \note Networking is enabled if `-DHPX_WITH_NETWORKING=On` was used at
    ///       configuration time and more than one locality is used or the
    ///       command line option `--hpx:expect-connecting-localities` was
    ///       specified
    HPX_API_EXPORT bool is_networking_enabled();
}

#endif
