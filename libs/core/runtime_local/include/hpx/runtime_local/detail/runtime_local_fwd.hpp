//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime_local/runtime_local_fwd.hpp

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    class HPX_CORE_EXPORT runtime;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    HPX_CORE_EXPORT runtime& get_runtime();
    HPX_CORE_EXPORT runtime*& get_runtime_ptr();

    /// Return true if networking is enabled.
    ///
    /// \note Networking is enabled if `-DHPX_WITH_NETWORKING=On` was used at
    ///       configuration time and more than one locality is used or the
    ///       command line option `--hpx:expect-connecting-localities` was
    ///       specified
    HPX_CORE_EXPORT bool is_networking_enabled();
}    // namespace hpx
