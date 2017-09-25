//  Copyright (c) 2007-2014 Hartmut Kaiser
//
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
    HPX_API_EXPORT runtime* get_runtime_ptr();
}

#endif
