//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/get_worker_thread_num.hpp

#if !defined(HPX_RUNTIME_GET_WORKER_THREAD_NUM_AUG_15_2015_1120AM)
#define HPX_RUNTIME_GET_WORKER_THREAD_NUM_AUG_15_2015_1120AM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>

#include <cstddef>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the current OS-thread running in the
    ///        runtime instance the current HPX-thread is executed with.
    ///
    /// This function returns the zero based index of the OS-thread which
    /// executes the current HPX-thread.
    ///
    /// \note   The returned value is zero based and its maximum value is
    ///         smaller than the overall number of OS-threads executed (as
    ///         returned by \a get_os_thread_count().
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         fail otherwise (it will return -1).
    HPX_API_EXPORT std::size_t get_worker_thread_num();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the current OS-thread running in the
    ///        runtime instance the current HPX-thread is executed with.
    ///
    /// This function returns the zero based index of the OS-thread which
    /// executes the current HPX-thread.
    ///
    /// \param ec [in,out] this represents the error status on exit.
    ///
    /// \note   The returned value is zero based and its maximum value is
    ///         smaller than the overall number of OS-threads executed (as
    ///         returned by \a get_os_thread_count(). It will return -1 if
    ///         the current thread is not a known thread or if the runtime is
    ///         not in running state.
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         fail otherwise (it will return -1).
    HPX_API_EXPORT std::size_t get_worker_thread_num(error_code& ec);
}

#endif
