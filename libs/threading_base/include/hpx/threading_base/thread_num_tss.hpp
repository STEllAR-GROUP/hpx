//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail {
    // set/get the global thread Id to/from thread local storage
    HPX_EXPORT std::size_t set_thread_num_tss(std::size_t num);
    HPX_EXPORT std::size_t get_thread_num_tss();

    // this struct holds the local thread Id and the pool index
    // associated with the thread
    struct thread_pool
    {
        std::size_t local_thread_num;
        std::size_t pool_index;
    };
    HPX_EXPORT void set_thread_pool_tss(const thread_pool&);
    HPX_EXPORT thread_pool get_thread_pool_tss();

    ///////////////////////////////////////////////////////////////////////////
    struct reset_tss_helper
    {
        reset_tss_helper(std::size_t thread_num)
          : thread_num_(set_thread_num_tss(thread_num))
        {
        }

        ~reset_tss_helper()
        {
            set_thread_num_tss(thread_num_);
        }

        std::size_t previous_thread_num() const
        {
            return thread_num_;
        }

    private:
        std::size_t thread_num_;
    };
}}}    // namespace hpx::threads::detail

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

#include <hpx/config/warnings_suffix.hpp>

#endif
