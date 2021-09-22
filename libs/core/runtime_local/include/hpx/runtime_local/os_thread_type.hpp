//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>
#include <string>
#include <thread>

namespace hpx { namespace runtime_local {

    ///////////////////////////////////////////////////////////////////////////
    /// Types of kernel threads registered with the runtime
    enum class os_thread_type
    {
        unknown = -1,
        main_thread = 0,    ///< kernel thread represents main thread
        worker_thread,      ///< kernel thread is used to schedule HPX threads
        io_thread,          ///< kernel thread can be used for IO operations
        timer_thread,       ///< kernel is used by timer operations
        parcel_thread,      ///< kernel is used by networking operations
        custom_thread       ///< kernel is registered by the application
    };

    /// Return a human-readable name representing one of the kernel thread types
    HPX_CORE_EXPORT std::string get_os_thread_type_name(os_thread_type type);

    ///////////////////////////////////////////////////////////////////////////
    /// Registration data for kernel threads that is maintained by the runtime
    /// internally
    struct os_thread_data
    {
        std::string label_;     ///< name used for thread registration
        std::thread::id id_;    ///< thread id of corresponding kernel thread
        std::uint64_t native_handle_;    ///< the threads native handle
        os_thread_type type_;            ///< HPX thread type
    };

}}    // namespace hpx::runtime_local

///////////////////////////////////////////////////////////////////////////////
// These functions are officially part of the API
namespace hpx {

    using runtime_local::os_thread_data;
    using runtime_local::os_thread_type;

    using runtime_local::get_os_thread_type_name;
}    // namespace hpx
