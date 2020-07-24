//  Copyright (c) 2012 Maciej Brodowicz
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/os_thread_type.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

#if defined(HPX_HAVE_PAPI) && defined(__linux__) && !defined(__ANDROID) &&     \
    !defined(ANDROID)
#include <sys/syscall.h>
#endif

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    // enumerates active OS threads and maintains their metadata

    class thread_mapper;

    namespace detail {

        // type for callback function invoked when thread is unregistered
        using thread_mapper_callback_type =
            util::function_nonser<bool(std::uint32_t)>;

        // thread-specific data
        class HPX_EXPORT os_thread_data
        {
        public:
            os_thread_data() = default;
            os_thread_data(
                std::string const& label, runtime_local::os_thread_type type);

        protected:
            friend class util::thread_mapper;

            void invalidate();
            bool is_valid() const;

        private:
            // label of this thread
            std::string label_;

            // associated thread ID, typically an ID of a kernel thread
            std::thread::id id_;

            // the native_handle() of the associated thread
            std::uint64_t tid_;

#if defined(HPX_HAVE_PAPI) && defined(__linux__) && !defined(__ANDROID) &&     \
    !defined(ANDROID)
            // the Linux thread id (required by PAPI)
            pid_t linux_tid_;
#endif

            // callback function invoked when unregistering a thread
            thread_mapper_callback_type cleanup_;

            // type of this OS thread in the context of the runtime
            runtime_local::os_thread_type type_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT thread_mapper
    {
    public:
        HPX_NON_COPYABLE(thread_mapper);

    public:
        using callback_type = detail::thread_mapper_callback_type;

        // erroneous thread index
        static constexpr std::uint32_t invalid_index = std::uint32_t(-1);

        // erroneous low-level thread ID
        static constexpr std::uint64_t invalid_tid = std::uint64_t(-1);

    public:
        thread_mapper();
        ~thread_mapper();

        ///////////////////////////////////////////////////////////////////////
        // registers invoking OS thread with a unique label
        std::uint32_t register_thread(
            char const* label, runtime_local::os_thread_type type);

        // unregisters the calling OS thread
        bool unregister_thread();

        ///////////////////////////////////////////////////////////////////////
        // returns unique index based on registered thread label
        std::uint32_t get_thread_index(std::string const& label) const;

        // returns the number of threads registered so far
        std::uint32_t get_thread_count() const;

        ///////////////////////////////////////////////////////////////////////
        // register callback function for a thread, invoked when unregistering
        // that thread
        bool register_callback(std::uint32_t tix, callback_type const&);

        // cancel callback
        bool revoke_callback(std::uint32_t tix);

        // returns thread id
        std::thread::id get_thread_id(std::uint32_t tix) const;

        // returns low level thread id (native_handle)
        std::uint64_t get_thread_native_handle(std::uint32_t tix) const;

#if defined(HPX_HAVE_PAPI) && defined(__linux__) && !defined(__ANDROID) &&     \
    !defined(ANDROID)
        pid_t get_linux_thread_id(std::uint32_t tix) const;
#endif

        // returns the label of registered thread tix
        std::string const& get_thread_label(std::uint32_t tix) const;

        // returns the type of the registered thread
        runtime_local::os_thread_type get_thread_type(std::uint32_t tix) const;

        // enumerate all registered OS threads
        bool enumerate_os_threads(
            util::function_nonser<bool(os_thread_data const&)> const& f) const;

        // retrieve all data stored for a given thread
        os_thread_data get_os_thread_data(std::string const& label) const;

    private:
        using mutex_type = hpx::lcos::local::spinlock;

        using thread_map_type = std::vector<detail::os_thread_data>;
        using label_map_type = std::map<std::string, std::size_t>;

        // main lock
        mutable mutex_type mtx_;

        // mapping from thread IDs to thread indices
        thread_map_type thread_map_;

        // mapping between HPX thread labels and thread indices
        label_map_type label_map_;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
