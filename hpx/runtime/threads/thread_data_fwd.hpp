//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_data_fwd.hpp

#if !defined(HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM)
#define HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx {
    /// \cond NOINTERNAL
    class HPX_EXPORT thread;
    /// \endcond
}    // namespace hpx

namespace hpx { namespace threads {
    /// \cond NOINTERNAL
    struct HPX_EXPORT topology;

    class HPX_EXPORT executor;
    /// \endcond

    /// The function \a get_thread_count returns the number of currently
    /// known threads.
    ///
    /// \param state    [in] This specifies the thread-state for which the
    ///                 number of threads should be retrieved.
    ///
    /// \note If state == unknown this function will not only return the
    ///       number of currently existing threads, but will add the number
    ///       of registered task descriptions (which have not been
    ///       converted into threads yet).
    HPX_API_EXPORT std::int64_t get_thread_count(
        thread_state_enum state = unknown);

    /// The function \a get_thread_count returns the number of currently
    /// known threads.
    ///
    /// \param priority [in] This specifies the thread-priority for which the
    ///                 number of threads should be retrieved.
    /// \param state    [in] This specifies the thread-state for which the
    ///                 number of threads should be retrieved.
    ///
    /// \note If state == unknown this function will not only return the
    ///       number of currently existing threads, but will add the number
    ///       of registered task descriptions (which have not been
    ///       converted into threads yet).
    HPX_API_EXPORT std::int64_t get_thread_count(
        thread_priority priority, thread_state_enum state = unknown);

    /// The function \a enumerate_threads will invoke the given function \a f
    /// for each thread with a matching thread state.
    ///
    /// \param f        [in] The function which should be called for each
    ///                 matching thread. Returning 'false' from this function
    ///                 will stop the enumeration process.
    /// \param state    [in] This specifies the thread-state for which the
    ///                 threads should be enumerated.
    HPX_API_EXPORT bool enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state = unknown);
}}    // namespace hpx::threads

namespace std {
    template <>
    struct hash<::hpx::threads::thread_id>
    {
        std::size_t operator()(::hpx::threads::thread_id const& v) const
            noexcept
        {
            std::hash<::hpx::threads::thread_data const*> hasher_;
            return hasher_(static_cast<::hpx::threads::thread_data*>(v.get()));
        }
    };
}    // namespace std

#endif
