//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// appease inspect: hpxinspect:nominmax

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>

#include <hpx/config/warnings_prefix.hpp>

#ifdef DOXYGEN
namespace hpx {
    ///
    /// \brief A binary semaphore is a semaphore object that has only two
    ///        states. \a binary_semaphore is an alias for specialization of
    ///        \a hpx::counting_semaphore with \a LeastMaxValue being 1.
    ///        HPX's implementation of \a binary_semaphore is more efficient
    ///        than the default implementation of a counting semaphore with a
    ///        unit resource count (\a hpx::counting_semaphore).
    ///
    class binary_semaphore
    {
    public:
        binary_semaphore(binary_semaphore const&) = delete;
        binary_semaphore& operator=(binary_semaphore const&) = delete;
        binary_semaphore(binary_semaphore&&) = delete;
        binary_semaphore& operator=(binary_semaphore&&) = delete;

    public:
        ///
        /// \brief Constructs an object of type \a hpx::binary_semaphore
        ///        with the internal counter initialized to \a value.
        ///
        /// \param value The initial value of the internal semaphore lock
        ///              count. Normally this value should be zero (which
        ///              is the default), values greater than zero are
        ///              equivalent to the same number of signals pre-set,
        ///              and negative values are equivalent to the same
        ///              number of waits pre-set.
        ///
        explicit binary_semaphore(std::ptrdiff_t value = 1);

        ~binary_semaphore() = default;

        /// \copydoc hpx::counting_semaphore::max()
        static constexpr std::ptrdiff_t max() noexcept;

        /// \copydoc hpx::counting_semaphore::release()
        void release(std::ptrdiff_t update = 1);

        /// \copydoc hpx::counting_semaphore::try_acquire()
        bool try_acquire() noexcept;

        /// \copydoc hpx::counting_semaphore::acquire()
        void acquire();

        /// \copydoc hpx::counting_semaphore::try_acquire_until()
        bool try_acquire_until(hpx::chrono::steady_time_point const& abs_time);

        /// \copydoc hpx::counting_semaphore::try_acquire_for()
        bool try_acquire_for(hpx::chrono::steady_duration const& rel_time);
    };
}    // namespace hpx
#else

////////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // A binary semaphore should be more efficient than the default
    // implementation of a counting semaphore with a unit resource count.
    namespace detail {

        template <typename Mutex = hpx::spinlock>
        class binary_semaphore : public counting_semaphore<1, Mutex>
        {
        public:
            binary_semaphore(binary_semaphore const&) = delete;
            binary_semaphore& operator=(binary_semaphore const&) = delete;
            binary_semaphore(binary_semaphore&&) = delete;
            binary_semaphore& operator=(binary_semaphore&&) = delete;

        public:
            explicit binary_semaphore(std::ptrdiff_t value = 1)
              : counting_semaphore<1, Mutex>(value)
            {
            }

            ~binary_semaphore() = default;
        };
    }    // namespace detail

    using binary_semaphore = detail::binary_semaphore<>;
}    // namespace hpx

/// \cond NOINTERN
namespace hpx::lcos::local {

    template <typename Mutex = hpx::spinlock>
    using cpp20_binary_semaphore HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::cpp20_binary_semaphore is deprecated, use "
        "hpx::binary_semaphore instead") = hpx::detail::binary_semaphore<Mutex>;
}
/// \endcond

#endif

#include <hpx/config/warnings_suffix.hpp>
