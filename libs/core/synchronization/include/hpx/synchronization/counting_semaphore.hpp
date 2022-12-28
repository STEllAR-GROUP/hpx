//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// appease inspect: hpxinspect:nominmax

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/detail/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

#ifdef DOXYGEN
namespace hpx {
    ///
    /// \brief A semaphore is a protected variable (an entity storing a
    ///        value) or abstract data type (an entity grouping several
    ///        variables that may or may not be numerical) which
    ///        constitutes the classic method for restricting access to
    ///        shared resources, such as shared memory, in a
    ///        multiprogramming environment. Semaphores exist in many
    ///        variants, though usually the term refers to a counting
    ///        semaphore, since a binary semaphore is better known as a
    ///        mutex. A counting semaphore is a counter for a set of
    ///        available resources, rather than a locked/unlocked flag of a
    ///        single resource. It was invented by Edsger Dijkstra.
    ///        Semaphores are the classic solution to preventing race
    ///        conditions in the dining philosophers problem, although they
    ///        do not prevent resource deadlocks.
    ///
    ///        Counting semaphores can be used for synchronizing multiple
    ///        threads as well: one thread waiting for several other
    ///        threads to touch (signal) the semaphore, or several threads
    ///        waiting for one other thread to touch this semaphore. Unlike
    ///        \a hpx::mutex a \a counting_semaphore is not tied to threads
    ///        of execution -- acquiring a semaphore can occur on a
    ///        different thread than releasing the semaphore, for example.
    ///        All operations on \a counting_semaphore can be performed
    ///        concurrently and without any relation to specific threads of
    ///        execution, with the exception of the destructor which cannot
    ///        be performed concurrently but can be performed on a different
    ///        thread.
    ///
    ///        Semaphores are lightweight synchronization primitives used
    ///        to constrain concurrent access to a shared resource. They
    ///        are widely used to implement other synchronization
    ///        primitives and, whenever both are applicable, can be more
    ///        efficient than condition variables.
    ///
    ///        A counting semaphore is a semaphore object that models a
    ///        non-negative resource count.
    ///
    ///        Class template \a counting_semaphore maintains an internal
    ///        counter that is initialized when the semaphore is created.
    ///        The counter is decremented when a thread acquires the
    ///        semaphore, and is incremented when a thread releases the
    ///        semaphore. If a thread tries to acquire the semaphore when
    ///        the counter is zero, the thread will block until another
    ///        thread increments the counter by releasing the semaphore.
    ///
    ///        Specializations of \a hpx::counting_semaphore are not
    ///        \namedrequirement{DefaultConstructible},
    ///        \namedrequirement{CopyConstructible},
    ///        \namedrequirement{MoveConstructible},
    ///        \namedrequirement{CopyAssignable},
    ///        or \namedrequirement{MoveAssignable}.
    ///
    /// \note \a couting_semaphore's \a try_acquire() can spuriously fail.
    ///
    /// \tparam LeastMaxValue \a counting_semaphore allows more than one
    ///                        concurrent access to the same resource, for
    ///                        at least \a LeastMaxValue concurrent
    ///                        accessors. As its name indicates, the
    ///                        \a LeastMaxValue is the minimum max value,
    ///                        not the actual max value. Thus \a max() can
    ///                        yield a number larger than \a LeastMaxValue.
    ///
    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX>
    class counting_semaphore
    {
    public:
        counting_semaphore(counting_semaphore const&) = delete;
        counting_semaphore& operator=(counting_semaphore const&) = delete;
        counting_semaphore(counting_semaphore&&) = delete;
        counting_semaphore& operator=(counting_semaphore&&) = delete;

    public:
        ///
        /// \brief Returns The maximum value of counter. This value is
        ///        greater than or equal to \a LeastMaxValue.
        ///
        /// \returns The internal counter's maximum possible value, as a
        ///          \a std::ptrdiff_t.
        ///
        static constexpr std::ptrdiff_t max() noexcept;

        ///
        /// \brief Constructs an object of type \a hpx::counting_semaphore
        ///        with the internal counter initialized to \a value.
        ///
        /// \param value The initial value of the internal semaphore lock
        ///              count. Normally this value should be zero (which
        ///              is the default), values greater than zero are
        ///              equivalent to the same number of signals pre-set,
        ///              and negative values are equivalent to the same
        ///              number of waits pre-set.
        ///
        explicit counting_semaphore(std::ptrdiff_t value);

        ~counting_semaphore() = default;

        ///
        /// \brief Atomically increments the internal counter by the value
        ///        of update. Any thread(s) waiting for the counter to be
        ///        greater than 0, such as due to being blocked in acquire,
        ///        will subsequently be unblocked.
        ///
        /// \pre Both `update >= 0` and `update <= max() - counter` are
        ///      \a true, where \a counter is the value of the internal
        ///      counter.
        ///
        /// \note Synchronization: Strongly happens before invocations of
        ///                        \a try_acquire that observe the result
        ///                        of the effects.
        ///
        /// \throws std::system_error
        ///
        /// \param update the amount to increment the internal counter by
        ///
        void release(std::ptrdiff_t update = 1);

        ///
        /// \brief Tries to atomically decrement the internal counter by 1
        ///        if it is greater than 0; no blocking occurs regardless.
        ///
        /// \return \a true if it decremented the internal counter,
        ///         otherwise \a false
        ///
        bool try_acquire() noexcept;

        ///
        /// \brief Repeatedly performs the following steps, in order:
        ///            - Evaluates try_acquire. If the result is true,
        ///              returns.
        ///            - Blocks on *this until counter is greater than
        ///              zero.
        ///
        /// \throws std::system_error
        ///
        /// \returns \a void.
        void acquire();

        ///
        /// \brief Tries to atomically decrement the internal counter by 1
        ///        if it is greater than 0; otherwise blocks until it is
        ///        greater than 0 and can successfully decrement the
        ///        internal counter, or the \a abs_time time point has been
        ///        passed.
        ///
        /// \param abs_time the earliest time the function must wait until
        ///                 in order to fail
        ///
        /// \throws std::system_error
        ///
        /// \return \a true if it decremented the internal counter,
        ///         otherwise \a false.
        ///
        bool try_acquire_until(hpx::chrono::steady_time_point const& abs_time);

        ///
        /// \brief Tries to atomically decrement the internal counter by 1
        ///        if it is greater than 0; otherwise blocks until it is
        ///        greater than 0 and can successfully decrement the
        ///        internal counter, or the \a rel_time duration has been
        ///        exceeded.
        ///
        /// \throws std::system_error
        ///
        /// \param rel_time the minimum duration the function must wait
        ///                 for to fail
        ///
        /// \return \a true if it decremented the internal counter,
        ///         otherwise false
        ///
        bool try_acquire_for(hpx::chrono::steady_duration const& rel_time);
    };

    ///
    /// A semaphore is a protected variable (an entity storing a value) or
    /// abstract data type (an entity grouping several variables that may or may
    /// not be numerical) which constitutes the classic method for restricting
    /// access to shared resources, such as shared memory, in a multiprogramming
    /// environment. Semaphores exist in many variants, though usually the term
    /// refers to a counting semaphore, since a binary semaphore is better known
    /// as a mutex. A counting semaphore is a counter for a set of available
    /// resources, rather than a locked/unlocked flag of a single resource. It
    /// was invented by Edsger Dijkstra. Semaphores are the classic solution to
    /// preventing race conditions in the dining philosophers problem, although
    /// they do not prevent resource deadlocks.
    ///
    /// Counting semaphores can be used for synchronizing multiple threads as
    /// well: one thread waiting for several other threads to touch (signal) the
    /// semaphore, or several threads waiting for one other thread to touch this
    /// semaphore. Unlike
    /// \a hpx::mutex a \a counting_semaphore_var is not tied to threads
    /// of execution -- acquiring a semaphore can occur on a different thread
    /// than releasing the semaphore, for example. All operations on \a
    /// counting_semaphore_var can be performed concurrently and without any
    /// relation to specific threads of execution, with the exception of the
    /// destructor which cannot be performed concurrently but can be performed
    /// on a different thread.
    ///
    /// Semaphores are lightweight synchronization primitives used to constrain
    /// concurrent access to a shared resource. They are widely used to
    /// implement other synchronization primitives and, whenever both are
    /// applicable, can be more efficient than condition variables.
    ///
    /// A counting semaphore is a semaphore object that models a non-negative
    /// resource count.
    ///
    /// Class template \a counting_semaphore_var maintains an internal counter
    /// that is initialized when the semaphore is created. The counter is
    /// decremented when a thread acquires the semaphore, and is incremented
    /// when a thread releases the semaphore. If a thread tries to acquire the
    /// semaphore when the counter is zero, the thread will block until another
    /// thread increments the counter by releasing the semaphore.
    ///
    /// Specializations of \a hpx::counting_semaphore_var are not
    /// \namedrequirement{DefaultConstructible},
    /// \namedrequirement{CopyConstructible},
    /// \namedrequirement{MoveConstructible}, \namedrequirement{CopyAssignable},
    /// or \namedrequirement{MoveAssignable}.
    ///
    /// \note \a counting_semaphore_var's \a try_acquire() can spuriously fail.
    ///
    /// \tparam Mutex Type of mutex
    /// \tparam N     The initial value of the internal semaphore lock
    ///               count.
    ///
    template <typename Mutex = hpx::spinlock, int N = 0>
    class counting_semaphore_var
    {
    private:
        using mutex_type = Mutex;

    public:
        ///
        /// Constructs an object of type \a hpx::counting_semaphore_value with
        /// the internal counter initialized to \a N.
        ///
        /// \param value The initial value of the internal semaphore lock
        ///              count. Normally this value should be zero, values
        ///              greater than zero are equivalent to the same number of
        ///              signals pre-set, and negative values are equivalent to
        ///              the same number of waits pre-set. Defaults to \a N
        ///              (which in turn defaults to zero).
        ///
        explicit counting_semaphore_var(std::ptrdiff_t value = N);

        counting_semaphore_var(counting_semaphore_var const&) = delete;
        counting_semaphore_var& operator=(
            counting_semaphore_var const&) = delete;

        ///
        /// \brief Wait for the semaphore to be signaled.
        ///
        /// \param count The value by which the internal lock count will be
        ///              decremented. At the same time this is the minimum
        ///              value of the lock count at which the thread is not
        ///              yielded.
        ///
        void wait(std::ptrdiff_t count = 1);

        ///
        /// \brief Try to wait for the semaphore to be signaled
        ///
        /// \param count The value by which the internal lock count will be
        ///              decremented. At the same time this is the minimum
        ///              value of the lock count at which the thread is not
        ///              yielded.
        ///
        /// \return \a try_wait returns true if the calling thread was able to
        ///         acquire the requested amount of credits.
        ///         \a try_wait returns false if not sufficient credits are
        ///         available at this point in time.
        ///
        bool try_wait(std::ptrdiff_t count = 1);

        ///
        /// \brief Signal the semaphore.
        ///
        /// \param count The value by which the internal lock count will be
        ///              incremented.
        ///
        void signal(std::ptrdiff_t count = 1);

        ///
        /// \brief Unblock all acquirers.
        ///
        /// \return std::ptrdiff_t internal lock count after the operation.
        ///
        std::ptrdiff_t signal_all();

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

namespace hpx {

    namespace detail {

        template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX,
            typename Mutex = hpx::spinlock>
        class counting_semaphore
        {
        public:
            counting_semaphore(counting_semaphore const&) = delete;
            counting_semaphore& operator=(counting_semaphore const&) = delete;
            counting_semaphore(counting_semaphore&&) = delete;
            counting_semaphore& operator=(counting_semaphore&&) = delete;

        protected:
            using mutex_type = Mutex;

        public:
            static constexpr std::ptrdiff_t(max)() noexcept
            {
                return LeastMaxValue;
            }

            explicit counting_semaphore(std::ptrdiff_t value) noexcept
              : sem_(value)
            {
            }

            ~counting_semaphore() = default;

            void release(std::ptrdiff_t update = 1)
            {
                std::unique_lock<mutex_type> l(mtx_);
                sem_.signal(HPX_MOVE(l), update);
            }

            bool try_acquire() noexcept
            {
                std::unique_lock<mutex_type> l(mtx_);
                return sem_.try_acquire(l);
            }

            void acquire()
            {
                std::unique_lock<mutex_type> l(mtx_);
                sem_.wait(l, 1);
            }

            bool try_acquire_until(
                hpx::chrono::steady_time_point const& abs_time)
            {
                std::unique_lock<mutex_type> l(mtx_);
                return sem_.wait_until(l, abs_time, 1);
            }

            bool try_acquire_for(hpx::chrono::steady_duration const& rel_time)
            {
                return try_acquire_until(rel_time.from_now());
            }

        protected:
            mutable mutex_type mtx_;
            hpx::lcos::local::detail::counting_semaphore sem_;
        };
    }    // namespace detail

    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX>
    using counting_semaphore = detail::counting_semaphore<LeastMaxValue>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = hpx::spinlock, int N = 0>
    class counting_semaphore_var
      : public detail::counting_semaphore<PTRDIFF_MAX, Mutex>
    {
    private:
        using mutex_type = Mutex;

    public:
        explicit counting_semaphore_var(std::ptrdiff_t value = N) noexcept
          : detail::counting_semaphore<PTRDIFF_MAX, Mutex>(value)
        {
        }

        counting_semaphore_var(counting_semaphore_var const&) = delete;
        counting_semaphore_var& operator=(
            counting_semaphore_var const&) = delete;

        void wait(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            this->sem_.wait(l, count);
        }

        bool try_wait(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            return this->sem_.try_wait(l, count);
        }

        void signal(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            this->sem_.signal(HPX_MOVE(l), count);
        }

        std::ptrdiff_t signal_all()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            return this->sem_.signal_all(HPX_MOVE(l));
        }
    };
}    // namespace hpx

/// \cond NOINTERN
namespace hpx::lcos::local {

    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX,
        typename Mutex = hpx::spinlock>
    using cpp20_counting_semaphore HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::cpp20_counting_semaphore is deprecated, use "
        "hpx::counting_semaphore instead") =
        hpx::detail::counting_semaphore<LeastMaxValue, Mutex>;

    template <typename Mutex = hpx::spinlock, int N = 0>
    using counting_semaphore_var HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::counting_semaphore_var is deprecated, use "
        "hpx::counting_semaphore_var instead") =
        hpx::counting_semaphore_var<Mutex, N>;

    using counting_semaphore HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::counting_semaphore is deprecated, use "
        "hpx::counting_semaphore_var<> instead") =
        hpx::counting_semaphore_var<>;
}    // namespace hpx::lcos::local
     /// \endcond

#endif

#include <hpx/config/warnings_suffix.hpp>
