//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2006-7 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_RECURSIVE_MUTEX_AUG_03_2009_0459PM)
#define HPX_LCOS_RECURSIVE_MUTEX_AUG_03_2009_0459PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Mutex>
        struct thread_id_from_mutex
        {
            typedef std::size_t thread_id_type;

            static thread_id_type invalid_id() { return thread_id_type(~0u); }

            static thread_id_type call()
            {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
                return (thread_id_type)GetCurrentThreadId();
#else
                return (thread_id_type)pthread_self();
#endif
            };
        };

        template <>
        struct thread_id_from_mutex<lcos::local::spinlock>
        {
            typedef std::size_t thread_id_type;

            static thread_id_type invalid_id() { return thread_id_type(~0u); }

            static thread_id_type call()
            {
                return hpx::threads::get_self_ptr() ?
                    (thread_id_type)hpx::threads::get_self_id().get() :
                    thread_id_from_mutex<void>::call();
            };
        };

        /// An exclusive-ownership recursive mutex which implements Boost.Thread's
        /// TimedLockable concept.
        template <typename Mutex = local::spinlock>
        struct recursive_mutex_impl
        {
            HPX_NON_COPYABLE(recursive_mutex_impl);

        private:
            typedef typename thread_id_from_mutex<Mutex>::thread_id_type
                thread_id_type;

            boost::atomic<std::uint64_t> recursion_count;
            boost::atomic<thread_id_type> locking_thread_id;
            Mutex mtx;

        public:
            recursive_mutex_impl(char const* const desc = "recursive_mutex_impl")
              : recursion_count(0)
              , locking_thread_id(thread_id_from_mutex<Mutex>::invalid_id())
              , mtx(desc)
            {}

            /// Attempts to acquire ownership of the \a recursive_mutex.
            /// Never blocks.
            ///
            /// \returns \a true if ownership was acquired; otherwise, \a false.
            ///
            /// \throws Never throws.
            bool try_lock()
            {
                thread_id_type const id = thread_id_from_mutex<Mutex>::call();
                HPX_ASSERT(id != thread_id_from_mutex<Mutex>::invalid_id());

                return try_recursive_lock(id) || try_basic_lock(id);
            }

            /// Acquires ownership of the \a recursive_mutex. Suspends the
            /// current HPX-thread if ownership cannot be obtained immediately.
            ///
            /// \throws Throws \a hpx#bad_parameter if an error occurs while
            ///         suspending. Throws \a hpx#yield_aborted if the mutex is
            ///         destroyed while suspended. Throws \a hpx#null_thread_id if
            ///         called outside of a HPX-thread.
            void lock()
            {
                thread_id_type const id = thread_id_from_mutex<Mutex>::call();
                HPX_ASSERT(id != thread_id_from_mutex<Mutex>::invalid_id());

                if (!try_recursive_lock(id))
                {
                    mtx.lock();
                    locking_thread_id.exchange(id);
                    util::ignore_lock(&mtx);
                    util::register_lock(this);
                    recursion_count.store(1);
                }
            }

            /// Attempts to acquire ownership of the \a recursive_mutex.
            /// Suspends the current HPX-thread until \a wait_until if ownership
            /// cannot be obtained immediately.
            ///
            /// \returns \a true if ownership was acquired; otherwise, \a false.
            ///
            /// \throws Throws \a hpx#bad_parameter if an error occurs while
            ///         suspending. Throws \a hpx#yield_aborted if the mutex is
            ///         destroyed while suspended. Throws \a hpx#null_thread_id if
            ///         called outside of a HPX-thread.
//             bool timed_lock(::boost::system_time const& wait_until);
//             {
//                 threads::thread_id_repr_type const current_thread_id =
//                     threads::get_self_id().get();
//
//                 return try_recursive_lock(current_thread_id) ||
//                        try_timed_lock(current_thread_id, wait_until);
//             }

            /// Attempts to acquire ownership of the \a recursive_mutex.
            /// Suspends the current HPX-thread until \a timeout if ownership cannot
            /// be obtained immediately.
            ///
            /// \returns \a true if ownership was acquired; otherwise, \a false.
            ///
            /// \throws Throws \a hpx#bad_parameter if an error occurs while
            ///         suspending. Throws \a hpx#yield_aborted if the mutex is
            ///         destroyed while suspended. Throws \a hpx#null_thread_id if
            ///         called outside of a HPX-thread.
//             template<typename Duration>
//             bool timed_lock(Duration const& timeout)
//             {
//                 return timed_lock(boost::get_system_time() + timeout);
//             }
//
//             bool timed_lock(boost::xtime const& timeout)
//             {
//                 return timed_lock(boost::posix_time::ptime(timeout));
//             }
//
            /// Release ownership of the \a recursive_mutex.
            ///
            /// \throws Throws \a hpx#bad_parameter if an error occurs while
            ///         releasing the mutex. Throws \a hpx#null_thread_id if called
            ///         outside of a HPX-thread.
            void unlock()
            {
                if (0 == --recursion_count)
                {
                    locking_thread_id.exchange(
                        thread_id_from_mutex<Mutex>::invalid_id());
                    util::unregister_lock(this);
                    util::reset_ignored(&mtx);
                    mtx.unlock();
                }
            }

        private:
            bool try_recursive_lock(thread_id_type current_thread_id)
            {
                if (locking_thread_id.load(boost::memory_order_acquire) ==
                    current_thread_id)
                {
                    if (++recursion_count == 1)
                        util::register_lock(this);
                    return true;
                }
                return false;
            }

            bool try_basic_lock(thread_id_type current_thread_id)
            {
                if (mtx.try_lock())
                {
                    locking_thread_id.exchange(current_thread_id);
                    util::ignore_lock(&mtx);
                    util::register_lock(this);
                    recursion_count.store(1);
                    return true;
                }
                return false;
            }

//             bool try_timed_lock(threads::thread_id_repr_type current_thread_id,
//                 ::boost::system_time const& target)
//             {
//                 if (mtx.timed_lock(target))
//                 {
//                     locking_thread_id.exchange(current_thread_id);
//                     recursion_count.store(1);
//                     return true;
//                 }
//                 return false;
//             }
        };
    }

    typedef detail::recursive_mutex_impl<> recursive_mutex;
}}}

#endif
