///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2014-2016 MongoDB, Inc.
//  Copyright (c) 2008-2014 WiredTiger, Inc.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_LCOS_LOCAL_SHARED_SPINLOCK_HPP
#define HPX_LCOS_LOCAL_SHARED_SPINLOCK_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/yield_k.hpp>

#include <boost/atomic.hpp>

#include <mutex>

/*
 * Based on "Spinlocks and Read-Write Locks" by Dr. Steven Fuerst:
 *      http://locklessinc.com/articles/locks/
 *
 * Dr. Fuerst further credits:
 *      There exists a form of the ticket lock that is designed for read-write
 * locks. An example written in assembly was posted to the Linux kernel mailing
 * list in 2002 by David Howells from RedHat. This was a highly optimized
 * version of a read-write ticket lock developed at IBM in the early 90's by
 * Joseph Seigh. Note that a similar (but not identical) algorithm was published
 * by John Mellor-Crummey and Michael Scott in their landmark paper "Scalable
 * Reader-Writer Synchronization for Shared-Memory Multiprocessors".
 *
 * The following is an explanation of this code. First, the underlying lock
 * structure.
 *
 *      struct {
 *              uint16_t writers;       Now serving for writers
 *              uint16_t readers;       Now serving for readers
 *              uint16_t next;          Next available ticket number
 *              uint16_t __notused;     Padding
 *      }
 *
 * First, imagine a store's 'take a number' ticket algorithm. A customer takes
 * a unique ticket number and customers are served in ticket order. In the data
 * structure, 'writers' is the next writer to be served, 'readers' is the next
 * reader to be served, and 'next' is the next available ticket number.
 *
 * Next, consider exclusive (write) locks. The 'now serving' number for writers
 * is 'writers'. To lock, 'take a number' and wait until that number is being
 * served; more specifically, atomically copy and increment the current value of
 * 'next', and then wait until 'writers' equals that copied number.
 *
 * Shared (read) locks are similar. Like writers, readers atomically get the
 * next number available. However, instead of waiting for 'writers' to equal
 * their number, they wait for 'readers' to equal their number.
 *
 * This has the effect of queuing lock requests in the order they arrive
 * (incidentally avoiding starvation).
 *
 * Each lock/unlock pair requires incrementing both 'readers' and 'writers'.
 * In the case of a reader, the 'readers' increment happens when the reader
 * acquires the lock (to allow read-lock sharing), and the 'writers' increment
 * happens when the reader releases the lock. In the case of a writer, both
 * 'readers' and 'writers' are incremented when the writer releases the lock.
 *
 * For example, consider the following read (R) and write (W) lock requests:
 *
 *                                              writers readers next
 *                                              0       0       0
 *      R: ticket 0, readers match      OK      0       1       1
 *      R: ticket 1, readers match      OK      0       2       2
 *      R: ticket 2, readers match      OK      0       3       3
 *      W: ticket 3, writers no match   block   0       3       4
 *      R: ticket 2, unlock                     1       3       4
 *      R: ticket 0, unlock                     2       3       4
 *      R: ticket 1, unlock                     3       3       4
 *      W: ticket 3, writers match      OK      3       3       4
 *
 * Note the writer blocks until 'writers' equals its ticket number and it does
 * not matter if readers unlock in order or not.
 *
 * Readers or writers entering the system after the write lock is queued block,
 * and the next ticket holder (reader or writer) will unblock when the writer
 * unlocks. An example, continuing from the last line of the above example:
 *
 *                                              writers readers next
 *      W: ticket 3, writers match      OK      3       3       4
 *      R: ticket 4, readers no match   block   3       3       5
 *      R: ticket 5, readers no match   block   3       3       6
 *      W: ticket 6, writers no match   block   3       3       7
 *      W: ticket 3, unlock                     4       4       7
 *      R: ticket 4, readers match      OK      4       5       7
 *      R: ticket 5, readers match      OK      4       6       7
 *
 * The 'next' field is a 2-byte value so the available ticket number wraps at
 * 64K requests. If a thread's lock request is not granted until the 'next'
 * field cycles and the same ticket is taken by another thread, we could grant
 * a lock to two separate threads at the same time, and bad things happen: two
 * writer threads or a reader thread and a writer thread would run in parallel,
 * and lock waiters could be skipped if the unlocks race. This is unlikely, it
 * only happens if a lock request is blocked by 64K other requests. The fix is
 * to grow the lock structure fields, but the largest atomic instruction we have
 * is 8 bytes, the structure has no room to grow.
 */
// from http://locklessinc.com/articles/locks

namespace hpx { namespace lcos { namespace local {

    class shared_spinlock
    {
//         struct ticket
//         {
//             std::uint16_t writers; // Now serving for writers
//             std::uint16_t readers; // Now serving for readers
//             std::uint16_t next;    // Next available ticket number
//             std::uint16_t unused;  // padding...
//         };

        boost::atomic<std::uint64_t> ticket_;

        std::uint16_t get_writers(std::uint64_t ticket)
        {
            return ticket & 0xFFFF;
        }

        std::uint16_t get_readers(std::uint64_t ticket)
        {
            return (ticket >> 16) & 0xFFFF;
        }

        std::uint16_t get_next(std::uint64_t ticket)
        {
            return (ticket >> 32) & 0xFFFF;
        }

        std::uint64_t set(std::int64_t writers, std::int64_t readers, std::int64_t next)
        {
            return
                writers |
                (std::uint64_t(readers) << 16) |
                (std::uint64_t(next) << 32);
        }

        std::uint16_t next_ticket()
        {
            std::uint64_t val = set(0, 0, 1);
            std::uint64_t res = ticket_.fetch_add(val, boost::memory_order_acq_rel);
            return get_next(res);
        }

        bool compare_writer(std::uint16_t next)
        {
            std::uint64_t ticket = ticket_.load(boost::memory_order_acquire);
            return next == get_writers(ticket);
        }

        bool compare_reader(std::uint16_t next)
        {
            std::uint64_t ticket = ticket_.load(boost::memory_order_acquire);
            return next == get_readers(ticket);
        }

        HPX_NON_COPYABLE(shared_spinlock);

    public:

        /// Locks the spinlock. The spinlock is in unlocked state after the call
        shared_spinlock()
          : ticket_{0}
        {}

        /// Locks the spinlock, blocks if the spinlock is not available
        void lock()
        {
            std::uint16_t ticket = next_ticket();
            for (std::size_t k = 0; !compare_writer(ticket); ++k)
            {
                hpx::util::detail::yield_k(k, "hpx::lcos::local::shared_spinlock::lock");
            }
            util::register_lock(this);
        }

        /// Tries to lock the spinlock, returns false if the the spinlock is not
        /// available
        bool try_lock()
        {
            std::uint64_t old_ticket = ticket_.load(boost::memory_order_acquire);

            std::uint16_t old_writers = get_writers(old_ticket);
            std::uint16_t old_readers = get_readers(old_ticket);
            std::uint16_t old_next = get_next(old_ticket);

            // This write lock can only be granted if the lock was last granted to
            // a writer and there are no readers or writers blocked on the lock,
            // that is, if this thread's ticket would be the next ticket granted.
            // Do the cheap test to see if this can possibly succeed (and confirm
            // the lock is in the correct state to grant this write lock).
            if (old_writers != old_next)
                return false;

            std::uint16_t new_next = old_next + 1;
            std::uint64_t new_ticket = set(old_writers, old_readers, new_next);

            if (ticket_.compare_exchange_weak(old_ticket, new_ticket))
            {
                util::register_lock(this);
                return true;
            }

            return false;
        }

        /// Unlocks the spinlock
        void unlock()
        {
            std::uint64_t old_ticket = ticket_.load(boost::memory_order_acquire);
            std::uint64_t new_ticket = 0;
            std::size_t k = 0;
            do
            {
                std::uint16_t writers = get_writers(old_ticket);
                std::uint16_t readers = get_readers(old_ticket);
                std::uint16_t next = get_next(old_ticket);
                ++writers;
                ++readers;
                new_ticket = set(writers, readers, next);

                hpx::util::detail::yield_k(k, "hpx::lcos::local::shared_spinlock::unlock");
                ++k;
            }
            while (!ticket_.compare_exchange_weak(old_ticket, new_ticket));

            util::unregister_lock(this);
        }

        /// Locks the spinlock for shared ownership, blocks if the spinlock is
        /// not available
        void lock_shared()
        {
            std::uint16_t next = next_ticket();
            for (std::size_t k = 0; !compare_reader(next); ++k)
            {
                hpx::util::detail::yield_k(k, "hpx::lcos::local::shared_spinlock::lock_shared");
            }

            std::uint64_t old_ticket = ticket_.load(boost::memory_order_acquire);
            std::uint64_t new_ticket = 0;
            std::size_t k = 0;
            do {
                std::uint16_t writers = get_writers(old_ticket);
                std::uint16_t readers = get_readers(old_ticket) + 1;
                next = get_next(old_ticket);
                new_ticket = set(writers, readers, next);
                hpx::util::detail::yield_k(k, "hpx::lcos::local::shared_spinlock::lock_shared");
                ++k;
            }
            while (!ticket_.compare_exchange_weak(old_ticket, new_ticket));

            util::register_lock(this);
        }

        /// Tries to lock the spinlock for shared ownership, returns false if
        /// the spinlock is not available
        bool trylock_shared()
        {
            std::uint64_t old_ticket = ticket_.load(boost::memory_order_acquire);

            std::uint16_t old_writers = get_writers(old_ticket);
            std::uint16_t old_readers = get_readers(old_ticket);
            std::uint16_t old_next = get_next(old_ticket);

            // This read lock can only be granted if the lock was last granted to
            // a reader and there are no readers or writers blocked on the lock,
            // that is, if this thread's ticket would be the next ticket granted.
            // Do the cheap test to see if this can possibly succeed (and confirm
            // the lock is in the correct state to grant this read lock).
            if (old_readers != old_next)
                return false;

            std::uint64_t new_ticket = set(old_writers, old_next + 1, old_next + 1);

            if (ticket_.compare_exchange_weak(old_ticket, new_ticket))
            {
                util::register_lock(this);
                return true;
            }

            return false;
        }

        /// Unloacks the spinlock (shared ownership)
        void unlock_shared()
        {
            std::uint64_t old_ticket = ticket_.load(boost::memory_order_acquire);
            std::uint64_t new_ticket = 0;
            std::size_t k = 0;
            do
            {
                std::uint16_t writers = get_writers(new_ticket);
                std::uint16_t readers = get_readers(new_ticket);
                std::uint16_t next = get_next(new_ticket);
                new_ticket = set(writers + 1, readers, next);

                hpx::util::detail::yield_k(k, "hpx::lcos::local::shared_spinlock::unlock_shared");
                ++k;
            }
            while (!ticket_.compare_exchange_weak(old_ticket, new_ticket));

            util::unregister_lock(this);
        }
    };

}}}

#endif
