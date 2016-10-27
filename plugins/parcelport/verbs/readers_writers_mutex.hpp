///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2014-2016 MongoDB, Inc.
//  Copyright (c) 2008-2014 WiredTiger, Inc.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_READERS_WRITERS_MUTEX_HPP
#define HPX_READERS_WRITERS_MUTEX_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/yield_k.hpp>
//
#include <boost/thread/locks.hpp>

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
// from http://locklessinc.com/articles/locks/

#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))
#define cmpxchg(P, O, N)  __sync_bool_compare_and_swap((P), (O), (N))
#define atomic_inc(P)     __sync_add_and_fetch((P), 1)

// Compiler instruction reordering rwl_barrier_
#define rwl_barrier_() asm volatile("": : :"memory")

#pragma GCC push_options
#pragma GCC optimize ("O0")

namespace hpx {
namespace lcos {
namespace local {

    class readers_writer_mutex
    {
    private:

        typedef union {
            uint64_t u;
            struct {
                uint32_t wr;        // Writers and readers
            } i;
            struct {
                uint16_t writers;   // Now serving for writers
                uint16_t readers;   // Now serving for readers
                uint16_t next;      // Next available ticket number
                uint16_t notused;   // Padding to 8 bytes
            } s;
        } readwrite_ticket;

        readwrite_ticket ticket;

    public:
        readers_writer_mutex() : ticket{0} {}

        //
        // acquire lock for a unique writer
        //
        void lock()
        {
            // memory ordering barrier
            rwl_barrier_();

            uint16_t val = atomic_xadd(&ticket.s.next, 1);
            while (val != ticket.s.writers) {
                // std::cout << "rw lock sleep write " << val << " " << ticket.s.next << " \n";
                hpx::util::detail::yield_k(4, nullptr);
            }

            // memory ordering rwl_barrier_
            rwl_barrier_();
        }

        //
        // unlock writer
        //
        void unlock()
        {
            // only one writer can enter unlock at a time, so we do not need atomic ops
            readwrite_ticket new_ticket = ticket;
            //
            ++new_ticket.s.writers;
            ++new_ticket.s.readers;
            ticket.i.wr = new_ticket.i.wr;
        }

        //
        // try to obtain unique writer lock
        //
        bool try_lock()
        {
            readwrite_ticket new_ticket, old_ticket;
            new_ticket = old_ticket = ticket;

            /*
             * This write lock can only be granted if the lock was last granted to
             * a writer and there are no readers or writers blocked on the lock,
             * that is, if this thread's ticket would be the next ticket granted.
             * Do the cheap test to see if this can possibly succeed (and confirm
             * the lock is in the correct state to grant this write lock).
             */
            if (old_ticket.s.writers != old_ticket.s.next)
                return false;

            // The replacement lock value is a result of allocating a new ticket.
            ++new_ticket.s.next;

            return (cmpxchg(&ticket.u, old_ticket.u, new_ticket.u) ? true : false);
        }

        //
        // obtain a reader lock, many readers may have the lock simultaneously
        //
        void lock_shared()
        {
            // memory ordering rwl_barrier_
            rwl_barrier_();

            uint16_t val = atomic_xadd(&ticket.s.next, 1);
            while (val != ticket.s.readers) {
                // std::cout << "rw lock sleep read " << val << " " << ticket.s.readers << "\n";
                hpx::util::detail::yield_k(0, nullptr);
            }

            // only one writer can lock, so no need for atomic increment
            ++ticket.s.readers;

            // memory ordering rwl_barrier_
            rwl_barrier_();
        }

        //
        // unlock one reader
        //
        void unlock_shared()
        {
            atomic_inc(&ticket.s.writers);
        }

        //
        // try to obtain a reader lock
        //
        bool try_lock_shared()
        {
            readwrite_ticket new_ticket, old_ticket;
            new_ticket = old_ticket = ticket;
            //
            /*
             * This read lock can only be granted if the lock was last granted to
             * a reader and there are no readers or writers blocked on the lock,
             * that is, if this thread's ticket would be the next ticket granted.
             * Do the cheap test to see if this can possibly succeed (and confirm
             * the lock is in the correct state to grant this read lock).
             */
            if (old_ticket.s.readers != new_ticket.s.next)
                return false;
            //
            /*
             * The replacement lock value is a result of allocating a new ticket and
             * incrementing the reader value to match it.
             */
            new_ticket.s.readers = new_ticket.s.next = old_ticket.s.next + 1;
            return (cmpxchg(&ticket.u, old_ticket.u, new_ticket.u) ? true : false);
        }

        // return true if a reader or writer has the lock
        bool owns_lock()
        {
            return ((ticket.s.writers != ticket.s.next)
                || (ticket.s.readers != ticket.s.next));
        }
    };
}
}
}

#pragma GCC pop_options

#endif
