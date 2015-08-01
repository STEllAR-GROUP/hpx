//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_UTIL_READERS_WRITER_LOCK_HPP_
#define HPX_UTIL_READERS_WRITER_LOCK_HPP_

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    // readers_writer_lock is a simple object that can be used to lock a
    // shared mutex for either multiple concurrent readers or a single writer.
    // The constructor does not lock the lock, it must be requested via
    // either read_lock, or write_lock depending upon requirements.
    //
    // example of usage is
    //   declare a mutex
    //      hpx::lcos::local::shared_mutex> mutex;
    //   take read access to a resource
    //      readers_writer_lock<hpx::lcos::local::shared_mutex> rw_lock(mutex);
    //      rw_lock.read_lock();
    //   elsewhere...
    //      readers_writer_lock<hpx::lcos::local::shared_mutex> rw_lock(mutex);
    //      rw_lock.write_lock();

    template <typename MutexType = hpx::lcos::local::shared_mutex>
    class readers_writer_lock
    {
    public:
        readers_writer_lock(MutexType &mutex) :
            _mutex(&mutex),
            _shared_lock(mutex,  boost::defer_lock),
            _upgrade_lock(mutex, boost::defer_lock),
            _unique_lock(mutex,  boost::defer_lock)
        {
        }

        // not really needed as destructors of locks would "do the right thing" anyway
        ~readers_writer_lock() {
            unlock();
        }

        // manually unlock whichever mode we are in
        void unlock() {
            if (_unique_lock.owns_lock()) {
                _upgrade_lock = boost::upgrade_lock<MutexType> (boost::move(_unique_lock));
                _upgrade_lock.unlock();
            }
            else if (_shared_lock.owns_lock()) {
                _shared_lock.unlock();
            }
        }

        // lock for multiple reader access
        void read_lock() {
            _shared_lock.lock();
        }

        // lock for single writer access
        void write_lock() {
            _upgrade_lock.lock();
            _unique_lock = boost::move(_upgrade_lock);
        }

    private:

        MutexType                               *_mutex;
        boost::shared_lock<MutexType>            _shared_lock;
        boost::upgrade_lock<MutexType>           _upgrade_lock;
        boost::unique_lock<MutexType>            _unique_lock;
    };
}}

#endif /* SPIN_GLASS_SOLVER_ASYNC_READERS_WRITER_LOCK_HPP_ */
