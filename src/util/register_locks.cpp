//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/thread_specific_ptr.hpp>

#include <boost/ptr_container/ptr_map.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    std::string backtrace();
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
#if HPX_VERIFY_LOCKS
    struct register_locks
    {
        typedef boost::ptr_map<void const*, util::register_lock_data>
            held_locks_map;

        struct tls_tag {};
        static hpx::util::thread_specific_ptr<held_locks_map, tls_tag> held_locks_;

        static held_locks_map& get_lock_map()
        {
            if (NULL == held_locks_.get())
            {
                held_locks_.reset(new held_locks_map());
            }

            BOOST_ASSERT(NULL != held_locks_.get());
            return *held_locks_.get();
        }
    };

    hpx::util::thread_specific_ptr<register_locks::held_locks_map, register_locks::tls_tag>
        register_locks::held_locks_;

    ///////////////////////////////////////////////////////////////////////////
    bool register_lock(void const* lock, util::register_lock_data* data)
    {
        if (LHPX_ENABLED(debug) && 0 != threads::get_self_ptr())
        {
            register_locks::held_locks_map& held_locks =
                register_locks::get_lock_map();

            register_locks::held_locks_map::iterator it = held_locks.find(lock);
            if (it != held_locks.end())
                return false;     // this lock is already registered

            std::pair<register_locks::held_locks_map::iterator, bool> p;
            if (!data) {
                p = held_locks.insert(lock, new util::register_lock_data);
            }
            else {
                p = held_locks.insert(lock, data);
            }
            return p.second;
        }
        return true;
    }

    // unregister the given lock from this HPX-thread
    bool unregister_lock(void const* lock)
    {
        if (LHPX_ENABLED(debug) && 0 != threads::get_self_ptr())
        {
            register_locks::held_locks_map& held_locks =
                register_locks::get_lock_map();

            register_locks::held_locks_map::iterator it = held_locks.find(lock);
            if (it == held_locks.end())
                return false;     // this lock is not registered

            held_locks.erase(lock);
        }
        return true;
    }

    // verify that no locks are held by this HPX-thread
    void verify_no_locks()
    {
        if (LHPX_ENABLED(debug) && 0 != threads::get_self_ptr())
        {
            register_locks::held_locks_map const& held_locks =
                register_locks::get_lock_map();

            // we create a log message if there are still registered locks for
            // this OS-thread
            if (!held_locks.empty()) {
                BOOST_ASSERT(false);
                std::string back_trace(hpx::detail::backtrace());
                if (back_trace.empty()) {
                    LERR_(debug)
                        << "suspending thread while at least one lock is being held "
                           "(stack backtrace was disabled at compile time)";
                }
                else {
                    LERR_(debug)
                        << "suspending thread while at least one lock is being held, "
                        << "stack backtrace: " << back_trace;
                }
            }
        }
    }

    void force_error_on_lock()
    {
        // For now just do the same as during suspension. We can't reliably
        // tell whether there are still locks held as those could have been
        // acquired in a different OS thread.
        verify_no_locks();

//         if (0 != threads::get_self_ptr())
//         {
//             register_locks::held_locks_map const& held_locks =
//                 register_locks::get_lock_map();
//
//             // we throw an error if there are still registered locks for
//             // this OS-thread
//             if (!held_locks.empty()) {
//                 HPX_THROW_EXCEPTION(invalid_status, "force_error_on_lock",
//                     "At least one lock is held while thread is being terminated "
//                     "or interrupted.");
//             }
//         }
    }
#else
    bool register_lock(void const*, util::register_lock_data*)
    {
        return true;
    }

    bool unregister_lock(void const*)
    {
        return true;
    }

    void verify_no_locks()
    {
    }

    void force_error_on_lock()
    {
    }
#endif
}}


