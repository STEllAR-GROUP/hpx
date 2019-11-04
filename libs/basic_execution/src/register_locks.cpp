//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/errors.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

#ifdef HPX_HAVE_VERIFY_LOCKS
    namespace detail {

        struct lock_data
        {
            lock_data()
              : ignore_(false)
              , user_data_(nullptr)
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
              , backtrace_(hpx::util::trace())
#endif
            {
            }

            lock_data(register_lock_data* data)
              : ignore_(false)
              , user_data_(data)
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
              , backtrace_(hpx::detail::trace())
#endif
            {
            }

            ~lock_data()
            {
                delete user_data_;
            }

            bool ignore_;
            register_lock_data* user_data_;
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
            std::string backtrace_;
#endif
        };

        struct register_locks
        {
            using held_locks_map = std::map<void const*, lock_data>;

            struct held_locks_data
            {
                held_locks_data()
                  : enabled_(true)
                  , ignore_all_locks_(false)
                {
                }

                held_locks_map data_;
                bool enabled_;
                bool ignore_all_locks_;
            };

            static HPX_NATIVE_TLS held_locks_data held_locks_;

            static bool lock_detection_enabled_;

            static held_locks_map& get_lock_map()
            {
                return held_locks_.data_;
            }

            static bool get_lock_enabled()
            {
                return held_locks_.enabled_;
            }

            static void set_lock_enabled(bool enable)
            {
                held_locks_.enabled_ = enable;
            }

            static bool get_ignore_all_locks()
            {
                return !held_locks_.ignore_all_locks_;
            }

            static void set_ignore_all_locks(bool enable)
            {
                held_locks_.ignore_all_locks_ = enable;
            }
        };

        HPX_NATIVE_TLS register_locks::held_locks_data
            register_locks::held_locks_;
        bool register_locks::lock_detection_enabled_ = false;

        struct reset_lock_enabled_on_exit
        {
            reset_lock_enabled_on_exit()
              : old_value_(register_locks::get_lock_enabled())
            {
                register_locks::set_lock_enabled(false);
            }
            ~reset_lock_enabled_on_exit()
            {
                register_locks::set_lock_enabled(old_value_);
            }

            bool old_value_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void enable_lock_detection()
    {
        detail::register_locks::lock_detection_enabled_ = true;
    }

    static registered_locks_error_handler_type registered_locks_error_handler;

    void set_registered_locks_error_handler(
        registered_locks_error_handler_type f)
    {
        registered_locks_error_handler = f;
    }

    static register_locks_predicate_type register_locks_predicate;

    void set_register_locks_predicate(register_locks_predicate_type f)
    {
        register_locks_predicate = f;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool register_lock(void const* lock, util::register_lock_data* data)
    {
        using detail::register_locks;

        if (register_locks::lock_detection_enabled_ &&
            (!register_locks_predicate || register_locks_predicate()))
        {
            register_locks::held_locks_map& held_locks =
                register_locks::get_lock_map();

            register_locks::held_locks_map::iterator it = held_locks.find(lock);
            if (it != held_locks.end())
                return false;    // this lock is already registered

            std::pair<register_locks::held_locks_map::iterator, bool> p;
            if (!data)
            {
                p = held_locks.insert(
                    std::make_pair(lock, detail::lock_data()));
            }
            else
            {
                p = held_locks.insert(
                    std::make_pair(lock, detail::lock_data(data)));
            }
            return p.second;
        }
        return true;
    }

    // unregister the given lock from this HPX-thread
    bool unregister_lock(void const* lock)
    {
        using detail::register_locks;

        if (register_locks::lock_detection_enabled_ &&
            (!register_locks_predicate || register_locks_predicate()))
        {
            register_locks::held_locks_map& held_locks =
                register_locks::get_lock_map();

            register_locks::held_locks_map::iterator it = held_locks.find(lock);
            if (it == held_locks.end())
                return false;    // this lock is not registered

            held_locks.erase(lock);
        }
        return true;
    }

    // verify that no locks are held by this HPX-thread
    namespace detail {

        inline bool some_locks_are_not_ignored(
            register_locks::held_locks_map const& held_locks)
        {
            using iterator = register_locks::held_locks_map::const_iterator;

            iterator end = held_locks.end();
            for (iterator it = held_locks.begin(); it != end; ++it)
            {
                //lock_data const& data = *(*it).second;
                if (!it->second.ignore_)
                    return true;
            }

            return false;
        }
    }    // namespace detail

    void verify_no_locks()
    {
        using detail::register_locks;

        bool enabled = register_locks::get_ignore_all_locks() &&
            register_locks::get_lock_enabled();

        if (enabled && register_locks::lock_detection_enabled_ &&
            (!register_locks_predicate || register_locks_predicate()))
        {
            register_locks::held_locks_map& held_locks =
                register_locks::get_lock_map();

            // we create a log message if there are still registered locks for
            // this OS-thread
            if (!held_locks.empty())
            {
                // Temporarily disable verifying locks in case verify_no_locks
                // gets called recursively.
                detail::reset_lock_enabled_on_exit e;

                if (detail::some_locks_are_not_ignored(held_locks))
                {
                    if (registered_locks_error_handler)
                    {
                        registered_locks_error_handler();
                    }
                    else
                    {
                        HPX_THROW_EXCEPTION(invalid_status, "verify_no_locks",
                            "suspending thread while at least one lock is "
                            "being held (default handler)");
                    }
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

        //{
        //    register_locks::held_locks_map const& held_locks =
        //       register_locks::get_lock_map();
        //
        //    // we throw an error if there are still registered locks for
        //    // this OS-thread
        //    if (!held_locks.empty()) {
        //        HPX_THROW_EXCEPTION(invalid_status, "force_error_on_lock",
        //            "At least one lock is held while thread is being "
        //            terminated or interrupted.");
        //    }
        //}
    }

    namespace detail {

        void set_ignore_status(void const* lock, bool status)
        {
            if (register_locks::lock_detection_enabled_ &&
                (!register_locks_predicate || register_locks_predicate()))
            {
                register_locks::held_locks_map& held_locks =
                    register_locks::get_lock_map();

                register_locks::held_locks_map::iterator it =
                    held_locks.find(lock);
                if (it == held_locks.end())
                {
                    // this can happen if the lock was registered to be ignore
                    // on a different OS thread
                    // HPX_THROW_EXCEPTION(
                    //     invalid_status, "set_ignore_status",
                    //     "The given lock has not been registered.");
                    return;
                }

                it->second.ignore_ = status;
            }
        }
    }    // namespace detail

    void ignore_lock(void const* lock)
    {
        detail::set_ignore_status(lock, true);
    }

    void reset_ignored(void const* lock)
    {
        detail::set_ignore_status(lock, false);
    }

    void ignore_all_locks()
    {
        detail::register_locks::set_ignore_all_locks(true);
    }

    void reset_ignored_all()
    {
        detail::register_locks::set_ignore_all_locks(false);
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

    void verify_no_locks() {}

    void force_error_on_lock() {}

    void ignore_lock(void const* lock) {}

    void reset_ignored(void const* lock) {}

    void ignore_all_locks() {}

    void reset_ignored_all() {}
#endif
}}    // namespace hpx::util
