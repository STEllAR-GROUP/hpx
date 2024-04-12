//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#ifdef HPX_HAVE_VERIFY_LOCKS
#include <hpx/assert.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>
#include <hpx/lock_registration/detail/register_locks.hpp>
#include <hpx/modules/errors.hpp>
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
#include <hpx/debugging/backtrace.hpp>
#endif

#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        lock_data::lock_data([[maybe_unused]] std::size_t trace_depth)
          : ignore_(false)
          , user_data_(nullptr)
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
          , backtrace_(hpx::util::trace(trace_depth))
#endif
        {
        }

        lock_data::lock_data(
            register_lock_data* data, [[maybe_unused]] std::size_t trace_depth)
          : ignore_(false)
          , user_data_(data)
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
          , backtrace_(hpx::util::trace(trace_depth))
#endif
        {
        }

        lock_data::~lock_data()
        {
            delete user_data_;
        }

        struct held_locks_data_ptr
        {
            held_locks_data_ptr()
              : data_(std::make_unique<held_locks_data>())
            {
            }

            void reinitialize()
            {
                data_ = std::make_unique<held_locks_data>();
            }

            // note: this invalidates the stored pointer - this is intentional
            std::unique_ptr<held_locks_data> release() && noexcept
            {
                HPX_ASSERT(!!data_);

                std::unique_ptr<held_locks_data> result;

                using std::swap;
                swap(result, data_);

                return result;
            }

            void set(std::unique_ptr<held_locks_data>&& data) noexcept
            {
                data_ = HPX_MOVE(data);
            }

            std::unique_ptr<held_locks_data> data_;
        };

        struct register_locks
        {
            using held_locks_map = held_locks_data::held_locks_map;

            static held_locks_data_ptr& get_held_locks()
            {
                thread_local held_locks_data_ptr held_locks;
                if (!held_locks.data_)
                {
                    held_locks.reinitialize();
                }
                return held_locks;
            }

            static bool lock_detection_enabled_;
            static std::size_t lock_detection_trace_depth_;

            static held_locks_map& get_lock_map()
            {
                return get_held_locks().data_->map_;
            }

            static bool get_lock_enabled()
            {
                return get_held_locks().data_->enabled_;
            }

            static void set_lock_enabled(bool enable)
            {
                get_held_locks().data_->enabled_ = enable;
            }

            static bool get_ignore_all_locks()
            {
                return !get_held_locks().data_->ignore_all_locks_;
            }

            static bool set_ignore_all_locks(bool enable)
            {
                bool& val = get_held_locks().data_->ignore_all_locks_;
                if (val != enable)
                {
                    val = enable;
                    return true;
                }
                return false;
            }
        };

        bool register_locks::lock_detection_enabled_ = false;
        std::size_t register_locks::lock_detection_trace_depth_ =
            HPX_HAVE_THREAD_BACKTRACE_DEPTH;
    }    // namespace detail

    // retrieve the current thread_local data about held locks
    std::unique_ptr<held_locks_data> get_held_locks_data()
    {
        return HPX_MOVE(detail::register_locks::get_held_locks()).release();
    }

    // set the current thread_local data about held locks
    void set_held_locks_data(std::unique_ptr<held_locks_data>&& data)
    {
        detail::register_locks::get_held_locks().set(HPX_MOVE(data));
    }

    ///////////////////////////////////////////////////////////////////////////
    void enable_lock_detection() noexcept
    {
        detail::register_locks::lock_detection_enabled_ = true;
    }

    void disable_lock_detection() noexcept
    {
        detail::register_locks::lock_detection_enabled_ = false;
    }

    void trace_depth_lock_detection(std::size_t value) noexcept
    {
        detail::register_locks::lock_detection_trace_depth_ = value;
    }

    namespace {

        registered_locks_error_handler_type registered_locks_error_handler;
    }

    void set_registered_locks_error_handler(
        registered_locks_error_handler_type f) noexcept
    {
        registered_locks_error_handler = HPX_MOVE(f);
    }

    namespace {

        register_locks_predicate_type register_locks_predicate;
    }

    void set_register_locks_predicate(register_locks_predicate_type f) noexcept
    {
        register_locks_predicate = HPX_MOVE(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool register_lock(
        void const* lock, util::register_lock_data* data) noexcept
    {
        using detail::register_locks;

        try
        {
            if (register_locks::lock_detection_enabled_ &&
                (!register_locks_predicate || register_locks_predicate()))
            {
                register_locks::held_locks_map& held_locks =
                    register_locks::get_lock_map();

                if (held_locks.find(lock) != held_locks.end())
                    return false;    // this lock is already registered

                std::pair<register_locks::held_locks_map::iterator, bool> p;
                if (!data)
                {
                    p = held_locks.emplace(
                        lock, register_locks::lock_detection_trace_depth_);
                }
                else
                {
                    p = held_locks.emplace(std::piecewise_construct,
                        std::forward_as_tuple(lock),
                        std::forward_as_tuple(
                            data, register_locks::lock_detection_trace_depth_));
                }
                return p.second;
            }
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    // unregister the given lock from this HPX-thread
    bool unregister_lock(void const* lock) noexcept
    {
        using detail::register_locks;

        try
        {
            if (register_locks::lock_detection_enabled_ &&
                (!register_locks_predicate || register_locks_predicate()))
            {
                register_locks::held_locks_map& held_locks =
                    register_locks::get_lock_map();

                if (held_locks.find(lock) == held_locks.end())
                    return false;    // this lock is not registered

                held_locks.erase(lock);
            }
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    // verify that no locks are held by this HPX-thread
    namespace detail {

        inline bool some_locks_are_not_ignored(
            register_locks::held_locks_map const& held_locks) noexcept
        {
            auto const end = held_locks.end();
            for (auto it = held_locks.begin(); it != end; ++it)
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
            // we create a log message if there are still registered locks for
            // this OS-thread
            if (register_locks::held_locks_map const& held_locks =
                    register_locks::get_lock_map();
                !held_locks.empty())
            {
                // Temporarily disable verifying locks in case verify_no_locks
                // gets called recursively.
                auto old_value = register_locks::get_lock_enabled();

                register_locks::set_lock_enabled(false);
                auto on_exit = hpx::experimental::scope_exit([old_value] {
                    register_locks::set_lock_enabled(old_value);
                });

                if (detail::some_locks_are_not_ignored(held_locks))
                {
                    if (registered_locks_error_handler)
                    {
                        registered_locks_error_handler();
                    }
                    else
                    {
                        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                            "verify_no_locks",
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
        //        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
        //            "force_error_on_lock",
        //            "At least one lock is held while thread is being "
        //            terminated or interrupted.");
        //    }
        //}
    }

    namespace detail {

        bool set_ignore_status(void const* lock, bool status)
        {
            if (register_locks::lock_detection_enabled_ &&
                (!register_locks_predicate || register_locks_predicate()))
            {
                register_locks::held_locks_map& held_locks =
                    register_locks::get_lock_map();

                auto const it = held_locks.find(lock);
                if (it == held_locks.end())
                {
                    // this can happen if the lock was registered to be ignored
                    // on a different OS thread
                    return false;
                }

                if (it->second.ignore_ != status)
                {
                    it->second.ignore_ = status;
                    return true;
                }
            }
            return false;
        }
    }    // namespace detail

    bool ignore_lock(void const* lock) noexcept
    {
        try
        {
            return detail::set_ignore_status(lock, true);
        }
        catch (...)
        {
            return false;
        }
    }

    bool reset_ignored(void const* lock) noexcept
    {
        try
        {
            return detail::set_ignore_status(lock, false);
        }
        catch (...)
        {
            return false;
        }
    }

    bool ignore_all_locks() noexcept
    {
        try
        {
            return detail::register_locks::set_ignore_all_locks(true);
        }
        catch (...)
        {
            return false;
        }
    }

    bool reset_ignored_all() noexcept
    {
        try
        {
            return detail::register_locks::set_ignore_all_locks(false);
        }
        catch (...)
        {
            return false;
        }
    }
}    // namespace hpx::util

#endif
