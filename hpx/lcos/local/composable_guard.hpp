//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP
#define HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util_fwd.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        struct debug_object
        {
#ifdef HPX_DEBUG
            HPX_STATIC_CONSTEXPR int debug_magic = 0x2cab;

            int magic;

            debug_object()
              : magic(debug_magic)
            {}

            ~debug_object() {
                check();
                magic = ~debug_magic;
            }

            void check() {
                HPX_ASSERT(magic != ~debug_magic);
                HPX_ASSERT(magic == debug_magic);
            }
#else
            void check() {}
#endif
        };

        struct guard_task;

        typedef boost::atomic<guard_task*> guard_atomic;

        HPX_API_EXPORT void free(guard_task* task);

        typedef util::unique_function_nonser<void()> guard_function;
    }

    class guard : public detail::debug_object
    {
    public:
        detail::guard_atomic task;

        guard() : task(nullptr) {}
        ~guard() {
            detail::free(task.load());
        }
    };

    class guard_set : public detail::debug_object
    {
        std::vector<std::shared_ptr<guard> > guards;
        // the guards need to be sorted, but we don't
        // want to sort them more often than necessary
        bool sorted;

        void sort();

    public:
        guard_set() : guards(), sorted(true) {}
        ~guard_set() {}

        std::shared_ptr<guard> get(std::size_t i) { return guards[i]; }

        void add(std::shared_ptr<guard> const& guard_ptr) {
            guards.push_back(guard_ptr);
            sorted = false;
        }

        friend HPX_API_EXPORT void run_guarded(
            guard_set& guards, detail::guard_function task);
    };

    /// Conceptually, a guard acts like a mutex on an asynchronous task. The
    /// mutex is locked before the task runs, and unlocked afterwards.
    HPX_API_EXPORT void run_guarded(guard& guard, detail::guard_function task);

    template <typename F, typename ...Args>
    void run_guarded(guard& guard, F&& f, Args&&... args)
    {
        return run_guarded(guard, detail::guard_function(
            util::deferred_call(std::forward<F>(f), std::forward<Args>(args)...)));
    }

    /// Conceptually, a guard_set acts like a set of mutexes on an asynchronous task.
    /// The mutexes are locked before the task runs, and unlocked afterwards.
    HPX_API_EXPORT void run_guarded(guard_set& guards, detail::guard_function task);

    template <typename F, typename ...Args>
    void run_guarded(guard_set& guards, F&& f, Args&&... args)
    {
        return run_guarded(guards, detail::guard_function(
            util::deferred_call(std::forward<F>(f), std::forward<Args>(args)...)));
    }
}}}

#endif /*HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP*/
