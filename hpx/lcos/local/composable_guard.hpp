//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP
#define HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP

#include <hpx/config.hpp>
#include <hpx/util_fwd.hpp>
#include <hpx/util/assert.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace hpx { namespace lcos { namespace local {
    struct guard_task;
}}}

typedef boost::atomic<hpx::lcos::local::guard_task *> guard_atomic;

const int DEBUG_MAGIC = 0x2cab;
struct DebugObject {
#ifdef HPX_DEBUG
    int magic;
#endif
    DebugObject()
#ifdef HPX_DEBUG
    : magic(DEBUG_MAGIC)
#endif
    {}
    ~DebugObject() {
        check();
#ifdef HPX_DEBUG
        magic = ~DEBUG_MAGIC;
#endif
    }
    void check() {
#ifdef HPX_DEBUG
        HPX_ASSERT(magic != ~DEBUG_MAGIC);
        HPX_ASSERT(magic == DEBUG_MAGIC);
#endif
    }
};

namespace hpx { namespace lcos { namespace local {
struct guard_task;
HPX_API_EXPORT void free(guard_task *task);

struct guard : DebugObject {
    guard_atomic task;

    guard() : task((guard_task *)0) {}
    ~guard() {
        free(task.load());
    }
};

class guard_set : DebugObject {
    std::vector<std::shared_ptr<guard> > guards;
    // the guards need to be sorted, but we don't
    // want to sort them more often than necessary
    bool sorted;
    void sort();
public:
    guard_set() : guards(), sorted(true) {}
    ~guard_set() {}

    void add(std::shared_ptr<guard> const& guard_ptr) {
        guards.push_back(guard_ptr);
        sorted = false;
    }

    friend HPX_API_EXPORT void run_guarded(guard_set& guards,
        util::function_nonser<void()> task);
    std::shared_ptr<guard> get(std::size_t i) { return guards[i]; }
};

/// Conceptually, a guard acts like a mutex on an asyncrhonous task. The
/// mutex is locked before the task runs, and unlocked afterwards.
HPX_API_EXPORT void run_guarded(guard& guard, util::function_nonser<void()> task);

/// Conceptually, a guard_set acts like a set of mutexes on an asyncrhonous task.
/// The mutexes are locked before the task runs, and unlocked afterwards.
HPX_API_EXPORT void run_guarded(guard_set& guards,
        util::function_nonser<void()> task);
}}}

#endif /*HPX_LCOS_LOCAL_COMPOSABLE_GUARD_HPP*/
