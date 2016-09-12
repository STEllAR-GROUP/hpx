//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/local/composable_guard.hpp>

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/function.hpp>

#include <boost/atomic.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace lcos { namespace local
{
    static void run_composable(detail::guard_task* task);
    static void run_async(detail::guard_task* task);

    namespace detail
    {
        // A link in the list of tasks attached
        // to a guard
        struct guard_task : detail::debug_object
        {
            guard_atomic next;
            detail::guard_function run;
            bool const single_guard;

            guard_task()
              : next(nullptr), run(nullptr), single_guard(true) {}
            guard_task(bool sg)
              : next(nullptr), run(nullptr), single_guard(sg) {}
        };

        void free(guard_task* task)
        {
            if (task == nullptr)
                return;
            task->check();
            delete task;
        }
    }

    void guard_set::sort()
    {
        if (!sorted) {
            std::sort(guards.begin(), guards.end());
            (*guards.begin())->check();
            sorted = true;
        }
    }

    struct stage_data : public detail::debug_object
    {
        guard_set gs;
        detail::guard_function task;
        detail::guard_task** stages;

        stage_data(detail::guard_function task_,
            std::vector<std::shared_ptr<guard> >& guards)
          : task(std::move(task_))
          , stages(new detail::guard_task*[guards.size()])
        {
            std::size_t const n = guards.size();
            for (std::size_t i=0; i<n; i++) {
                stages[i] = new detail::guard_task(false);
            }
        }

        ~stage_data() {
            delete[] stages;
            stages = nullptr;
        }
    };

    static void run_guarded(guard& g, detail::guard_task* task)
    {
        HPX_ASSERT(task != nullptr);
        task->check();
        detail::guard_task* prev = g.task.exchange(task);
        if (prev != nullptr) {
            prev->check();
            detail::guard_task* zero = nullptr;
            if (!prev->next.compare_exchange_strong(zero, task)) {
                run_async(task);
                free(prev);
            }
        } else {
            run_async(task);
        }
    }

    struct stage_task_cleanup
    {
        stage_data* sd;
        std::size_t n;
        stage_task_cleanup(stage_data* sd_, std::size_t n_) : sd(sd_), n(n_) {}
        ~stage_task_cleanup() {
            detail::guard_task* zero = nullptr;
            // The tasks on the other guards had single_task marked,
            // so they haven't had their next field set yet. Setting
            // the next field is necessary if they are going to
            // continue processing.
            for (std::size_t k=0; k<n; k++) {
                detail::guard_task* lt = sd->stages[k];
                lt->check();
                HPX_ASSERT(!lt->single_guard);
                zero = nullptr;
                if (!lt->next.compare_exchange_strong(zero, lt)) {
                    HPX_ASSERT(zero != lt);
                    run_async(zero);
                    free(lt);
                }
            }
            delete sd;
        }
    };

    static void stage_task(stage_data* sd, std::size_t i, std::size_t n)
    {
        // if this is the last task in the set...
        if (i+1 == n) {
            stage_task_cleanup stc(sd, n);
            sd->task();
        } else {
            std::size_t k = i + 1;
            detail::guard_task* stage = sd->stages[k];
            stage->run = util::bind(stage_task, sd, k, n);
            HPX_ASSERT(!stage->single_guard);
            run_guarded(*sd->gs.get(k), stage);
        }
    }

    void run_guarded(guard_set& guards, detail::guard_function task)
    {
        std::size_t n = guards.guards.size();
        if (n == 0) {
            task();
            return;
        } else if (n == 1) {
            run_guarded(*guards.guards[0], std::move(task));
            guards.check();
            return;
        }
        guards.sort();
        stage_data* sd = new stage_data(std::move(task), guards.guards);
        int k = 0;
        sd->stages[k]->run = util::bind(stage_task, sd, k, n);
        sd->gs = guards;
        detail::guard_task* stage = sd->stages[k]; //-V108
        run_guarded(*sd->gs.get(k), stage); //-V106
    }

    void run_guarded(guard& guard, detail::guard_function task)
    {
        detail::guard_task* tptr = new detail::guard_task();
        tptr->run = std::move(task);
        run_guarded(guard, tptr);
    }

    static void run_async(detail::guard_task* task)
    {
        HPX_ASSERT(task != nullptr);
        task->check();
        hpx::apply(&run_composable, task);
    }

    // This class exists so that a destructor is
    // used to perform cleanup. By using a destructor
    // we ensure the code works even if exceptions are
    // thrown.
    struct run_composable_cleanup
    {
        detail::guard_task* task;
        run_composable_cleanup(detail::guard_task* task_) : task(task_) {}
        ~run_composable_cleanup() {
            detail::guard_task* zero = nullptr;
            // If single_guard is false, then this is one of the
            // setup tasks for a multi-guarded task. By not setting
            // the next field, we halt processing on items queued
            // to this guard.
            HPX_ASSERT(task != nullptr);
            task->check();
            if (!task->next.compare_exchange_strong(zero, task)) {
                HPX_ASSERT(task->next.load()!=nullptr);
                run_async(zero);
                free(task);
            }
        }
    };

    static void run_composable(detail::guard_task* task)
    {
        HPX_ASSERT(task != nullptr);
        task->check();
        if (task->single_guard) {
            run_composable_cleanup rcc(task);
            task->run();
        } else {
            task->run();
        }
    }
}}}
