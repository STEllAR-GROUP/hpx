//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "hpx/lcos/local/composable_guard.hpp"
#include <hpx/apply.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace lcos { namespace local {

void run_composable(guard_task *task);
void run_async(guard_task *task);

// A link in the list of tasks attached
// to a guard
struct guard_task : DebugObject {
    guard_atomic next;
    util::function_nonser<void()> run;
    const bool single_guard;

    guard_task()
      : next((guard_task *)0), run((void(*)())0), single_guard(true) {}
    guard_task(bool sg)
      : next((guard_task*)0), run((void(*)())0), single_guard(sg) {}
};

void free(guard_task *task) {
    if(task == NULL)
        return;
    task->check();
    delete task;
}

bool sort_guard(boost::shared_ptr<guard> const& l1,
        boost::shared_ptr<guard> const& l2) {
    return boost::get_pointer(l1) < boost::get_pointer(l2);
}

void guard_set::sort() {
    if(!sorted) {
        std::sort(guards.begin(),guards.end(),sort_guard);
        (*guards.begin())->check();
        sorted = true;
    }
}

struct stage_data : public DebugObject {
    guard_set gs;
    util::function_nonser<void()> task;
    guard_task **stages;
    stage_data(util::function_nonser<void()> task_,
        std::vector<boost::shared_ptr<guard> >& guards);
    ~stage_data() {
        delete[] stages;
        stages = NULL;
    }
};

void run_guarded(guard& g,guard_task *task) {
    HPX_ASSERT(task != NULL);
    task->check();
    guard_task *prev = g.task.exchange(task);
    if(prev != NULL) {
        prev->check();
        guard_task *zero = NULL;
        if(!prev->next.compare_exchange_strong(zero,task)) {
            run_async(task);
            free(prev);
        }
    } else {
        run_async(task);
    }
}

struct stage_task_cleanup {
    stage_data *sd;
    std::size_t n;
    stage_task_cleanup(stage_data *sd_,std::size_t n_) : sd(sd_), n(n_) {}
    ~stage_task_cleanup() {
        guard_task *zero = NULL;
        // The tasks on the other guards had single_task marked,
        // so they haven't had their next field set yet. Setting
        // the next field is necessary if they are going to
        // continue processing.
        for(std::size_t k=0;k<n;k++) {
            guard_task *lt = sd->stages[k];
            lt->check();
            HPX_ASSERT(!lt->single_guard);
            zero = NULL;
            if(!lt->next.compare_exchange_strong(zero,lt)) {
                HPX_ASSERT(zero != lt);
                run_async(zero);
                free(lt);
            }
        }
        delete sd;
    }
};

void stage_task(stage_data *sd,std::size_t i,std::size_t n) {
    // if this is the last task in the set...
    if(i+1 == n) {
        stage_task_cleanup stc(sd,n);
        sd->task();
    } else {
        std::size_t k = i + 1;
        guard_task *stage = sd->stages[k];
        stage->run = boost::bind(stage_task,sd,k,n);
        HPX_ASSERT(!stage->single_guard);
        run_guarded(*sd->gs.get(k),stage);
    }
}


stage_data::stage_data(util::function_nonser<void()> task_,
        std::vector<boost::shared_ptr<guard> >& guards)
  : task(task_), stages(new guard_task*[guards.size()])
{
    const std::size_t n = guards.size();
    for(std::size_t i=0;i<n;i++) {
        stages[i] = new guard_task(false);
    }
}

void run_guarded(guard_set& guards,util::function_nonser<void()> task) {
    std::size_t n = guards.guards.size();
    if(n == 0) {
        task();
        return;
    } else if(n == 1) {
        run_guarded(*guards.guards[0],task);
        guards.check();
        return;
    }
    guards.sort();
    stage_data *sd = new stage_data(task,guards.guards);
    int k = 0;
    sd->stages[k]->run = boost::bind(stage_task,sd,k,n);
    sd->gs = guards;
    guard_task *stage = sd->stages[k]; //-V108
    run_guarded(*sd->gs.get(k),stage); //-V106
}

void run_guarded(guard& guard,util::function_nonser<void()> task) {
    guard_task *tptr = new guard_task();
    tptr->run = task;
    run_guarded(guard,tptr);
}

void run_async(guard_task *task) {
    HPX_ASSERT(task != NULL);
    task->check();
    hpx::apply(&run_composable,task);
}

// This class exists so that a destructor is
// used to perform cleanup. By using a destructor
// we ensure the code works even if exceptions are
// thrown.
struct run_composable_cleanup {
    guard_task *task;
    run_composable_cleanup(guard_task *task_) : task(task_) {}
    ~run_composable_cleanup() {
        guard_task *zero = 0;
        // If single_guard is false, then this is one of the
        // setup tasks for a multi-guarded task. By not setting
        // the next field, we halt processing on items queued
        // to this guard.
        HPX_ASSERT(task != NULL);
        task->check();
        if(!task->next.compare_exchange_strong(zero,task)) {
            HPX_ASSERT(task->next.load()!=NULL);
            run_async(zero);
            free(task);
        }
    }
};

void run_composable(guard_task *task) {
    HPX_ASSERT(task != NULL);
    task->check();
    if(task->single_guard) {
        run_composable_cleanup rcc(task);
        task->run();
    } else {
        task->run();
    }
}
}}}
