..
    Copyright (C) 2019 Tapasweni Pathak

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_task_based_programming:

================================================================
Designing and Writie task Based Programs: Task Based Programming
================================================================

Task based programming is a different approach to writing your code. In task
based programming

- your program should be a tree of tasks
- ach task will depend on other tasks
- and have child tasks
- you don't have to have the same on all nodes (c.f. MPI)

.. figure:: /_static/images/DAG-weather.jpg

- each one of these boxes will contain a sub-tree of (possibly millions) of
  tasks

Task Goals
=====

- Make these gaps as small as possible
  - Keep breaking tasks into smaller tasks so the scheduler can fill gaps
  - limit of task size/granularity is a function of overheads
    - context switch (actually swapping stack, registers etc)
    - time to create, queue, dequeue a task


.. figure:: /_static/images/Task-waits.jpg

Bad tasks (or scheduling)
=========================

.. figure:: /_static/images/tasks-bad.jpg

Good tasks
==========

.. figure:: /_static/images/tasks-good.jpg

Task decomposition
==================

- Breaking a program into tasks should be straightforward
- More functional
  - tasks should accept inputs and return results
  - modifying global state should be avoided
    - race conditions and other thread related problems (deadlocks etc)
  - the leaf nodes of the tree are the smallest bits of work you can express
    - but those leaf nodes might be broken further by HPX
    - even ``parallel::for(...)`` loops decompose into tasks
    - all parallel::algorithm's are made up of tasks
  - HPX differs from (most) other libraries because the same API and the same scheduling/runtime can be used for the whole heirarchy of tasks
  - We aim to replace OpenMP+MPI+Acc with a single framework
    - based soundly on C++
    - from top to bottom (of the task tree)

Task scheduling and lifetime
============================

- Each task goes onto one of the schedulers
  - where a task goes is controlled by executors
  - schedulers maintain a high priority and normal queue
  - schedulers can steal (some do, some don't)
  - you can choose a scheduler when you create an executor (not really though)
- Tasks can be
  - Running : context is active, code is running as it would on native thread
  - Suspended : task ran, but then had to wait for something
  - Staged : task has been created, but cannot be run yet
  - Pending :it is ready to run, but waiting in a queue
  - Terminated : awaiting cleanup

Suspended tasks
===============

* A task that is running requires a value from a future

    * the future is not ready (:disappointed:)
        * Use CPS - don't call get ever!

    * `auto val = future->get()` would block (if we were not HPX)

    * the current task cannot progress so it changes state to _suspended_

    * the scheduler puts it into the suspended list

    * the future that was needed has the suspended task added to its internal
    (shared state) waiting list

    * when that future become ready, the task will be changed (back) to _pending_

    * and go back onto the queue so that when a worker is free, it can run


* The same process happens when a task tries to take a lock but can't get it

    * The shared state inside the mutex will `notify` the task and do the 'right thing'

    * the current task cannot progress so it changes state to _suspended_
        * Look for spinlock mutexes in the HPX source
        * Spin for a bit, then suspend when some deadline reached

    * the scheduler puts it into the suspended list


* This is **one** reason why all the `std::thread`, `std::mutex` etc code has been reimplemented

    * You can use std::lock_guard<> etc, but not the mutexes inside them
    * the locks are just wrappers around the mutexes where the real work is done


Staged tasks
============

* Like a suspended task, but it hasn't run yet

* A staged task is what exists when a continuation or `when_xxx` creates a task
but it can't run until the dependencies are satisfied

* It's a suspended task that hasn't yet started

* When is the task actually created?
    * When the code that instantiates it is executed

    * Inside continuations this might be when another task completes
        * `future.then(another_task.then(more_tasks));`

    * Outside continuations it can be 'up-front'
        * a loop generating futures and attaching to previous iterations
            `future[i] = future[i-1].then(another_task);`
        * it can be confusing

    * Session (Resource management) will look again at this question
