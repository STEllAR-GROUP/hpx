//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_block.hpp
/// \page hpx::experimental::task_canceled_exception, hpx::experimental::task_block, hpx::experimental::define_task_block, hpx::experimental::define_task_block_restore_thread
/// \headerfile hpx/task_block.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/parallel/task_group.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <exception>
#include <memory>    // std::addressof
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::experimental {

    namespace detail {

        struct define_task_block_impl;
    }    // namespace detail

    /// The class \a task_canceled_exception defines the type of objects thrown
    /// by task_block::run or task_block::wait if they detect that an exception
    /// is pending within the current parallel region.
    class HPX_ALWAYS_EXPORT task_canceled_exception : public hpx::exception
    {
    public:
        task_canceled_exception() noexcept
          : hpx::exception(hpx::error::task_canceled_exception)
        {
        }
    };

    /// The class task_block defines an interface for forking and joining
    /// parallel tasks. The \a define_task_block and
    /// \a define_task_block_restore_thread
    /// function templates create an object of type task_block and pass a
    /// reference to that object to a user-provided callable object.
    ///
    /// An object of class \a task_block cannot be constructed, destroyed,
    /// copied, or moved except by the implementation of the task region
    /// library. Taking the address of a task_block object via operator& or
    /// addressof is ill formed. The result of obtaining its address by any
    /// other means is unspecified.
    ///
    /// A \a task_block is active if it was created by the nearest enclosing
    /// task block, where "task block" refers to an invocation of
    /// define_task_block or define_task_block_restore_thread and "nearest
    /// enclosing" means the most recent invocation that has not yet completed.
    /// Code designated for execution in another thread by means other than the
    /// facilities in this section (e.g., using thread or async) are not
    /// enclosed in the task region and a task_block passed to (or captured by)
    /// such code is not active within that code. Performing any operation on a
    /// task_block that is not active results in undefined behavior.
    ///
    /// The \a task_block that is active before a specific call to the run
    /// member function is not active within the asynchronous function that
    /// invoked run. (The invoked function should not, therefore, capture the \a
    /// task_block from the surrounding block.)
    ///
    /// \code Example:
    ///     define_task_block([&](auto& tr) {
    ///         tr.run([&] {
    ///             tr.run([] { f(); });              // Error: tr is not active
    ///             define_task_block([&](auto& tr) { // Nested task block
    ///                 tr.run(f);                    // OK: inner tr is active
    ///                 /// ...
    ///             });
    ///         }); /// ...
    ///     });
    /// \endcode
    ///
    /// \tparam ExPolicy The execution policy an instance of a \a task_block was
    ///         created with. This defaults to \a parallel_policy.
    ///
    template <typename ExPolicy = hpx::execution::parallel_policy>
    class task_block
    {
    private:
        /// \cond NOINTERNAL
        using mutex_type = hpx::spinlock;

        friend struct detail::define_task_block_impl;

        explicit task_block(ExPolicy const& policy = ExPolicy())
          : id_(threads::get_self_id())
          , policy_(policy)
        {
        }

        void wait_for_completion()
        {
            tasks_.wait();
        }

        ~task_block()
        {
            try
            {
                wait_for_completion();
            }
            catch (...)
            {
            }
        }

        task_block(task_block const&) = delete;
        task_block& operator=(task_block const&) = delete;

        task_block* operator&() const = delete;

        void add_exception(std::exception_ptr&& p)
        {
            tasks_.add_exception(HPX_MOVE(p));
        }
        /// \endcond

    public:
        /// Refers to the type of the execution policy used to create the
        /// \a task_block.
        using execution_policy = ExPolicy;

        /// Return the execution policy instance used to create this
        /// \a task_block
        constexpr execution_policy const& get_execution_policy() const noexcept
        {
            return policy_;
        }

        /// Causes the expression f() to be invoked asynchronously.
        /// The invocation of f is permitted to run on an unspecified thread
        /// in an unordered fashion relative to the sequence of operations
        /// following the call to run(f) (the continuation), or indeterminately
        /// sequenced within the same thread as the continuation.
        ///
        /// The call to \a run synchronizes with the invocation of f. The
        /// completion of f() synchronizes with the next invocation of wait on
        /// the same task_block or completion of the nearest enclosing
        /// task block (i.e., the \a define_task_block or
        /// \a define_task_block_restore_thread that created this task block).
        ///
        /// Requires: F shall be MoveConstructible. The expression, (void)f(),
        ///           shall be well-formed.
        ///
        /// Precondition: this shall be the active task_block.
        ///
        /// Postconditions: A call to run may return on a different thread than
        ///                 that on which it was called.
        ///
        /// \note The call to \a run is sequenced before the continuation as if
        ///       \a run returns on the same thread.
        ///       The invocation of the user-supplied callable object f may be
        ///       immediate or may be delayed until compute resources are
        ///       available. \a run might or might not return before invocation
        ///       of f completes.
        ///
        /// \throw task_canceled_exception, as described in Exception Handling.
        ///
        template <typename F, typename... Ts>
        void run(F&& f, Ts&&... ts)
        {
            // The proposal requires that the task_block should be
            // 'active' to be usable.
            if (id_ != threads::get_self_id())
            {
                HPX_THROW_EXCEPTION(hpx::error::task_block_not_active,
                    "task_block::run", "the task_block is not active");
            }

            tasks_.run(
                policy_.executor(), HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        /// Causes the expression f() to be invoked asynchronously using the
        /// given executor. The invocation of f is permitted to run on an
        /// unspecified thread associated with the given executor and in an
        /// unordered fashion relative to the sequence of operations following
        /// the call to run(exec, f) (the continuation), or indeterminately
        /// sequenced within the same thread as the continuation.
        ///
        /// The call to \a run synchronizes with the invocation of f. The
        /// completion of f() synchronizes with the next invocation of wait on
        /// the same task_block or completion of the nearest enclosing task
        /// block (i.e., the \a define_task_block or
        /// \a define_task_block_restore_thread that created this task block).
        ///
        /// Requires: Executor shall be a type modeling the Executor concept.
        ///           F shall be MoveConstructible. The expression, (void)f(),
        ///           shall be well-formed.
        ///
        /// Precondition: this shall be the active task_block.
        ///
        /// Postconditions: A call to run may return on a different thread than
        ///                 that on which it was called.
        ///
        /// \note The call to \a run is sequenced before the continuation as if
        ///       \a run returns on the same thread.
        ///       The invocation of the user-supplied callable object f may be
        ///       immediate or may be delayed until compute resources are
        ///       available. \a run might or might not return before invocation
        ///       of f completes.
        ///
        /// \throw task_canceled_exception, as described in Exception Handling.
        ///        The function will also throw an \a exception_list holding all
        ///        exceptions that were caught while executing the tasks.
        ///
        template <typename Executor, typename F, typename... Ts>
        void run(Executor&& exec, F&& f, Ts&&... ts)
        {
            // The proposal requires that the task_block should be
            // 'active' to be usable.
            if (id_ != threads::get_self_id())
            {
                HPX_THROW_EXCEPTION(hpx::error::task_block_not_active,
                    "task_block::run", "the task_block is not active");
            }

            tasks_.run(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        /// Blocks until the tasks spawned using this task_block have finished.
        ///
        /// Precondition: this shall be the active task_block.
        ///
        /// Postcondition: All tasks spawned by the nearest enclosing task
        ///                region have finished. A call to wait may return on a
        ///                different thread than that on which it was called.
        ///
        /// \note The call to \a wait is sequenced before the continuation as if
        ///       \a wait returns on the same thread.
        ///
        /// \throw This function may throw \a task_canceled_exception, as
        ///        described in Exception Handling. The function will also throw
        ///        a \a exception_list holding all exceptions that were caught
        ///        while executing the tasks.
        ///
        /// \code Example:
        ///     define_task_block([&](auto& tr) {
        ///         tr.run([&]{ process(a, w, x); }); // Process a[w] through a[x]
        ///         if (y < x) tr.wait();   // Wait if overlap between [w, x) and [y, z)
        ///         process(a, y, z);       // Process a[y] through a[z]
        ///     });
        /// \endcode
        ///
        void wait()
        {
            // The proposal requires that the task_block should be
            // 'active' to be usable.
            if (id_ != threads::get_self_id())
            {
                HPX_THROW_EXCEPTION(hpx::error::task_block_not_active,
                    "task_block::run", "the task_block is not active");
                return;
            }

            tasks_.wait();
        }

        /// Returns a reference to the execution policy used to construct this
        /// object.
        ///
        /// Precondition: this shall be the active task_block.
        ///
        ExPolicy& policy() noexcept
        {
            return policy_;
        }

        /// Returns a reference to the execution policy used to construct this
        /// object.
        ///
        /// Precondition: this shall be the active task_block.
        ///
        constexpr ExPolicy const& policy() const noexcept
        {
            return policy_;
        }

    private:
        hpx::experimental::task_group tasks_;
        threads::thread_id_type id_;
        ExPolicy policy_;
    };

    namespace detail {

        /// \cond NOINTERNAL
        struct define_task_block_impl
        {
            template <typename ExPolicy, typename F>
            void operator()(ExPolicy&& policy, F&& f) const
            {
                static_assert(hpx::is_execution_policy_v<ExPolicy>,
                    "hpx::is_execution_policy_v<ExPolicy>");

                using policy_type = std::decay_t<ExPolicy>;
                task_block<policy_type> trh(HPX_FORWARD(ExPolicy, policy));

                // invoke the user supplied function
                hpx::detail::try_catch_exception_ptr([&]() { f(trh); },
                    [&](std::exception_ptr e) {
                        trh.add_exception(HPX_MOVE(e));
                    });

                // regardless of whether f(trh) has thrown an exception we need
                // to obey the contract and wait for all tasks to join
                trh.wait_for_completion();
            }
        };

        inline constexpr define_task_block_impl define_task_block{};
        /// \endcond
    }    // namespace detail

    /// Constructs a \a task_block, \a tr, using the given execution policy
    /// \a policy,and invokes the expression
    /// \a f(tr) on the user-provided object, \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the task block may be parallelized.
    /// \tparam F   The type of the user defined function to invoke inside the
    ///             define_task_block (deduced). \a F shall be MoveConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param f    The user defined function to invoke inside the task block.
    ///             Given an lvalue \a tr of type \a task_block, the
    ///             expression, (void)f(tr), shall be well-formed.
    ///
    /// Postcondition: All tasks spawned from \a f have finished execution.
    ///                A call to define_task_block may return on a different
    ///                thread than that on which it was called.
    ///
    /// \throws exception_list, as specified in Exception Handling.
    ///
    /// \note It is expected (but not mandated) that f will (directly or
    ///       indirectly) call tr.run(_callable_object_).
    ///
    // clang-format off
    template <typename ExPolicy, typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<std::decay_t<ExPolicy>>
        )>
    // clang-format on
    decltype(auto) define_task_block(ExPolicy&& policy, F&& f)
    {
        if constexpr (hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>)
        {
            return hpx::async(policy.executor(), detail::define_task_block,
                HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(F, f));
        }
        else
        {
            detail::define_task_block(
                HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(F, f));
        }
    }

    /// Constructs a \a task_block, tr, and invokes the expression
    /// \a f(tr) on the user-provided object, \a f. This version uses
    /// \a parallel_policy for task scheduling.
    ///
    /// \tparam F   The type of the user defined function to invoke inside the
    ///             define_task_block (deduced). \a F shall be
    ///             MoveConstructible.
    ///
    /// \param f    The user defined function to invoke inside the task block.
    ///             Given an lvalue \a tr of type \a task_block, the expression,
    ///             (void)f(tr), shall be well-formed.
    ///
    /// Postcondition: All tasks spawned from \a f have finished execution.
    ///                A call to define_task_block may return on a different
    ///                thread than that on which it was called.
    ///
    /// \throws exception_list, as specified in Exception Handling.
    ///
    /// \note It is expected (but not mandated) that f will (directly or
    ///       indirectly) call tr.run(_callable_object_).
    ///
    template <typename F>
    void define_task_block(F&& f)
    {
        detail::define_task_block(hpx::execution::par, HPX_FORWARD(F, f));
    }

    /// Constructs a \a task_block, tr, and invokes the expression
    /// \a f(tr) on the user-provided object, \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the task block may be parallelized.
    /// \tparam F   The type of the user defined function to invoke inside the
    ///             define_task_block (deduced). \a F shall be MoveConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param f    The user defined function to invoke inside the define_task_block.
    ///             Given an lvalue \a tr of type \a task_block, the
    ///             expression, (void)f(tr), shall be well-formed.
    ///
    /// \throws exception_list, as specified in Exception Handling.
    ///
    /// Postcondition: All tasks spawned from \a f have finished execution.
    ///                A call to \a define_task_block_restore_thread always
    ///                returns on the
    ///                same thread as that on which it was called.
    ///
    /// \note It is expected (but not mandated) that f will (directly or
    ///       indirectly) call tr.run(_callable_object_).
    ///
    template <typename ExPolicy, typename F>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
    define_task_block_restore_thread(ExPolicy&& policy, F&& f)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        // By design we always return on the same (HPX-) thread as we started
        // executing define_task_block_restore_thread.
        return define_task_block(
            HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(F, f));
    }

    /// Constructs a \a task_block, tr, and invokes the expression
    /// \a f(tr) on the user-provided object, \a f. This version uses
    /// \a parallel_policy for task scheduling.
    ///
    /// \tparam F   The type of the user defined function to invoke inside the
    ///             define_task_block (deduced). \a F shall be
    ///             MoveConstructible.
    ///
    /// \param f    The user defined function to invoke inside the
    ///             define_task_block. Given an lvalue \a tr of type
    ///             \a task_block, the expression, (void)f(tr), shall be
    ///             well-formed.
    ///
    /// \throws exception_list, as specified in Exception Handling.
    ///
    /// Postcondition: All tasks spawned from \a f have finished execution.
    ///                A call to \a define_task_block_restore_thread always
    ///                returns on the same thread as that on which it was
    ///                called.
    ///
    /// \note It is expected (but not mandated) that f will (directly or
    ///       indirectly) call tr.run(_callable_object_).
    ///
    template <typename F>
    void define_task_block_restore_thread(F&& f)
    {
        // By design we always return on the same (HPX-) thread as we started
        // executing define_task_block_restore_thread.
        define_task_block_restore_thread(
            hpx::execution::par, HPX_FORWARD(F, f));
    }
}    // namespace hpx::experimental

/// \cond NOINTERNAL
namespace std {

    template <typename ExPolicy>
    hpx::experimental::task_block<ExPolicy>* addressof(
        hpx::experimental::task_block<ExPolicy>&) = delete;
}
/// \endcond

namespace hpx::parallel {

    using task_canceled_exception HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::task_canceled_exception is deprecated, use "
        "hpx::experimental::task_canceled_exception instead") =
        hpx::experimental::task_canceled_exception;

    template <typename ExPolicy = hpx::execution::parallel_policy>
    using task_block HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::task_block is deprecated, use "
        "hpx::experimental::task_block instead") =
        hpx::experimental::task_block<ExPolicy>;

    template <typename ExPolicy, typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_async_execution_policy<std::decay_t<ExPolicy>>::value)>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::define_task_block is deprecated, use "
        "hpx::experimental::define_task_block instead")
    hpx::future<void> define_task_block(ExPolicy&& policy, F&& f)
    {
        return hpx::experimental::define_task_block(policy, f);
    }

    template <typename ExPolicy, typename F,
        HPX_CONCEPT_REQUIRES_(
            !hpx::is_async_execution_policy<std::decay_t<ExPolicy>>::value)>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::define_task_block is deprecated, use "
        "hpx::experimental::define_task_block instead")
    void define_task_block(ExPolicy&& policy, F&& f)
    {
        return hpx::experimental::define_task_block(policy, f);
    }

    template <typename F>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::define_task_block is deprecated, use "
        "hpx::experimental::define_task_block instead")
    void define_task_block(F&& f)
    {
        return hpx::experimental::define_task_block(f);
    }

    template <typename ExPolicy, typename F>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::define_task_block is deprecated, use "
        "hpx::experimental::define_task_block instead")
    util::detail::algorithm_result_t<ExPolicy> define_task_block_restore_thread(
        ExPolicy&& policy, F&& f)
    {
        return hpx::experimental::define_task_block_restore_thread(policy, f);
    }

    template <typename F>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel:v2::define_task_block is deprecated, use "
        "hpx::experimental::define_task_block instead")
    void define_task_block_restore_thread(F&& f)
    {
        return hpx::experimental::define_task_block_restore_thread(f);
    }

}    // namespace hpx::parallel
