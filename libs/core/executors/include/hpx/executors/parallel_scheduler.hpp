// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/stdexec_forward.hpp>
#include <hpx/executors/parallel_scheduler_backend.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/executors/thread_pool_scheduler_bulk.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <cstddef>
#include <exception>
#include <memory>
#include <tuple>
#include <type_traits>
#include <variant>

namespace hpx::execution::experimental {

#if defined(HPX_HAVE_STDEXEC)
    // Forward declaration for parallel_scheduler_domain
    class parallel_scheduler;

    inline parallel_scheduler get_parallel_scheduler();

    // Virtual bulk dispatch infrastructure for P2079R10.
    //
    // transform_sender must return a single concrete type, but we
    // need two execution paths:
    //   - Fast path (default HPX backend): thread_pool_bulk_sender
    //     with work-stealing, NUMA awareness, etc.
    //   - Virtual path (custom backends): routes through
    //     backend->schedule_bulk_chunked/unchunked().
    //
    // Solution: type-erase the operation state behind a virtual
    // base class. Cost: one heap allocation per bulk operation.
    // For bulk work processing thousands of elements, this is
    // negligible.
    namespace detail {

        // Virtual base for type-erased bulk operation states.
        struct base_parallel_bulk_op
        {
            virtual ~base_parallel_bulk_op() = default;
            virtual void start() noexcept = 0;
        };

        // Fast path: wraps thread_pool_bulk_sender's connected
        // operation state. Zero overhead beyond the heap allocation.
        template <typename FastSender, typename Receiver>
        struct fast_parallel_bulk_op final : base_parallel_bulk_op
        {
            using inner_op_t =
                hpx::execution::experimental::connect_result_t<FastSender,
                    Receiver>;

            inner_op_t inner_;

            fast_parallel_bulk_op(FastSender&& s, Receiver&& r)
              : inner_(hpx::execution::experimental::connect(
                    HPX_MOVE(s), HPX_MOVE(r)))
            {
            }

            void start() noexcept override
            {
                hpx::execution::experimental::start(inner_);
            }
        };

        // Virtual dispatch path: connects child sender to an internal
        // receiver. When the child completes with values, creates a
        // bulk_item_proxy and calls backend->schedule_bulk_chunked()
        // or schedule_bulk_unchunked().
        template <typename F, bool IsChunked, typename ChildSender,
            typename Receiver>
        struct virtual_parallel_bulk_op final : base_parallel_bulk_op
        {
            std::shared_ptr<parallel_scheduler_backend> backend_;
            std::size_t count_;
            F f_;
            std::decay_t<Receiver> receiver_;

            // Pre-allocated storage for the backend.
            alignas(parallel_scheduler_storage_alignment)
                std::byte storage_[parallel_scheduler_storage_size];

            // Heap-allocated proxy (created when child completes).
            // Must be a member so it survives async backend execution.
            std::unique_ptr<parallel_scheduler_bulk_item_receiver_proxy>
                active_proxy_;

            // Internal receiver that catches child's completion and
            // triggers the backend bulk dispatch.
            struct child_receiver
            {
                using receiver_concept =
                    hpx::execution::experimental::receiver_t;
                virtual_parallel_bulk_op* self_;

                template <typename... Vs>
                friend void tag_invoke(
                    hpx::execution::experimental::set_value_t,
                    child_receiver&& r, Vs&&... vs) noexcept
                {
                    r.self_->do_bulk(HPX_FORWARD(Vs, vs)...);
                }

                friend void tag_invoke(
                    hpx::execution::experimental::set_error_t,
                    child_receiver&& r, std::exception_ptr ep) noexcept
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(r.self_->receiver_), HPX_MOVE(ep));
                }

                friend void tag_invoke(
                    hpx::execution::experimental::set_stopped_t,
                    child_receiver&& r) noexcept
                {
                    hpx::execution::experimental::set_stopped(
                        HPX_MOVE(r.self_->receiver_));
                }

                friend auto tag_invoke(hpx::execution::experimental::get_env_t,
                    child_receiver const& r) noexcept
                {
                    return hpx::execution::experimental::get_env(
                        r.self_->receiver_);
                }
            };

            // Connected child sender's operation state.
            hpx::execution::experimental::connect_result_t<ChildSender,
                child_receiver>
                child_op_;

            virtual_parallel_bulk_op(
                std::shared_ptr<parallel_scheduler_backend> b,
                std::size_t count, F f, ChildSender&& child, Receiver&& rcvr)
              : backend_(HPX_MOVE(b))
              , count_(count)
              , f_(HPX_MOVE(f))
              , receiver_(HPX_FORWARD(Receiver, rcvr))
              , child_op_(hpx::execution::experimental::connect(
                    HPX_FORWARD(ChildSender, child), child_receiver{this}))
            {
            }

            void start() noexcept override
            {
                hpx::execution::experimental::start(child_op_);
            }

            // Called by child_receiver::set_value when the child
            // sender completes. Creates a type-erased bulk proxy
            // that captures the values and calls f(i, values...)
            // in execute(), then dispatches to the backend.
            template <typename... Vs>
            void do_bulk(Vs&&... vs) noexcept
            {
                // Concrete proxy that captures values from the
                // child sender and invokes the bulk function.
                struct concrete_proxy final
                  : parallel_scheduler_bulk_item_receiver_proxy
                {
                    virtual_parallel_bulk_op& op_;
                    std::tuple<std::decay_t<Vs>...> values_;

                    concrete_proxy(virtual_parallel_bulk_op& o, Vs&&... vs)
                      : op_(o)
                      , values_(HPX_FORWARD(Vs, vs)...)
                    {
                    }

                    void execute(
                        std::size_t begin, std::size_t end) noexcept override
                    {
                        if constexpr (IsChunked)
                        {
                            // Chunked: f expects (begin, end, ...vals)
                            std::apply(
                                [&](auto&... vals) {
                                    op_.f_(begin, end, vals...);
                                },
                                values_);
                        }
                        else
                        {
                            // Unchunked: f expects (index, ...vals)
                            for (std::size_t i = begin; i < end; ++i)
                            {
                                std::apply(
                                    [&](auto&... vals) { op_.f_(i, vals...); },
                                    values_);
                            }
                        }
                    }

                    void set_value() noexcept override
                    {
                        // Bulk passes child values through to receiver.
                        std::apply(
                            [&](auto&&... vals) {
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(op_.receiver_), HPX_MOVE(vals)...);
                            },
                            std::move(values_));
                    }

                    void set_error(std::exception_ptr ep) noexcept override
                    {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(op_.receiver_), HPX_MOVE(ep));
                    }

                    void set_stopped() noexcept override
                    {
                        hpx::execution::experimental::set_stopped(
                            HPX_MOVE(op_.receiver_));
                    }

                    bool stop_requested() const noexcept override
                    {
                        return stdexec::get_stop_token(
                            stdexec::get_env(op_.receiver_))
                            .stop_requested();
                    }
                };

                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        active_proxy_ = std::make_unique<concrete_proxy>(
                            *this, HPX_FORWARD(Vs, vs)...);
                        auto& proxy_ref =
                            static_cast<concrete_proxy&>(*active_proxy_);

                        std::span<std::byte> span(storage_);
                        if constexpr (IsChunked)
                        {
                            backend_->schedule_bulk_chunked(
                                count_, proxy_ref, span);
                        }
                        else
                        {
                            backend_->schedule_bulk_unchunked(
                                count_, proxy_ref, span);
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver_), HPX_MOVE(ep));
                    });
            }
        };

        // Unified sender returned by parallel_scheduler_domain's
        // transform_sender. Holds either the fast-path
        // thread_pool_bulk_sender or virtual dispatch data.
        template <typename FastSender, typename ChildSender, typename F,
            bool IsChunked>
        struct parallel_bulk_dispatch_sender
        {
            using sender_concept = stdexec::sender_t;

            struct fast_path_data
            {
                FastSender sender_;
            };

            struct virtual_path_data
            {
                std::shared_ptr<parallel_scheduler_backend> backend_;
                std::size_t count_;
                F f_;
                ChildSender child_;
            };

            std::variant<fast_path_data, virtual_path_data> data_;

            // Completion signatures: same as the child sender's,
            // with set_error(exception_ptr) added (bulk can fail).
            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                parallel_bulk_dispatch_sender const&, Env const&)
                -> hpx::execution::experimental::
                    transform_completion_signatures_of<ChildSender, Env,
                        hpx::execution::experimental::completion_signatures<
                            hpx::execution::experimental::set_error_t(
                                std::exception_ptr)>>;

            // Unified operation state: holds type-erased op via
            // unique_ptr<base_parallel_bulk_op>.
            template <typename Receiver>
            struct dispatch_op
            {
                std::unique_ptr<base_parallel_bulk_op> impl_;

                explicit dispatch_op(std::unique_ptr<base_parallel_bulk_op> p)
                  : impl_(HPX_MOVE(p))
                {
                }

                dispatch_op(dispatch_op&&) = delete;
                dispatch_op(dispatch_op const&) = delete;
                dispatch_op& operator=(dispatch_op&&) = delete;
                dispatch_op& operator=(dispatch_op const&) = delete;

                friend void tag_invoke(hpx::execution::experimental::start_t,
                    dispatch_op& os) noexcept
                {
                    os.impl_->start();
                }
            };

            // connect: creates the right op state behind the
            // type-erased pointer.
            template <typename Receiver>
            friend dispatch_op<std::decay_t<Receiver>> tag_invoke(
                hpx::execution::experimental::connect_t,
                parallel_bulk_dispatch_sender&& self, Receiver&& rcvr)
            {
                if (auto* fast = std::get_if<fast_path_data>(&self.data_))
                {
                    return dispatch_op<std::decay_t<Receiver>>{
                        std::make_unique<fast_parallel_bulk_op<FastSender,
                            std::decay_t<Receiver>>>(HPX_MOVE(fast->sender_),
                            HPX_FORWARD(Receiver, rcvr))};
                }
                else
                {
                    auto& vp = std::get<virtual_path_data>(self.data_);
                    return dispatch_op<std::decay_t<Receiver>>{
                        std::make_unique<virtual_parallel_bulk_op<F, IsChunked,
                            ChildSender, std::decay_t<Receiver>>>(
                            HPX_MOVE(vp.backend_), vp.count_, HPX_MOVE(vp.f_),
                            HPX_MOVE(vp.child_), HPX_FORWARD(Receiver, rcvr))};
                }
            }
        };

    }    // namespace detail

    // P2079R10: Domain for parallel_scheduler bulk operations.
    // The existing thread_pool_domain checks __completes_on with
    // thread_pool_policy_scheduler, but parallel_scheduler's sender
    // returns parallel_scheduler as the completion scheduler.
    // This domain bridges the gap by extracting the underlying
    // thread_pool_policy_scheduler and delegating to HPX's optimized
    // thread_pool_bulk_sender.
    struct parallel_scheduler_domain : stdexec::default_domain
    {
        template <bulk_chunked_or_unchunked_sender Sender, typename Env>
        auto transform_sender(hpx::execution::experimental::set_value_t,
            Sender&& sndr, Env const& env) const
        {
            if constexpr (hpx::execution::experimental::stdexec_internal::
                              __completes_on<Sender, parallel_scheduler, Env>)
            {
                // Extract bulk parameters using structured binding
                auto&& [tag, data, child] = sndr;
                auto&& [pol, shape, f] = data;

                // Get the parallel_scheduler from the child sender's
                // completion scheduler (completes_on pattern)
                auto par_sched = [&]() {
                    if constexpr (
                        hpx::is_invocable_v<
                            hpx::execution::experimental::
                                get_completion_scheduler_t<
                                    hpx::execution::experimental::set_value_t>,
                            decltype(hpx::execution::experimental::get_env(
                                child))>)
                    {
                        return hpx::execution::experimental::
                            get_completion_scheduler<
                                hpx::execution::experimental::set_value_t>(
                                hpx::execution::experimental::get_env(child));
                    }
                    else
                    {
                        return hpx::execution::experimental::
                            get_parallel_scheduler();
                    }
                }();

                // Extract the underlying thread pool scheduler from the
                // backend. For the default HPX backend this returns the
                // concrete thread_pool_policy_scheduler; for custom backends
                // it returns nullptr (bulk goes through virtual dispatch).
                auto const* underlying_ptr =
                    par_sched.get_underlying_scheduler();
                auto const* pu_mask_ptr = par_sched.get_pu_mask();

                // Only bulk_chunked_t uses the chunked f(begin, end, ...)
                // signature. Both bulk_t (P3481R5 high-level) and
                // bulk_unchunked_t use the unchunked f(index, ...) signature
                // that HPX's bulk users pass. Treating bulk_t as chunked here
                // would force f(begin, end, ...) on user lambdas that take a
                // single index, causing a template instantiation failure.
                constexpr bool is_chunked = stdexec::__sender_for<Sender,
                    hpx::execution::experimental::bulk_chunked_t>;

                // Determine parallelism at compile time from policy type
                // (pol is a __policy_wrapper, use __get() to unwrap)
                constexpr bool is_parallel =
                    !is_sequenced_policy_v<std::decay_t<decltype(pol.__get())>>;

                constexpr bool is_unsequenced = is_unsequenced_bulk_policy_v<
                    std::decay_t<decltype(pol.__get())>>;

                auto iota_shape =
                    hpx::util::counting_shape(decltype(shape){0}, shape);

                // Compute the fast-path sender type (needed even on the
                // virtual path so both branches return the same type).
                using fast_sender_t = hpx::execution::experimental::detail::
                    thread_pool_bulk_sender<hpx::launch,
                        std::decay_t<decltype(child)>,
                        std::decay_t<decltype(iota_shape)>,
                        std::decay_t<decltype(f)>, is_chunked, is_parallel,
                        is_unsequenced>;

                using dispatch_sender_t =
                    detail::parallel_bulk_dispatch_sender<fast_sender_t,
                        std::decay_t<decltype(child)>,
                        std::decay_t<decltype(f)>, is_chunked>;

                // Fast path: default HPX backend with underlying scheduler
                // available. Create optimized thread_pool_bulk_sender
                // with work-stealing, NUMA awareness, etc.
                if (underlying_ptr != nullptr && pu_mask_ptr != nullptr)
                {
                    auto underlying = *underlying_ptr;
                    hpx::threads::mask_type pu_mask = *pu_mask_ptr;

                    auto fast_sender = fast_sender_t(HPX_MOVE(underlying),
                        HPX_FORWARD(decltype(child), child),
                        HPX_MOVE(iota_shape), HPX_FORWARD(decltype(f), f),
                        HPX_MOVE(pu_mask));

                    return dispatch_sender_t{
                        typename dispatch_sender_t::fast_path_data{
                            HPX_MOVE(fast_sender)}};
                }

                // Virtual dispatch path: custom backend without an
                // underlying thread_pool_policy_scheduler. Routes
                // through backend->schedule_bulk_chunked/unchunked().
                return dispatch_sender_t{
                    typename dispatch_sender_t::virtual_path_data{
                        par_sched.get_backend(),
                        static_cast<std::size_t>(shape),
                        HPX_FORWARD(decltype(f), f),
                        HPX_FORWARD(decltype(child), child)}};
            }
            else
            {
                // P2079R10: bulk operations require the parallel_scheduler
                // in the environment. Add a continues_on transition to the
                // parallel_scheduler before the bulk algorithm.
                static_assert(
                    hpx::execution::experimental::stdexec_internal::
                        __completes_on<Sender, parallel_scheduler, Env>,
                    "Cannot dispatch bulk algorithm to the parallel_scheduler: "
                    "no parallel_scheduler found in the environment. "
                    "Add a continues_on transition to the parallel_scheduler "
                    "before the bulk algorithm.");
            }
        }
    };

    // P2079R10 parallel_scheduler implementation.
    // Stores a shared_ptr<parallel_scheduler_backend> for replaceability.
    // The default backend wraps HPX's thread_pool_policy_scheduler.
    class parallel_scheduler
    {
    public:
        parallel_scheduler() = delete;

        // P2079R10: Construct from a backend shared_ptr.
        // This is the primary constructor used by get_parallel_scheduler().
        explicit parallel_scheduler(
            std::shared_ptr<parallel_scheduler_backend> backend) noexcept
          : backend_(HPX_MOVE(backend))
        {
        }

        parallel_scheduler(parallel_scheduler const& other) noexcept = default;
        parallel_scheduler(parallel_scheduler&& other) noexcept = default;
        parallel_scheduler& operator=(
            parallel_scheduler const&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler&&) noexcept = default;

        // P2079R10: equality means same backend implementation.
        friend bool operator==(parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            if (lhs.backend_ == rhs.backend_)
                return true;
            if (!lhs.backend_ || !rhs.backend_)
                return false;
            return lhs.backend_->equal_to(*rhs.backend_);
        }

        // P2079R10: query() member for forward progress guarantee
        // (modern stdexec pattern, preferred over tag_invoke)
        constexpr forward_progress_guarantee query(
            get_forward_progress_guarantee_t) const noexcept
        {
            return forward_progress_guarantee::parallel;
        }

        // P2079R10: operation_state owns the receiver and manages the
        // frontend/backend boundary. On start(), it checks the stop token
        // and then delegates to the backend.
        template <typename Receiver>
        struct operation_state
        {
            // Concrete receiver_proxy that adapts the actual Receiver
            // to the type-erased proxy interface.
            struct concrete_receiver_proxy final
              : parallel_scheduler_receiver_proxy
            {
                std::decay_t<Receiver>& receiver_;

                explicit concrete_receiver_proxy(
                    std::decay_t<Receiver>& rcvr) noexcept
                  : receiver_(rcvr)
                {
                }

                void set_value() noexcept override
                {
                    hpx::execution::experimental::set_value(
                        HPX_MOVE(receiver_));
                }

                void set_error(std::exception_ptr ep) noexcept override
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(receiver_), HPX_MOVE(ep));
                }

                void set_stopped() noexcept override
                {
                    hpx::execution::experimental::set_stopped(
                        HPX_MOVE(receiver_));
                }

                // P2079R10 4.2: allow backends to poll for cancellation.
                // Forwards the stop token state of the actual receiver.
                bool stop_requested() const noexcept override
                {
                    return stdexec::get_stop_token(stdexec::get_env(receiver_))
                        .stop_requested();
                }
            };

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver_;
            std::shared_ptr<parallel_scheduler_backend> backend_;
            // The proxy must be a member (not a local) because the
            // backend's schedule() posts work asynchronously. The
            // operation_state outlives the completion per the
            // sender/receiver protocol.
            concrete_receiver_proxy proxy_;

            // P2079R10 4.2: pre-allocated storage for the backend.
            alignas(parallel_scheduler_storage_alignment)
                std::byte storage_[parallel_scheduler_storage_size];

            template <typename Receiver_>
            operation_state(Receiver_&& receiver,
                std::shared_ptr<parallel_scheduler_backend> backend)
              : receiver_(HPX_FORWARD(Receiver_, receiver))
              , backend_(HPX_MOVE(backend))
              , proxy_(receiver_)
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                // P2079R10 4.1: if stop_token is stopped, complete
                // with set_stopped as soon as is practical.
                auto stop_token =
                    stdexec::get_stop_token(stdexec::get_env(os.receiver_));
                if (stop_token.stop_requested())
                {
                    stdexec::set_stopped(HPX_MOVE(os.receiver_));
                    return;
                }

                // Delegate to the backend via the member proxy,
                // passing pre-allocated storage per P2079R10 / P3927R2.
                os.backend_->schedule(
                    os.proxy_, std::span<std::byte>(os.storage_));
            }
        };

        // Nested sender type
        template <typename Scheduler>
        struct sender
        {
            Scheduler sched_;

            using sender_concept = stdexec::sender_t;
            using completion_signatures =
                stdexec::completion_signatures<stdexec::set_value_t(),
                    stdexec::set_error_t(std::exception_ptr),
                    stdexec::set_stopped_t()>;

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender const& s,
                Receiver&& receiver) noexcept(std::
                    is_nothrow_constructible_v<std::decay_t<Receiver>,
                        Receiver>)
            {
                return {
                    HPX_FORWARD(Receiver, receiver), s.sched_.get_backend()};
            }

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender&& s,
                Receiver&& receiver) noexcept(std::
                    is_nothrow_constructible_v<std::decay_t<Receiver>,
                        Receiver>)
            {
                return {
                    HPX_FORWARD(Receiver, receiver), s.sched_.get_backend()};
            }

            struct env
            {
                Scheduler const& sched_;

                // P2079R10: expose completion scheduler for set_value_t
                // and set_stopped_t
                auto query(
                    stdexec::get_completion_scheduler_t<stdexec::set_value_t>)
                    const noexcept
                {
                    return sched_;
                }

                auto query(
                    stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>)
                    const noexcept
                {
                    return sched_;
                }

#if defined(HPX_HAVE_STDEXEC)
                // Domain query
                parallel_scheduler_domain query(
                    stdexec::get_domain_t) const noexcept
                {
                    return {};
                }
#endif
            };

            friend env tag_invoke(stdexec::get_env_t, sender const& s) noexcept
            {
                return {s.sched_};
            }
        };

        // Direct schedule() member for modern stdexec
        sender<parallel_scheduler> schedule() const noexcept
        {
            return {*this};
        }

#if defined(HPX_HAVE_STDEXEC)
        // Domain customization for bulk operations
        parallel_scheduler_domain query(stdexec::get_domain_t) const noexcept
        {
            return {};
        }

        // Required for stdexec domain resolution: when a bulk sender's
        // completing domain is resolved, stdexec queries the completion
        // scheduler with get_completion_domain_t<set_value_t>. Without
        // this, the resolution falls to default_domain and our
        // parallel_scheduler_domain::transform_sender is never called.
        parallel_scheduler_domain query(
            stdexec::get_completion_domain_t<stdexec::set_value_t>)
            const noexcept
        {
            return {};
        }
#endif

        // Access the backend (for connect and domain transform).
        std::shared_ptr<parallel_scheduler_backend> const& get_backend()
            const noexcept
        {
            return backend_;
        }

        // HPX-specific: access the underlying thread pool scheduler
        // from the backend (returns nullptr for custom backends).
        thread_pool_policy_scheduler<hpx::launch> const*
        get_underlying_scheduler() const noexcept
        {
            return backend_ ? backend_->get_underlying_scheduler() : nullptr;
        }

        // HPX-specific: access the cached PU mask from the backend
        // (returns nullptr for custom backends).
        hpx::threads::mask_type const* get_pu_mask() const noexcept
        {
            return backend_ ? backend_->get_pu_mask() : nullptr;
        }

    private:
        std::shared_ptr<parallel_scheduler_backend> backend_;
    };

    // Stream output operator for parallel_scheduler
    inline std::ostream& operator<<(std::ostream& os, parallel_scheduler const&)
    {
        return os << "parallel_scheduler";
    }

    // P2079R10 get_parallel_scheduler function.
    // Uses query_parallel_scheduler_backend() to obtain the backend,
    // which can be replaced via set_parallel_scheduler_backend_factory().
    inline parallel_scheduler get_parallel_scheduler()
    {
        auto backend = query_parallel_scheduler_backend();
        if (!backend)
        {
            std::
                terminate();    // As per P2079R10, terminate if backend is unavailable
        }
        return parallel_scheduler(HPX_MOVE(backend));
    }

#endif    // HPX_HAVE_STDEXEC

}    // namespace hpx::execution::experimental
