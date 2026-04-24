// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/stdexec_forward.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/executors/thread_pool_scheduler_bulk.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>

namespace hpx::execution::experimental {

    // P2079R10: Abstract backend interface for parallel_scheduler.
    // This mirrors stdexec's system_context_replaceability::parallel_scheduler_backend
    // but is expressed as a simple abstract class rather than using stdexec's __any
    // type-erasure machinery.
    //
    // The backend is responsible for:
    //   - schedule(): post a unit of work to the execution context
    //   - schedule_bulk_chunked(): post chunked bulk work
    //   - schedule_bulk_unchunked(): post unchunked bulk work
    //
    // The receiver_proxy / bulk_item_receiver_proxy interfaces allow the backend
    // to complete operations without knowing the concrete receiver type.

    // P2079R10 receiver_proxy: type-erased completion interface.
    // The backend calls these to signal completion back to the frontend.
    // stop_requested() allows the backend to poll for cancellation during
    // execution (partial substitute for try_query<inplace_stop_token>).
    struct parallel_scheduler_receiver_proxy
    {
        virtual ~parallel_scheduler_receiver_proxy() = default;
        virtual void set_value() noexcept = 0;
        virtual void set_error(std::exception_ptr) noexcept = 0;
        virtual void set_stopped() noexcept = 0;
        // P2079R10 4.2: backends can poll this to check if work should stop.
        // Returns true if the associated stop token has been signalled.
        virtual bool stop_requested() const noexcept
        {
            return false;
        }
    };

    // P2079R10 bulk_item_receiver_proxy: extends receiver_proxy with
    // execute(begin, end) for bulk work items.
    struct parallel_scheduler_bulk_item_receiver_proxy
      : parallel_scheduler_receiver_proxy
    {
        virtual void execute(std::size_t begin, std::size_t end) noexcept = 0;
    };

    // P2079R10 4.2: Pre-allocated storage for backend operation states.
    // The frontend provides a std::span<std::byte> of this size to each
    // backend method so the backend can avoid heap allocation.
    // Backends that need more can fall back to their own allocation.
    static constexpr std::size_t parallel_scheduler_storage_size = 256;
    static constexpr std::size_t parallel_scheduler_storage_alignment =
        alignof(std::max_align_t);

    // P2079R10 / P3927R2: Abstract backend interface
    struct parallel_scheduler_backend
    {
        virtual ~parallel_scheduler_backend() = default;

        // Schedule a single unit of work. On completion, call proxy.set_value().
        // storage: pre-allocated scratch space from the frontend's
        //          operation_state (parallel_scheduler_storage_size bytes).
        // P3927R2: parameter order is (receiver, storage)
        virtual void schedule(parallel_scheduler_receiver_proxy& proxy,
            std::span<std::byte> storage) noexcept = 0;

        // Schedule chunked bulk work of size count.
        // The backend partitions [0, count) into subranges and calls
        // proxy.execute(begin, end) for each subrange, then proxy.set_value().
        // P3927R2: parameter order is (shape, receiver, storage)
        virtual void schedule_bulk_chunked(std::size_t count,
            parallel_scheduler_bulk_item_receiver_proxy& proxy,
            std::span<std::byte> storage) noexcept = 0;

        // Schedule unchunked bulk work of size count.
        // The backend calls proxy.execute(i, i+1) for each i in [0, count),
        // then proxy.set_value().
        // P3927R2: parameter order is (shape, receiver, storage)
        virtual void schedule_bulk_unchunked(std::size_t count,
            parallel_scheduler_bulk_item_receiver_proxy& proxy,
            std::span<std::byte> storage) noexcept = 0;

        // Equality: two backends are equal if they share the same execution
        // context. Used by parallel_scheduler::operator==.
        virtual bool equal_to(
            parallel_scheduler_backend const& other) const noexcept = 0;

        // Access the underlying thread pool scheduler (HPX-specific).
        // Returns nullptr if this backend doesn't wrap a thread_pool_policy_scheduler.
        // Used by parallel_scheduler_domain::transform_sender to create
        // optimized thread_pool_bulk_sender directly (bypassing virtual dispatch
        // for bulk operations when the default HPX backend is in use).
        virtual thread_pool_policy_scheduler<hpx::launch> const*
        get_underlying_scheduler() const noexcept
        {
            return nullptr;
        }

        // Access the cached PU mask (HPX-specific).
        // Returns nullptr if unavailable.
        virtual hpx::threads::mask_type const* get_pu_mask() const noexcept
        {
            return nullptr;
        }
    };

    namespace detail {

        // Default HPX backend: wraps the existing thread_pool_policy_scheduler.
        // This is the backend returned by query_parallel_scheduler_backend()
        // unless the user provides a replacement via weak linking.
        class hpx_parallel_scheduler_backend final
          : public parallel_scheduler_backend
        {
        public:
            explicit hpx_parallel_scheduler_backend(
                thread_pool_policy_scheduler<hpx::launch> sched)
              : scheduler_(sched)
              , pu_mask_(hpx::execution::experimental::detail::full_mask(
                    hpx::execution::experimental::get_first_core(scheduler_),
                    hpx::execution::experimental::processing_units_count(
                        hpx::execution::experimental::null_parameters,
                        scheduler_, hpx::chrono::null_duration, 0)))
            {
            }

            void schedule(parallel_scheduler_receiver_proxy& proxy,
                std::span<std::byte>) noexcept override
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        scheduler_.execute(
                            [&proxy]() mutable { proxy.set_value(); });
                    },
                    [&](std::exception_ptr ep) {
                        proxy.set_error(HPX_MOVE(ep));
                    });
            }

            void schedule_bulk_chunked(std::size_t count,
                parallel_scheduler_bulk_item_receiver_proxy& proxy,
                std::span<std::byte>) noexcept override
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        auto num_threads = static_cast<std::uint32_t>(hpx::
                                execution::experimental::processing_units_count(
                                    hpx::execution::experimental::
                                        null_parameters,
                                    scheduler_, hpx::chrono::null_duration, 0));
                        auto chunk_size = hpx::execution::experimental::detail::
                            get_bulk_scheduler_chunk_size_chunked(
                                num_threads, count);

                        // Execute chunks sequentially on the thread pool
                        scheduler_.execute([&proxy, count, chunk_size]() {
                            for (std::size_t begin = 0; begin < count;
                                begin += chunk_size)
                            {
                                auto end = (std::min) (begin +
                                        static_cast<std::size_t>(chunk_size),
                                    count);
                                proxy.execute(begin, end);
                            }
                            proxy.set_value();
                        });
                    },
                    [&](std::exception_ptr ep) {
                        proxy.set_error(HPX_MOVE(ep));
                    });
            }

            void schedule_bulk_unchunked(std::size_t count,
                parallel_scheduler_bulk_item_receiver_proxy& proxy,
                std::span<std::byte>) noexcept override
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        scheduler_.execute([&proxy, count]() {
                            for (std::size_t i = 0; i < count; ++i)
                            {
                                proxy.execute(i, i + 1);
                            }
                            proxy.set_value();
                        });
                    },
                    [&](std::exception_ptr ep) {
                        proxy.set_error(HPX_MOVE(ep));
                    });
            }

            bool equal_to(
                parallel_scheduler_backend const& other) const noexcept override
            {
                auto const* p =
                    dynamic_cast<hpx_parallel_scheduler_backend const*>(&other);
                return p != nullptr && p->scheduler_ == scheduler_;
            }

            thread_pool_policy_scheduler<hpx::launch> const*
            get_underlying_scheduler() const noexcept override
            {
                return &scheduler_;
            }

            hpx::threads::mask_type const* get_pu_mask() const noexcept override
            {
                return &pu_mask_;
            }

        private:
            thread_pool_policy_scheduler<hpx::launch> scheduler_;
            hpx::threads::mask_type pu_mask_;
        };

        // Singleton-like shared thread pool for parallel_scheduler
        inline hpx::threads::thread_pool_base* get_default_parallel_pool()
        {
            // clang-format off
            static hpx::threads::thread_pool_base* default_pool =
                hpx::threads::detail::get_self_or_default_pool();
            // clang-format on
            return default_pool;
        }

    }    // namespace detail

    // P2079R10: query_parallel_scheduler_backend()
    // Returns a shared_ptr to the parallel_scheduler_backend.
    // This is the default implementation; users can replace it
    // by providing their own shared_ptr<parallel_scheduler_backend>.
    //
    // Note: Unlike stdexec's approach, HPX uses a function
    // pointer that can be replaced at runtime via
    // set_parallel_scheduler_backend_factory(). This avoids platform-specific
    // weak-linking issues while providing the same replaceability.
    using parallel_scheduler_backend_factory_t =
        std::shared_ptr<parallel_scheduler_backend> (*)();

    namespace detail {

        // Default factory creates the HPX backend
        inline std::shared_ptr<parallel_scheduler_backend>
        default_parallel_scheduler_backend_factory()
        {
            auto pool = get_default_parallel_pool();
            if (!pool)
            {
                std::terminate();
            }
            return std::make_shared<hpx_parallel_scheduler_backend>(
                thread_pool_policy_scheduler<hpx::launch>(
                    pool, hpx::launch::async));
        }

        // Mutex protecting the live backend instance.
        inline std::mutex& get_backend_mutex() noexcept
        {
            static std::mutex mtx;
            return mtx;
        }

        // The live backend instance. nullptr until first query.
        // Protected by get_backend_mutex().
        inline std::shared_ptr<parallel_scheduler_backend>&
        get_backend_storage() noexcept
        {
            static std::shared_ptr<parallel_scheduler_backend> backend;
            return backend;
        }

        // Storage for the current factory (only used to create the first backend).
        inline parallel_scheduler_backend_factory_t&
        get_backend_factory_storage() noexcept
        {
            static parallel_scheduler_backend_factory_t factory =
                &default_parallel_scheduler_backend_factory;
            return factory;
        }

    }    // namespace detail

    // P2079R10: Get the current parallel_scheduler_backend.
    // Thread-safe. Creates the default backend on first call via the factory.
    // Can be replaced at any time via set_parallel_scheduler_backend().
    inline std::shared_ptr<parallel_scheduler_backend>
    query_parallel_scheduler_backend()
    {
        std::lock_guard<std::mutex> lock(detail::get_backend_mutex());
        auto& storage = detail::get_backend_storage();
        if (!storage)
        {
            storage = detail::get_backend_factory_storage()();
        }
        return storage;
    }

    // P2079R10: Replace the parallel scheduler backend factory.
    // The new factory is used the next time query_parallel_scheduler_backend()
    // creates a backend (only if no backend has been created yet, or after
    // set_parallel_scheduler_backend() clears the current one).
    inline parallel_scheduler_backend_factory_t
    set_parallel_scheduler_backend_factory(
        parallel_scheduler_backend_factory_t new_factory) noexcept
    {
        std::lock_guard<std::mutex> lock(detail::get_backend_mutex());
        auto& storage = detail::get_backend_factory_storage();
        auto old = storage;
        storage = new_factory;
        return old;
    }

    // P2079R10: Directly replace the active backend.
    // Takes effect immediately: the next get_parallel_scheduler() call
    // returns a scheduler backed by new_backend.
    // Thread-safe, but must not be called while active operations are
    // in-flight on the current backend.
    inline void set_parallel_scheduler_backend(
        std::shared_ptr<parallel_scheduler_backend> new_backend)
    {
        std::lock_guard<std::mutex> lock(detail::get_backend_mutex());
        detail::get_backend_storage() = HPX_MOVE(new_backend);
    }

}    // namespace hpx::execution::experimental

#endif    // HPX_HAVE_STDEXEC
