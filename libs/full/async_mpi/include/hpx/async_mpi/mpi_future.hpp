//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/mpi_base/mpi.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace mpi { namespace experimental {

    // by convention the title is 7 chars (for alignment)
    using print_on = debug::enable_print<false>;
    static constexpr print_on mpi_debug("MPI_FUT");

    namespace detail {

        using mutex_type = hpx::lcos::local::spinlock;

        // extract MPI error message
        HPX_EXPORT std::string error_message(int code);

        // mutex needed to protect mpi request vector, note that the
        // mpi poll function takes place inside the main scheduling loop
        // of hpx and not on an hpx worker thread, so we must use std:mutex
        HPX_EXPORT mutex_type& get_vector_mtx();

        // -----------------------------------------------------------------
        // An implementation of future_data for MPI
        struct future_data : hpx::lcos::detail::future_data<int>
        {
            HPX_NON_COPYABLE(future_data);

            using init_no_addref =
                typename hpx::lcos::detail::future_data<int>::init_no_addref;

            // default empty constructor
            future_data() = default;

            // constructor that takes a request
            future_data(init_no_addref no_addref, MPI_Request request)
              : hpx::lcos::detail::future_data<int>(no_addref)
              , request_(request)
            {
            }

            // constructor used for creation directly by invoke
            future_data(init_no_addref no_addref)
              : hpx::lcos::detail::future_data<int>(no_addref)
            {
            }

            // The native MPI request handle owned by this future data
            MPI_Request request_;
        };

        // -----------------------------------------------------------------
        // intrusive pointer for future_data
        using future_data_ptr = memory::intrusive_ptr<future_data>;

        // -----------------------------------------------------------------
        // a convenience structure to hold state vars
        // used extensivey with debug::print to display rank etc
        struct mpi_info
        {
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
        };

        // an instance of mpi_info that we store data in
        HPX_EXPORT mpi_info& get_mpi_info();

        // stream operator to display debug mpi_info
        HPX_EXPORT std::ostream& operator<<(
            std::ostream& os, mpi_info const& i);

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors is we enable the error handler
        HPX_EXPORT extern MPI_Errhandler hpx_mpi_errhandler;

        // function that converts an MPI error into an exception
        HPX_EXPORT void hpx_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        // we track requests and future data in two vectors even though
        // we have the request stored in the future data already
        // the reason for this is because we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item (using a list)
        HPX_EXPORT std::vector<MPI_Request>& get_active_requests();
        HPX_EXPORT std::vector<future_data_ptr>& get_active_futures();

        // -----------------------------------------------------------------
        // define a lockfree queue type to place requests in prior to handling
        // this is done only to avoid taking a lock every time a request is
        // returned from MPI. Instead the requests are placed into a queue
        // and the polling code pops them prior to calling Testany
        using queue_type = concurrency::ConcurrentQueue<future_data_ptr>;
        queue_type& get_request_queue();

        // -----------------------------------------------------------------
        // used internally to add an MPI_Request to the lockfree queue
        // that will be used by the polling routines to check when requests
        // have completed
        HPX_EXPORT void add_to_request_queue(future_data_ptr data);

        // -----------------------------------------------------------------
        // used internally to query how many requests are 'in flight'
        // the limiting executor can use this to throttle back a task if
        // too many mpi requests are being spawned at once
        // unfortunately, the lock-free queue can only return an estimate
        // of the queue size, so this is not guaranteed to be precise
        HPX_EXPORT std::size_t get_number_of_enqueued_requests();

        // -----------------------------------------------------------------
        // used internally to add a request to the main polling vector
        // that is passed to MPI_Testany
        HPX_EXPORT void add_to_request_vector(future_data_ptr data);

        // -----------------------------------------------------------------
        // used internally to query how many requests are 'in flight'
        // these are requests that are being polled for actively
        // and not the same as the requests enqueued
        HPX_EXPORT std::size_t get_number_of_active_requests();

    }    // namespace detail

    // -----------------------------------------------------------------
    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    HPX_EXPORT void set_error_handler();

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    HPX_EXPORT hpx::future<void> get_future(MPI_Request request);

    // -----------------------------------------------------------------
    // return a future from an async call to MPI_Ixxx function
    namespace detail {

        template <typename F, typename... Ts>
        hpx::future<int> async(F f, Ts&&... ts)
        {
            // create a future data shared state
            detail::future_data_ptr data =
                new detail::future_data(detail::future_data::init_no_addref{});

            // invoke the call to MPI_Ixxx, ignore the returned result for now
            int result = f(std::forward<Ts>(ts)..., &data->request_);
            (void) result;    // silence unused var warning

            // enqueue the future state internally for processing
            detail::add_to_request_queue(data);

            // return a future bound to the shared state
            using traits::future_access;
            return future_access<hpx::future<int>>::create(std::move(data));
        }
    }    // namespace detail

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::experimental::future ready
    // when found
    HPX_EXPORT void poll();

    // -----------------------------------------------------------------
    // This is not completely safe as it will return when the request vector is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    inline void wait()
    {
        hpx::util::yield_while([]() {
            std::unique_lock<detail::mutex_type> lk(
                detail::get_vector_mtx(), std::try_to_lock);
            if (!lk.owns_lock())
            {
                return true;
            }
            return (detail::get_active_futures().size() > 0);
        });
    }

    template <typename F>
    inline void wait(F&& f)
    {
        hpx::util::yield_while([&]() {
            std::unique_lock<detail::mutex_type> lk(
                detail::get_vector_mtx(), std::try_to_lock);
            if (!lk.owns_lock())
            {
                return true;
            }
            return (detail::get_active_futures().size() > 0) || f();
        });
    }

    // -----------------------------------------------------------------
    namespace detail {

        HPX_EXPORT void register_polling(hpx::threads::thread_pool_base&);
        HPX_EXPORT void unregister_polling(hpx::threads::thread_pool_base&);
    }    // namespace detail

    // initialize the hpx::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    HPX_EXPORT void init(bool init_mpi = false,
        std::string const& pool_name = "", bool init_errorhandler = false);

    // -----------------------------------------------------------------
    HPX_EXPORT void finalize(std::string const& pool_name = "");

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is
    // handled elsewhere
    struct HPX_NODISCARD enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "")
          : pool_name_(pool_name)
        {
            mpi::experimental::init(false, pool_name);
        }

        ~enable_user_polling()
        {
            mpi::experimental::finalize(pool_name_);
        }

    private:
        std::string pool_name_;
    };

    // -----------------------------------------------------------------
    template <typename... Args>
    inline void debug(Args&&... args)
    {
        mpi_debug.debug(detail::get_mpi_info(), std::forward<Args>(args)...);
    }
}}}    // namespace hpx::mpi::experimental
