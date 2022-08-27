//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_mpi/mpi_exception.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/mpi_base/mpi.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace mpi { namespace experimental {

    // -----------------------------------------------------------------
    namespace detail {

        using request_callback_function_type =
            hpx::move_only_function<void(int)>;

        HPX_CORE_EXPORT void add_request_callback(
            request_callback_function_type&& f, MPI_Request req);
        HPX_CORE_EXPORT void register_polling(hpx::threads::thread_pool_base&);
        HPX_CORE_EXPORT void unregister_polling(
            hpx::threads::thread_pool_base&);

    }    // namespace detail

    // by convention the title is 7 chars (for alignment)
    using print_on = debug::enable_print<false>;
    static constexpr print_on mpi_debug("MPI_FUT");

    namespace detail {

        using mutex_type = hpx::lcos::local::spinlock;

        // mutex needed to protect mpi request vector, note that the
        // mpi poll function takes place inside the main scheduling loop
        // of hpx and not on an hpx worker thread, so we must use std:mutex
        HPX_CORE_EXPORT mutex_type& get_vector_mtx();

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
                add_callback();
            }

            // constructor used for creation directly by invoke
            future_data(init_no_addref no_addref)
              : hpx::lcos::detail::future_data<int>(no_addref)
            {
            }

            // Used when the request was not available when constructing
            // future_data
            void add_callback()
            {
                add_request_callback(
                    [fdp = hpx::intrusive_ptr<future_data>(this)](int status) {
                        if (status == MPI_SUCCESS)
                        {
                            // mark the future as ready by setting the shared_state
                            fdp->set_data(MPI_SUCCESS);
                        }
                        else
                        {
                            fdp->set_exception(
                                std::make_exception_ptr(mpi_exception(status)));
                        }
                    },
                    request_);
            }

            // The native MPI request handle owned by this future data
            MPI_Request request_;
        };

        // -----------------------------------------------------------------
        // intrusive pointer for future_data
        using future_data_ptr = hpx::intrusive_ptr<future_data>;

        // -----------------------------------------------------------------
        // a convenience structure to hold state vars
        // used extensivey with debug::print to display rank etc
        struct mpi_info
        {
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
            // requests vector holds the requests that are checked
            std::atomic<std::uint32_t> requests_vector_size_{0};
            // requests queue holds the requests recently added
            std::atomic<std::uint32_t> requests_queue_size_{0};
        };

        // an instance of mpi_info that we store data in
        HPX_CORE_EXPORT mpi_info& get_mpi_info();

        // stream operator to display debug mpi_info
        HPX_CORE_EXPORT std::ostream& operator<<(
            std::ostream& os, mpi_info const& i);

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors is we enable the error handler
        HPX_CORE_EXPORT extern MPI_Errhandler hpx_mpi_errhandler;

        // function that converts an MPI error into an exception
        HPX_CORE_EXPORT void hpx_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        // we track requests and callbacks in two vectors even though
        // we have the request stored in the request_callback vector already
        // the reason for this is because we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item (using a list)
        HPX_CORE_EXPORT std::vector<MPI_Request>& get_requests_vector();

        // -----------------------------------------------------------------
        // define a lockfree queue type to place requests in prior to handling
        // this is done only to avoid taking a lock every time a request is
        // returned from MPI. Instead the requests are placed into a queue
        // and the polling code pops them prior to calling Testany
        using queue_type = concurrency::ConcurrentQueue<future_data_ptr>;

        // -----------------------------------------------------------------
        // used internally to query how many requests are 'in flight'
        // these are requests that are being polled for actively
        // and not the same as the requests enqueued
        HPX_CORE_EXPORT std::size_t get_number_of_active_requests();

    }    // namespace detail

    // -----------------------------------------------------------------
    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    HPX_CORE_EXPORT void set_error_handler();

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    HPX_CORE_EXPORT hpx::future<void> get_future(MPI_Request request);

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
            int result = f(HPX_FORWARD(Ts, ts)..., &data->request_);
            HPX_UNUSED(result);

            // Add callback after the request has been filled
            data->add_callback();

            // return a future bound to the shared state
            using traits::future_access;
            return future_access<hpx::future<int>>::create(HPX_MOVE(data));
        }
    }    // namespace detail

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::experimental::future ready
    // when found
    HPX_CORE_EXPORT hpx::threads::policies::detail::polling_status poll();

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
            return (detail::get_mpi_info().requests_vector_size_ > 0);
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
            return (detail::get_mpi_info().requests_vector_size_ > 0) || f();
        });
    }

    // initialize the hpx::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    HPX_CORE_EXPORT void init(bool init_mpi = false,
        std::string const& pool_name = "", bool init_errorhandler = false);

    // -----------------------------------------------------------------
    HPX_CORE_EXPORT void finalize(std::string const& pool_name = "");

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is
    // handled elsewhere
    struct [[nodiscard]] enable_user_polling
    {
        enable_user_polling(
            std::string const& pool_name = "", bool init_errorhandler = false)
          : pool_name_(pool_name)
        {
            mpi::experimental::init(false, pool_name, init_errorhandler);
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
        mpi_debug.debug(detail::get_mpi_info(), HPX_FORWARD(Args, args)...);
    }
}}}    // namespace hpx::mpi::experimental
