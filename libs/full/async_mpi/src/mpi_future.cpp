//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/mpi_base/mpi_environment.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

namespace hpx { namespace mpi { namespace experimental {
    namespace detail {

        // extract MPI error message
        std::string error_message(int code)
        {
            int N = 1023;
            std::unique_ptr<char[]> err_buff(new char[std::size_t(N) + 1]);
            err_buff[0] = '\0';

            MPI_Error_string(code, err_buff.get(), &N);

            return err_buff.get();
        }

        // mutex needed to protect mpi request vector, note that the
        // mpi poll function takes place inside the main scheduling loop
        // of hpx and not on an hpx worker thread, so we must use std:mutex
        mutex_type& get_vector_mtx()
        {
            static mutex_type vector_mtx;
            return vector_mtx;
        }

        // an MPI error handling type that we can use to intercept
        // MPI errors if we enable the error handler
        MPI_Errhandler hpx_mpi_errhandler = 0;

        // an instance of mpi_info that we store data in
        mpi_info& get_mpi_info()
        {
            static mpi_info mpi_info_;
            return mpi_info_;
        }

        // stream operator to display debug mpi_info
        HPX_EXPORT std::ostream& operator<<(std::ostream& os, mpi_info const&)
        {
            os << "R " << debug::dec<3>(get_mpi_info().rank_) << "/"
               << debug::dec<3>(get_mpi_info().size_);
            return os;
        }

        // function that converts an MPI error into an exception
        void hpx_MPI_Handler(MPI_Comm*, int* errorcode, ...)
        {
            mpi_debug.debug(debug::str<>("hpx_MPI_Handler"));
            HPX_THROW_EXCEPTION(invalid_status, "hpx_MPI_Handler",
                detail::error_message(*errorcode));
        }

        std::vector<MPI_Request>& get_active_requests()
        {
            static std::vector<MPI_Request> active_requests;
            return active_requests;
        }

        std::vector<future_data_ptr>& get_active_futures()
        {
            static std::vector<future_data_ptr> active_futures;
            return active_futures;
        }

        queue_type& get_request_queue()
        {
            static queue_type request_queue;
            return request_queue;
        }

        // used internally to add an MPI_Request to the lockfree queue
        // that will be used by the polling routines to check when requests
        // have completed
        void add_to_request_queue(future_data_ptr data)
        {
            // place this future data request in our queue for handling
            get_request_queue().enqueue(data);

            mpi_debug.debug(debug::str<>("request queued"), get_mpi_info(),
                "request", debug::hex<8>(data->request_), "active futures",
                debug::dec<3>(get_active_futures().size()));
        }

        // used internally to add a request to the main polling vector
        // that is passed to MPI_Testany
        void add_to_request_vector(future_data_ptr data)
        {
            // this will make a copy and increment the ref count
            get_active_futures().push_back(data);
            get_active_requests().push_back(data->request_);

            mpi_debug.debug(debug::str<>("push_back"), get_mpi_info(),
                "req_ptr", debug::ptr(get_active_requests().data()));

            mpi_debug.debug(debug::str<>("add request"), get_mpi_info(),
                "request", debug::hex<8>(data->request_), "vector size",
                debug::dec<3>(get_active_futures().size()), "non null",
                debug::dec<3>(get_number_of_active_requests()));
        }

        std::size_t get_number_of_enqueued_requests()
        {
            return get_request_queue().size_approx();
        }

        std::size_t get_number_of_active_requests()
        {
            return std::count_if(detail::get_active_requests().begin(),
                detail::get_active_requests().end(),
                [](MPI_Request r) { return r != MPI_REQUEST_NULL; });
        }

#if defined(HPX_DEBUG)
        std::atomic<std::size_t>& get_register_polling_count()
        {
            static std::atomic<std::size_t> register_polling_count{0};
            return register_polling_count;
        }
#endif
    }    // namespace detail

    // return a future object from a user supplied MPI_Request
    hpx::future<void> get_future(MPI_Request request)
    {
        if (request != MPI_REQUEST_NULL)
        {
            HPX_ASSERT_MSG(detail::get_register_polling_count() != 0,
                "MPI event polling has not been enabled on any pool. Make sure "
                "that MPI event polling is enabled on at least one thread "
                "pool.");

            // create a future data shared state with the request Id
            detail::future_data_ptr data(new detail::future_data(
                detail::future_data::init_no_addref{}, request));

            // queue the future state internally for processing
            detail::add_to_request_queue(data);

            // return a future bound to the shared state
            using traits::future_access;
            return future_access<hpx::future<void>>::create(std::move(data));
        }
        return hpx::make_ready_future<void>();
    }

    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    void set_error_handler()
    {
        mpi_debug.debug(debug::str<>("set_error_handler"));

        MPI_Comm_create_errhandler(
            detail::hpx_MPI_Handler, &detail::hpx_mpi_errhandler);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, detail::hpx_mpi_errhandler);
    }

    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::experimental::future
    // ready when found
    void poll()
    {
        std::unique_lock<detail::mutex_type> lk(
            detail::get_vector_mtx(), std::try_to_lock);
        if (!lk.owns_lock())
        {
            if (mpi_debug.is_enabled())
            {
                // for debugging
                static auto poll_deb = mpi_debug.make_timer(1,
                    debug::str<>("Poll - lock failed"), detail::get_mpi_info());

                mpi_debug.timed(poll_deb, "requests",
                    debug::dec<>(detail::get_active_requests().size()),
                    "futures",
                    debug::dec<>(detail::get_active_futures().size()));
            }
            return;
        }

        if (mpi_debug.is_enabled())
        {
            // for debugging
            static auto poll_deb = mpi_debug.make_timer(
                1, debug::str<>("Poll - lock success"), detail::get_mpi_info());

            mpi_debug.timed(poll_deb, "requests",
                debug::dec<>(detail::get_active_requests().size()), "futures",
                debug::dec<>(detail::get_active_futures().size()));
        }

        // have any requests been made that need to be handled?
        // create a future data shared state
        detail::future_data_ptr val;
        while (detail::get_request_queue().try_dequeue(val))
        {
            add_to_request_vector(std::move(val));
        }

        bool keep_trying = !detail::get_active_requests().empty();
        while (keep_trying)
        {
            int index = 0;
            int flag = false;
            MPI_Status status;

            int result = MPI_Testany(
                static_cast<int>(detail::get_active_requests().size()),
                &detail::get_active_requests()[0], &index, &flag, &status);

            if (result == MPI_SUCCESS)
            {
                if (mpi_debug.is_enabled())
                {
                    static auto poll_deb = mpi_debug.make_timer(1,
                        debug::str<>("Poll - success"), detail::get_mpi_info());

                    // clang-format off
                    mpi_debug.timed(poll_deb,
                        debug::str<>("Success"),
                        "index", debug::dec<>(index == MPI_UNDEFINED ? -1 : index),
                        "flag", debug::dec<>(flag),
                        "status", debug::hex(status.MPI_ERROR),
                        "requests", debug::dec<>(detail::get_active_requests().size()),
                        "futures", debug::dec<>(detail::get_active_futures().size()));
                    // clang-format on
                }

                if (index == MPI_UNDEFINED)
                    break;

                keep_trying = flag;
                if (keep_trying)
                {
                    auto req =
                        detail::get_active_requests()[std::size_t(index)];

                    mpi_debug.debug(debug::str<>("MPI_Testany(set)"),
                        detail::get_mpi_info(), "request", debug::hex<8>(req),
                        "vector size",
                        debug::dec<3>(detail::get_active_futures().size()),
                        "non null",
                        debug::dec<3>(detail::get_number_of_active_requests()));

                    // mark the future as ready by setting the shared_state
                    detail::get_active_futures()[std::size_t(index)]->set_data(
                        MPI_SUCCESS);

                    // remove the request from our vector to prevent retesting
                    detail::get_active_requests()[std::size_t(index)] =
                        MPI_REQUEST_NULL;

                    detail::get_active_futures()[std::size_t(index)] = nullptr;
                }
            }
            else
            {
                keep_trying = false;

                if (mpi_debug.is_enabled())
                {
                    auto poll_deb = mpi_debug.make_timer(1,
                        debug::str<>("Poll - <ERR>"), detail::get_mpi_info());

                    // clang-format off
                    mpi_debug.error(poll_deb,
                        debug::str<>("Poll <ERR>"),
                        "MPI_ERROR", detail::error_message(status.MPI_ERROR),
                        "status", debug::dec<>(status.MPI_ERROR),
                        "index", debug::dec<>(index),
                        "flag", debug::dec<>(flag));
                    // clang-format on
                }
            }
        }

        // if there are more than 25% NULL request handles in our vector,
        // compact them
        if (!detail::get_active_futures().empty())
        {
            std::size_t nulls =
                std::count(detail::get_active_requests().begin(),
                    detail::get_active_requests().end(), MPI_REQUEST_NULL);

            if (nulls > detail::get_active_requests().size() / 4)
            {
                // compact away any requests that have been replaced by
                // MPI_REQUEST_NULL
                auto end1 = std::remove(detail::get_active_requests().begin(),
                    detail::get_active_requests().end(), MPI_REQUEST_NULL);
                detail::get_active_requests().resize(
                    std::distance(detail::get_active_requests().begin(), end1));

                // compact away any null pointers in futures vector
                auto end2 = std::remove(detail::get_active_futures().begin(),
                    detail::get_active_futures().end(), nullptr);
                detail::get_active_futures().resize(
                    std::distance(detail::get_active_futures().begin(), end2));

                if (detail::get_active_requests().size() !=
                    detail::get_active_futures().size())
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "hpx::mpi::experimental::poll",
                        "Fatal Error: Mismatch in vectors");
                }

                mpi_debug.debug(debug::str<>("MPI_REQUEST_NULL"),
                    detail::get_mpi_info(), "vector size",
                    debug::dec<3>(detail::get_active_futures().size()),
                    "nulls ", debug::dec<>(nulls));
            }
        }
    }

    namespace detail {
        // -------------------------------------------------------------
        void register_polling(hpx::threads::thread_pool_base& pool)
        {
#if defined(HPX_DEBUG)
            ++get_register_polling_count();
#endif
            mpi_debug.debug(debug::str<>("enable polling"));
            auto* sched = pool.get_scheduler();
            sched->set_mpi_polling_function(&hpx::mpi::experimental::poll);
        }

        // -------------------------------------------------------------
        void unregister_polling(hpx::threads::thread_pool_base& pool)
        {
#if defined(HPX_DEBUG)
            {
                std::unique_lock<hpx::mpi::experimental::detail::mutex_type> lk(
                    detail::get_vector_mtx());
                bool active_requests_empty = get_active_requests().empty();
                bool active_futures_empty = get_active_futures().empty();
                lk.unlock();
                HPX_ASSERT_MSG(active_requests_empty,
                    "MPI request polling was disabled while there are "
                    "unprocessed MPI requests. Make sure MPI request polling "
                    "is not disabled too early.");
                HPX_ASSERT_MSG(active_futures_empty,
                    "MPI request polling was disabled while there are active "
                    "MPI futures. Make sure MPI request polling is not "
                    "disabled too early.");
            }
#endif
            mpi_debug.debug(debug::str<>("disable polling"));
            auto* sched = pool.get_scheduler();
            sched->clear_mpi_polling_function();
        }
    }    // namespace detail

    // initialize the hpx::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    void init(
        bool init_mpi, std::string const& pool_name, bool init_errorhandler)
    {
        if (init_mpi)
        {
            int required = MPI_THREAD_MULTIPLE;
            int minimal = MPI_THREAD_FUNNELED;
            int provided;
            hpx::util::mpi_environment::init(
                nullptr, nullptr, required, minimal, provided);
            if (provided < MPI_THREAD_FUNNELED)
            {
                mpi_debug.error(debug::str<>("hpx::mpi::experimental::init"),
                    "init failed");
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::mpi::experimental::init",
                    "the MPI installation doesn't allow multiple threads");
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::get_mpi_info().rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::get_mpi_info().size_);
        }
        else
        {
            // Check if MPI_Init has been called previously
            if (detail::get_mpi_info().size_ == -1)
            {
                int is_initialized = 0;
                MPI_Initialized(&is_initialized);
                if (is_initialized)
                {
                    MPI_Comm_rank(
                        MPI_COMM_WORLD, &detail::get_mpi_info().rank_);
                    MPI_Comm_size(
                        MPI_COMM_WORLD, &detail::get_mpi_info().size_);
                }
            }
        }

        mpi_debug.debug(debug::str<>("hpx::mpi::experimental::init"),
            detail::get_mpi_info());

        if (init_errorhandler)
        {
            set_error_handler();
            detail::get_mpi_info().error_handler_initialized_ = true;
        }

        // install polling loop on requested thread pool
        if (pool_name.empty())
        {
            detail::register_polling(hpx::resource::get_thread_pool(0));
        }
        else
        {
            detail::register_polling(hpx::resource::get_thread_pool(pool_name));
        }
    }

    // -----------------------------------------------------------------

    void finalize(std::string const& pool_name)
    {
        if (detail::get_mpi_info().error_handler_initialized_)
        {
            HPX_ASSERT(detail::hpx_mpi_errhandler != 0);
            detail::get_mpi_info().error_handler_initialized_ = false;
            MPI_Errhandler_free(&detail::hpx_mpi_errhandler);
            detail::hpx_mpi_errhandler = 0;
        }

        // clean up if we initialized mpi
        hpx::util::mpi_environment::finalize();

        mpi_debug.debug(debug::str<>("Clearing mode"), detail::get_mpi_info(),
            "disable_user_polling");

        if (pool_name.empty())
        {
            detail::unregister_polling(hpx::resource::get_thread_pool(0));
        }
        else
        {
            detail::unregister_polling(
                hpx::resource::get_thread_pool(pool_name));
        }
    }
}}}    // namespace hpx::mpi::experimental
