//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_MPI_FUTURE_HPP
#define HPX_MPI_FUTURE_HPP

#include <mpi.h>
//
#include <array>
#include <cstdio>
#include <exception>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

#include <hpx/concurrency/concurrentqueue.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/runtime/thread_pool_helpers.hpp>
#include <hpx/threading_base/print.hpp>

// The preferred mode of operation uses a std::vector and MPI_Testany
// but the use of a list and individual tests is supported for legacy
// debug reasons, uncomment one of these #defines to choose which is used
//#define HPX_MPI_LIST_STORAGE
#define HPX_MPI_VECTOR_STORAGE

namespace hpx { namespace mpi {

    using  print_on = debug::enable_print<false>;
    static print_on mpi_debug("MPI_FUT");

    namespace detail {
        // mutex needed to protect mpi request list, note that the
        // mpi poll function takes place inside the main scheduling loop
        // of hpx and not on an hpx worker thread, so we must use std:mutex
        static std::mutex list_mtx_;

        // -----------------------------------------------------------------
        // An implementation of future_data for MPI
        struct future_data : hpx::lcos::detail::future_data<int>
        {
            HPX_NON_COPYABLE(future_data);

            using init_no_addref = typename hpx::lcos::detail::future_data<int>::init_no_addref;

            // default empty constructor
            future_data() = default;

            // constructor that takes a request
            future_data(init_no_addref no_addref, MPI_Request request)
                : hpx::lcos::detail::future_data<int>(no_addref), request_(request)
            {}

            // constructor used for creation directly by invoke
            future_data(init_no_addref no_addref)
                : hpx::lcos::detail::future_data<int>(no_addref)
            {}

            // The native MPI request handle owned by this future data
            MPI_Request request_;
        };

        // -----------------------------------------------------------------
        // intrusive pointer for future_data
        using  future_data_ptr = memory::intrusive_ptr<future_data>;

        // -----------------------------------------------------------------
        // a convenience structure to hold state vars
        // used extensivey with debug::print to display rank etc
        struct mpi_info {
            bool mpi_initialized_ = false;
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
        };

        // an instance of mpi_info that we store data in
        static mpi_info mpi_info_ = {false, false, 0, 0};

        // stream operator to display debug mpi_info
        std::ostream &operator<<(std::ostream &os, const mpi_info &i) {
            os << "R " << debug::dec<3>(mpi_info_.rank_)
               << "/" << debug::dec<3>(mpi_info_.size_);
            return os;
        }

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors is we enable the error handler
        static MPI_Errhandler hpx_mpi_errhandler;

        // function that converts an MPI error into an exception
        void hpx_MPI_Handler(MPI_Comm *, int *, ...)
        {
            mpi_debug.debug(debug::str<>("hpx_MPI_Handler"));
            throw std::runtime_error("MPI error");
        }

        // -----------------------------------------------------------------
        // we track requests and future data in two vectors even though
        // we have the request stored in the future data already
        // the reason for this is because we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item using a list
        // define HPX_MPI_LIST_STORAGE or HPX_MPI_VECTOR_STORAGE
        // to select between different implementations
        // one will be removed in future versions if it underperforms
#ifdef HPX_MPI_LIST_STORAGE
        static std::list<future_data_ptr>   active_futures_;
#else
        static std::vector<MPI_Request>     active_requests_;
        static std::vector<future_data_ptr> active_futures_;
#endif

        // -----------------------------------------------------------------
        // define a lockfree queue type to place requests in prior to handling
        // this is done only to avoid taking a lock every time a request is
        // returned from MPI. Instead the requests are placed into a queue
        // and the polling code pops them prior to calling Testany
        using queue_type = moodycamel::ConcurrentQueue<future_data_ptr>;
        static queue_type request_queue_;

        // -----------------------------------------------------------------
        // used internally to add an MPI_Request to the lockfree queue
        // that will be used by the polling routines to check when requests
        // have completed
        void add_to_request_queue(future_data_ptr data)
        {
            // place this future data request in our queue for handling
            request_queue_.enqueue(data);

            // for debugging only
            if (mpi_debug.is_enabled()) {
                mpi_debug.debug(debug::str<>("request queued")
                    , mpi_info_
                    , "request", debug::hex<8>(data->request_)
                    , "active futures", debug::dec<3>(active_futures_.size()));
            }
        }

        // -----------------------------------------------------------------
        // used internally to add a request to the main polling vector/list
        // that is passed to MPI_Testany
        void add_to_request_list(future_data_ptr data)
        {
            // this will make a copy and increment the ref count
            active_futures_.push_back(data);
#ifdef HPX_MPI_VECTOR_STORAGE
            active_requests_.push_back(data->request_);
            mpi_debug.debug(debug::str<>("push_back")
                , mpi_info_
                , "req_ptr", debug::ptr(active_requests_.data()));
#endif
            // for debugging only
            if (mpi_debug.is_enabled()) {
                mpi_debug.debug(debug::str<>("add request")
                    , mpi_info_
                    , "request", debug::hex<8>(data->request_)
                    , "list size", debug::dec<3>(active_futures_.size()));
            }
        }
    }

    // -----------------------------------------------------------------
    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    void set_error_handler() {
        mpi_debug.debug(debug::str<>("set_error_handler"));
        MPI_Comm_create_errhandler(detail::hpx_MPI_Handler, &detail::hpx_mpi_errhandler);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, detail::hpx_mpi_errhandler);
    }

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    hpx::future<void> get_future(MPI_Request request)
    {
        if (request != MPI_REQUEST_NULL) {
            // create a future data shared state with the request Id
            detail::future_data_ptr data =
                new detail::future_data(detail::future_data::init_no_addref{}, request);
            // queue the future state internally for processing
            detail::add_to_request_queue(data);
            // return a future bound to the shared state
            using traits::future_access;
            return future_access<hpx::future<void>>::create(std::move(data));
        }
        return hpx::make_ready_future<void>();
    }

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
            // queue the future state internally for processing
            detail::add_to_request_queue(data);
            // return a future bound to the shared state
            using traits::future_access;
            return future_access<hpx::future<int>>::create(std::move(data));
        }
    }    // namespace detail

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::future ready
    // when found
    void poll()
    {
        std::unique_lock<std::mutex> lk(detail::list_mtx_, std::try_to_lock);
        if (!lk.owns_lock())
        {
            // for debugging
            static auto poll_deb = mpi_debug.make_timer(1
                , debug::str<>("Poll - lock failed"), detail::mpi_info_);
            mpi_debug.timed(poll_deb
                , "requests", debug::dec<>(detail::active_requests_.size())
                , "futures", debug::dec<>(detail::active_futures_.size()));
            return;
        }

        // for debugging
        static auto poll_deb = mpi_debug.make_timer(1
            , debug::str<>("Poll - lock success"), detail::mpi_info_);

        mpi_debug.timed(poll_deb
            , "requests", debug::dec<>(detail::active_requests_.size())
            , "futures", debug::dec<>(detail::active_futures_.size()));

        // have any requests been made that need to be handled?
        // create a future data shared state
        detail::future_data_ptr val;
        while (detail::request_queue_.try_dequeue(val))
        {
            add_to_request_list(std::move(val));
        }

#ifdef HPX_MPI_LIST_STORAGE
        std::list<detail::future_data_ptr>::iterator i = detail::active_futures_.begin();
         while (i != detail::active_futures_.end())
        {
            MPI_Request request = (*i)->request_;
            int flag;
            MPI_Status status;
            //
            auto req = request;
            MPI_Test(&request, &flag, &status);
            if (flag!=0) {
                mpi_debug.debug(debug::str<>("MPI_Test set_data")
                               , detail::mpi_info_
                               , "request", debug::hex<8>(req)
                               , "list size", debug::dec<3>(active_futures_.size()));

                // mark the future as ready by setting the shared_state
                (*i)->set_data(MPI_SUCCESS /*hpx::util::unused*/);
                i = detail::active_futures_.erase(i);
            }
            else {
                ++i;
                mpi_debug.timed(timer
                               , "no promise"
                               , "request", debug::hex<8>(req)
                               , "list size", debug::dec<3>(active_futures_.size()));
            }
            mpi_debug.timed(timer
                           , "request", debug::hex<8>(req)
                           , "list size", debug::dec<3>(active_futures_.size()));

        }
#else
        bool keep_trying = detail::active_requests_.size()>0;
        while (keep_trying) {
            int        index = 0;
            int        flag  = false;
            MPI_Status status;

            // clang-format off
            int size = static_cast<int>(detail::active_requests_.size());
            int result = MPI_Testany(size,
                            &detail::active_requests_[0],
                            &index,
                            &flag,
                            &status);
            // clang-format on

            if (result == MPI_SUCCESS) {
                // clang-format off
                mpi_debug.timed(poll_deb
                    , debug::str<>("Success")
                    , "index", debug::dec<>((index==MPI_UNDEFINED ? -1 : index))
                    , "flag", debug::dec<>(flag)
                    , "status", debug::hex<>(status.MPI_ERROR)
                    , "requests", debug::dec<>(detail::active_requests_.size())
                    , "futures", debug::dec<>(detail::active_futures_.size()));
                // clang-format on

                if (index==MPI_UNDEFINED) break;
                //
                keep_trying = flag;
                if (keep_trying) {
                    // clang-format off
                    auto req = detail::active_requests_[unsigned(index)];
                    mpi_debug.debug(debug::str<>("MPI_Testany(set)")
                        , detail::mpi_info_
                        , "request", debug::hex<8>(req)
                        , "list size", debug::dec<3>(detail::active_futures_.size()));
                    // clang-format on

                    // mark the future as ready by setting the shared_state
                    detail::active_futures_[unsigned(index)]->set_data(MPI_SUCCESS);
                    // remove the request from our list to prevent retesting
                    detail::active_requests_[unsigned(index)] = MPI_REQUEST_NULL;
                    detail::active_futures_[unsigned(index)]  = nullptr;
                }
            }
            else {
                keep_trying = false;
                int N = 1023;
                char err_buff[1024];
                /*int res = */MPI_Error_string(status.MPI_ERROR, err_buff, &N);
                // clang-format off
                mpi_debug.error(poll_deb
                    , debug::str<>("Poll <ERR>")
                    , "MPI_ERROR", err_buff
                    , "status", debug::dec<>(status.MPI_ERROR)
                    , "index", debug::dec<>(index)
                    , "flag", debug::dec<>(flag));
                // clang-format on
            }
        };
        // if there are more than 25% NULL request handles in our lists, compact them
        if (detail::active_futures_.size()>0) {
            std::size_t nulls = std::count(detail::active_requests_.begin(),
                detail::active_requests_.end(), MPI_REQUEST_NULL);
            if (nulls>detail::active_requests_.size()/4) {
                // compact away any requests that have been replaced by MPI_REQUEST_NULL
                auto end1 = std::remove(detail::active_requests_.begin(), detail::active_requests_.end(), MPI_REQUEST_NULL);
                detail::active_requests_.resize(std::distance(detail::active_requests_.begin(),end1));
                // compact away any null pointers in futures list
                auto end2 = std::remove(detail::active_futures_.begin(), detail::active_futures_.end(), nullptr);
                detail::active_futures_.resize(std::distance(detail::active_futures_.begin(),end2));
                if (detail::active_requests_.size() != detail::active_futures_.size()) {
                    throw std::runtime_error("Fatal Error: Mismatch in vectors");
                }
                mpi_debug.debug(debug::str<>("MPI_REQUEST_NULL")
                    , detail::mpi_info_
                    , "list size", debug::dec<3>(detail::active_futures_.size())
                    , "nulls ", debug::dec<>(nulls));
            }
        }
#endif
    }

    // -----------------------------------------------------------------
    // This is not completely safe as it will return when the request list is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    void wait() {
        hpx::util::yield_while([&]() {
            std::lock_guard<std::mutex> lk(detail::list_mtx_);
            return (detail::active_futures_.size()>0);
        });
    }

    template <typename F>
    void wait(F&& f) {
        hpx::util::yield_while([&]() {
            std::lock_guard<std::mutex> lk(detail::list_mtx_);
            return (detail::active_futures_.size() > 0) && f();
        });
    }

    // -----------------------------------------------------------------
    namespace detail {

        void register_polling(hpx::threads::thread_pool_base& pool)
        {
            auto* sched = pool.get_scheduler();

            mpi_debug.debug(debug::str<>("Setting mode"), detail::mpi_info_,
                "enable_user_polling");

            // always set polling function before enabling polling
            sched->set_user_polling_function(&hpx::mpi::poll);
            sched->add_remove_scheduler_mode(
                threads::policies::enable_user_polling,
                threads::policies::do_background_work);
        }
    }    // namespace detail

    // initialize the hpx::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    void init(bool init_mpi = false, std::string const& pool_name = "",
        bool init_errorhandler = false)
    {
        // Check if MPI_Init has been called previously
        int is_initialized_=0;
        MPI_Initialized(&is_initialized_);
        if (is_initialized_) {
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_info_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_info_.size_);
        }
        else if (init_mpi) {
            int required, provided;
            required = MPI_THREAD_MULTIPLE;
            MPI_Init_thread(0, nullptr, required, &provided);
            if (provided < MPI_THREAD_FUNNELED) {
                mpi_debug.error(debug::str<>("hpx::mpi::init"), "init failed");
                throw std::runtime_error(
                    "Your MPI installation doesn't allow multiple threads");
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_info_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_info_.size_);
            detail::mpi_info_.mpi_initialized_ = true;
        }
        mpi_debug.debug(debug::str<>("hpx::mpi::init"), detail::mpi_info_);

        if (init_errorhandler) {
            set_error_handler();
            detail::mpi_info_.error_handler_initialized_ = true;
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
    namespace detail
    {
        void unregister_polling(hpx::threads::thread_pool_base& pool)
        {
            auto* sched = pool.get_scheduler();
            sched->remove_scheduler_mode(threads::policies::enable_user_polling);
        }
    }    // namespace detail

    void finalize(std::string const& pool_name = "")
    {
        if (detail::mpi_info_.mpi_initialized_) {
            MPI_Finalize();
        }
        if (detail::mpi_info_.error_handler_initialized_) {
            mpi_debug.debug("error handler deletion not implemented");
        }
        //
        mpi_debug.debug(debug::str<>("Clearing mode")
              , detail::mpi_info_, "disable_user_polling");
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

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is
    // handled elsewhere
    struct enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "")
          : pool_name_(pool_name)
        {
            mpi::init(false, pool_name);
        }

        ~enable_user_polling()
        {
            mpi::finalize(pool_name_);
        }

        template <typename F>
        void wait(F&& f)
        {
            return mpi::wait(std::forward<F>(f));
        }

    private:
        std::string pool_name_;
    };

    // -----------------------------------------------------------------
    template <typename... Args>
    void debug(const Args&... args) {
        mpi_debug.debug(detail::mpi_info_, args...);
    }
}}

#endif
