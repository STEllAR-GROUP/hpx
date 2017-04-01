//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/logging.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/thread/mutex.hpp>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    io_service_pool::io_service_pool(std::size_t pool_size,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread,
            char const* pool_name, char const* name_postfix)
      : next_io_service_(0), stopped_(false),
        pool_size_(pool_size),
        on_start_thread_(on_start_thread), on_stop_thread_(on_stop_thread),
        pool_name_(pool_name), pool_name_postfix_(name_postfix)
    {
        LPROGRESS_ << pool_name;

        if (pool_size == 0)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "io_service_pool::io_service_pool",
                "io_service_pool size is 0");
            return;
        }

        // Give all the io_services work to do so that their run() functions
        // will not exit until they are explicitly stopped.
        for (std::size_t i = 0; i < pool_size; ++i)
        {
            io_services_.emplace_back(new boost::asio::io_service);
            work_.emplace_back(initialize_work(*io_services_[i]));
        }
    }

    io_service_pool::io_service_pool(
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread,
            char const* pool_name, char const* name_postfix)
      : next_io_service_(0), stopped_(false),
        pool_size_(2),
        on_start_thread_(on_start_thread), on_stop_thread_(on_stop_thread),
        pool_name_(pool_name), pool_name_postfix_(name_postfix)
    {
        LPROGRESS_ << pool_name;

        for (std::size_t i = 0; i < pool_size_; ++i)
        {
            io_services_.emplace_back(new boost::asio::io_service);
            work_.emplace_back(initialize_work(*io_services_[i]));
        }
    }

    io_service_pool::~io_service_pool()
    {
        std::lock_guard<boost::mutex> l(mtx_);
        stop_locked();
        join_locked();
        clear_locked();
    }

    void io_service_pool::thread_run(std::size_t index)
    {
        if (on_start_thread_)
            on_start_thread_(index, pool_name_postfix_);

        // use this thread for the given io service
        io_services_[index]->run();   // run io service

        if (on_stop_thread_)
            on_stop_thread_();
    }

    bool io_service_pool::run(bool join_threads)
    {
        std::lock_guard<boost::mutex> l(mtx_);

        // Create a pool of threads to run all of the io_services.
        if (!threads_.empty())   // should be called only once
        {
            HPX_ASSERT(pool_size_ == io_services_.size());
            HPX_ASSERT(threads_.size() == io_services_.size());
            HPX_ASSERT(work_.size() == io_services_.size());

            if (join_threads)
                join_locked();

            return false;
        }

        // Give all the io_services work to do so that their run() functions
        // will not exit until they are explicitly stopped.
        if (!io_services_.empty())
            clear_locked();

        if (io_services_.empty())
        {
            for (std::size_t i = 0; i < pool_size_; ++i)
            {
                io_services_.emplace_back(new boost::asio::io_service);
                work_.emplace_back(initialize_work(*io_services_[i]));
            }
        }

        for (std::size_t i = 0; i < pool_size_; ++i)
        {
            compat::thread thread(util::bind(
                        &io_service_pool::thread_run, this, i));
            threads_.emplace_back(std::move(thread));
        }

        next_io_service_ = 0;
        stopped_ = false;

        HPX_ASSERT(pool_size_ == io_services_.size());
        HPX_ASSERT(threads_.size() == io_services_.size());
        HPX_ASSERT(work_.size() == io_services_.size());

        if (join_threads)
            join_locked();

        return true;
    }

    void io_service_pool::join()
    {
        std::lock_guard<boost::mutex> l(mtx_);
        join_locked();
    }

    void io_service_pool::join_locked()
    {
        // Wait for all threads in the pool to exit.
        for (std::size_t i = 0; i < threads_.size(); ++i)
            threads_[i].join();
        threads_.clear();
    }

    void io_service_pool::stop()
    {
        std::lock_guard<boost::mutex> l(mtx_);
        stop_locked();
    }

    void io_service_pool::stop_locked()
    {
        if (!stopped_) {
            // Explicitly inform all work to exit.
            work_.clear();

            // Explicitly stop all io_services.
            for (std::size_t i = 0; !stopped_ && i < io_services_.size(); ++i)
                io_services_[i]->stop();

            stopped_ = true;
        }
    }

    void io_service_pool::clear()
    {
        std::lock_guard<boost::mutex> l(mtx_);
        clear_locked();
    }

    void io_service_pool::clear_locked()
    {
        if (stopped_) {
            next_io_service_ = 0;
            threads_.clear();
            work_.clear();
            io_services_.clear();
        }
    }

    bool io_service_pool::stopped()
    {
        std::lock_guard<boost::mutex> l(mtx_);
        return stopped_;
    }

    boost::asio::io_service& io_service_pool::get_io_service(int index)
    {
        // use this function for single group io_service pools only
        std::lock_guard<boost::mutex> l(mtx_);

        if (index == -1) {
            if (++next_io_service_ == pool_size_)
                next_io_service_ = 0;

            // Use a round-robin scheme to choose the next io_service to use.
            index = static_cast<int>(next_io_service_);
        }
        else {
            next_io_service_ = static_cast<std::size_t>(index);
        }

        return *io_services_[index]; //-V108
    }
}}  // namespace hpx::util

