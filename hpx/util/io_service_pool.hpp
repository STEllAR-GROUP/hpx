//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_IO_SERVICE_POOL_MAR_26_2008_1218PM)
#define HPX_UTIL_IO_SERVICE_POOL_MAR_26_2008_1218PM

#include <vector>

#include <boost/asio/io_service.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <hpx/config.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    /// A pool of io_service objects.
    class HPX_EXPORT io_service_pool : private boost::noncopyable
    {
    public:
        /// \brief Construct the io_service pool.
        /// \param pool_size
        ///                 [in] The number of threads to run to serve incoming
        ///                 requests
        /// \param start_thread
        ///                 [in]
        explicit io_service_pool(std::size_t pool_size = 2,
            HPX_STD_FUNCTION<void(std::size_t)> on_start_thread = HPX_STD_FUNCTION<void(std::size_t)>(),
            HPX_STD_FUNCTION<void()> on_stop_thread = HPX_STD_FUNCTION<void()>(),
            char const* pool_name = "");

        /// \brief Construct the io_service pool.
        /// \param start_thread
        ///                 [in]
        explicit io_service_pool(HPX_STD_FUNCTION<void(std::size_t)> on_start_thread,
            HPX_STD_FUNCTION<void()> on_stop_thread = HPX_STD_FUNCTION<void()>(),
            char const* pool_name = "");

        ~io_service_pool();

        /// \brief Run all io_service objects in the pool. If join_threads is true
        ///        this will also wait for all threads to complete
        bool run(bool join_threads = true);

        /// \brief Stop all io_service objects in the pool.
        void stop();

        /// \brief Join all io_service threads in the pool.
        void join();

        /// \brief Clear all internal data structures
        void clear();

        /// \brief Get an io_service to use.
        boost::asio::io_service& get_io_service(int index = -1);

        /// \brief Get number of threads associated with this I/O service.
        std::size_t size() const { return pool_size_; }

    protected:
        ///
        void thread_run(std::size_t index);

        void stop_locked();
        void join_locked();
        void clear_locked();

    private:
        typedef boost::shared_ptr<boost::asio::io_service> io_service_ptr;
        typedef boost::shared_ptr<boost::asio::io_service::work> work_ptr;

        boost::mutex mtx_;

        /// The pool of io_services.
        std::vector<io_service_ptr> io_services_;
        std::vector<boost::shared_ptr<boost::thread> > threads_;

        /// The work that keeps the io_services running.
        std::vector<work_ptr> work_;

        /// The next io_service to use for a connection.
        std::size_t next_io_service_;

        /// set to true if stopped
        bool stopped_;

        /// initial number of OS threads to execute in this pool
        std::size_t const pool_size_;

        /// call this for each thread start/stop
        HPX_STD_FUNCTION<void(std::size_t)> on_start_thread_;
        HPX_STD_FUNCTION<void()> on_stop_thread_;

#if defined(DEBUG)
        char const* pool_name_;
#endif
    };

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif
