//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Caveats:
// 1. pxfs callback functions will be executed on pxfs threads, so avoid hpx
//    calls, make it as simple as possible. Otherwise will trigger the 
//    "Runtime is not available, reporting error locally." error.
//    Put all operations inside the set_value function.
// 2. When using intrusive_ptr and shard_ptr, do not use reference, always copy
//    value.
// 3. Need to use Boost 1.56.0 or above for the intrusive_ptr detach() API.
// 4. Need to use pxfs patched version of OrangeFS at 
//    https://github.com/STEllAR-GROUP/hpxfs/tree/master/
//    orangefs-stable-with-asyncio-changes

#if !defined(HPX_COMPONENTS_IO_PXFS_FILE_HPP_SEP_11_2014_0550PM)
#define HPX_COMPONENTS_IO_PXFS_FILE_HPP_SEP_11_2014_0550PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/include/runtime.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/intrusive_ptr.hpp>

/* ------------------------  added pvfs header stuff --------------- */

#ifdef __cplusplus
extern "C" {
#endif

#include <pxfs.h>

#ifdef __cplusplus
} //extern "C"
#endif

/* -------------------------  end pvfs header stuff --------------- */

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace io
{
    ///////////////////////////////////////////////////////////////////////////
    struct registration_wrapper
    {
        registration_wrapper(hpx::runtime* rt, char const* name)
            : rt_(rt), requires_deregistration_(false)
        {
            // Register this thread with HPX, this should be done once for
            // each external OS-thread intended to invoke HPX functionality.
            // Calling this function more than once will silently fail (will
            // return false).
            requires_deregistration_ = rt_->register_thread(name);
        }
        ~registration_wrapper()
        {
            // Unregister the thread from HPX, this should be done once in the
            // end before the external thread exists.
            if (requires_deregistration_) rt_->unregister_thread();
        }

        hpx::runtime* rt_;
        bool requires_deregistration_;
    };

    struct general_data
    {
        lcos::local::promise<int> p_;
        boost::atomic<std::size_t> count_;
        hpx::runtime *rt_p_;

        general_data(hpx::runtime* rt_p) :
            count_(0), rt_p_(rt_p) {}

        lcos::future<int> get_future()
        {
            return p_.get_future();
        }

        friend void intrusive_ptr_add_ref(general_data* p)
        {
            ++p->count_;
        }

        friend void intrusive_ptr_release(general_data* p)
        {
            if (0 == --p->count_)
            {
                delete p;
            }
        }
    };

    struct read_data
    {
        lcos::local::promise<std::vector<char> > p_;
        std::vector<char> buf_;
        ssize_t len_;

        boost::atomic<std::size_t> count_;
        hpx::runtime *rt_p_;

        read_data(hpx::runtime* rt_p) :
            len_(0), count_(0), rt_p_(rt_p) {}

        lcos::future<std::vector<char> > get_future()
        {
            return p_.get_future();
        }

        friend void intrusive_ptr_add_ref(read_data* p)
        {
            ++p->count_;
        }

        friend void intrusive_ptr_release(read_data* p)
        {
            if (0 == --p->count_)
            {
                delete p;
            }
        }
    };

    struct write_data
    {
        lcos::local::promise<ssize_t> p_;
        ssize_t len_;

        boost::atomic<std::size_t> count_;
        hpx::runtime *rt_p_;

        write_data(hpx::runtime* rt_p) :
            len_(0), count_(0), rt_p_(rt_p) {}

        lcos::future<ssize_t> get_future()
        {
            return p_.get_future();
        }

        friend void intrusive_ptr_add_ref(write_data* p)
        {
            ++p->count_;
        }

        friend void intrusive_ptr_release(write_data* p)
        {
            if (0 == --p->count_)
            {
                delete p;
            }
        }
    };

    struct lseek_data
    {
        lcos::local::promise<off_t> p_;
        off_t offset_;

        boost::atomic<std::size_t> count_;
        hpx::runtime *rt_p_;

        lseek_data(hpx::runtime* rt_p) :
            offset_(-1), count_(0), rt_p_(rt_p) {}

        lcos::future<off_t> get_future()
        {
            return p_.get_future();
        }

        friend void intrusive_ptr_add_ref(lseek_data* p)
        {
            ++p->count_;
        }

        friend void intrusive_ptr_release(lseek_data* p)
        {
            if (0 == --p->count_)
            {
                delete p;
            }
        }
    };

    void set_int_value(hpx::lcos::local::promise<int>& p,
                int const result)
    {
        // notify the waiting HPX thread and return a value
        p.set_value(result);
    }

    void set_char_vector_value(
            boost::intrusive_ptr<read_data> p)
    {
        // format result
        std::vector<char> result(p->buf_.begin(),
                p->buf_.begin() + p->len_);
        // notify the waiting HPX thread and return a value
        p->p_.set_value(result);
    }

    void set_ssize_t_value(
            hpx::lcos::local::promise<ssize_t>& p,
                ssize_t const result)
    {
        // notify the waiting HPX thread and return a value
        p.set_value(result);
    }

    void set_off_t_value(
            hpx::lcos::local::promise<off_t>& p,
                off_t const result)
    {
        // notify the waiting HPX thread and return a value
        p.set_value(result);
    }

    int set_read_promise_cb(void *cdat, int status)
    {
        boost::intrusive_ptr<read_data> p(
                reinterpret_cast<read_data*>(cdat));

        // create a unique thread name based on UTC clock
        boost::posix_time::ptime now =
            boost::posix_time::microsec_clock::universal_time();
        std::string thread_name = std::string("pxfs_read_") +
            boost::posix_time::to_iso_string(now);

        // register this thread in order to be able to
        // call HPX functionality
        registration_wrapper wrap(p->rt_p_, thread_name.c_str());

        // Create an HPX thread to guarantee that the promise::set_value
        // function can be invoked safely.
        hpx::threads::register_thread(hpx::util::bind(
                    &set_char_vector_value, p));

        return status;
    }

    int set_write_promise_cb(void *cdat, int status)
    {
        boost::intrusive_ptr<write_data> p(
                reinterpret_cast<write_data*>(cdat));

        // create a unique thread name based on UTC clock
        boost::posix_time::ptime now =
            boost::posix_time::microsec_clock::universal_time();
        std::string thread_name = std::string("pxfs_write_") +
            boost::posix_time::to_iso_string(now);

        // register this thread in order to be able to
        // call HPX functionality
        registration_wrapper wrap(p->rt_p_, thread_name.c_str());

        // Create an HPX thread to guarantee that the promise::set_value
        // function can be invoked safely.
        hpx::threads::register_thread(hpx::util::bind(
                    &set_ssize_t_value,
                    boost::ref(p->p_), p->len_));

        return status;
    }

    int set_lseek_promise_cb(void *cdat, int status)
    {
        boost::intrusive_ptr<lseek_data> p(
                reinterpret_cast<lseek_data*>(cdat));

        // create a unique thread name based on UTC clock
        boost::posix_time::ptime now =
            boost::posix_time::microsec_clock::universal_time();
        std::string thread_name = std::string("pxfs_lseek_") +
            boost::posix_time::to_iso_string(now);

        // register this thread in order to be able to
        // call HPX functionality
        registration_wrapper wrap(p->rt_p_, thread_name.c_str());

        // Create an HPX thread to guarantee that the promise::set_value
        // function can be invoked safely.
        hpx::threads::register_thread(hpx::util::bind(
                    &set_off_t_value,
                    boost::ref(p->p_), p->offset_));

        return status;
    }

    int set_promise_cb(void *cdat, int status)
    {
        boost::intrusive_ptr<general_data> p(
                reinterpret_cast<general_data*>(cdat));

        // create a unique thread name based on UTC clock
        boost::posix_time::ptime now =
            boost::posix_time::microsec_clock::universal_time();
        std::string thread_name = std::string("pxfs_") +
            boost::posix_time::to_iso_string(now);

        // register this thread in order to be able to
        // call HPX functionality
        registration_wrapper wrap(p->rt_p_, thread_name.c_str());

        // Create an HPX thread to guarantee that the promise::set_value
        // function can be invoked safely.
        hpx::threads::register_thread(hpx::util::bind(
                    &set_int_value, boost::ref(p->p_), status));

        return status;
    }

    ///////////////////////////////////////////////////////////////////////////
    class pxfs_file
    {
    private:

    public:
        pxfs_file() : fd_(-1)
        {
            file_name_.clear();
            rt_p_ = hpx::get_runtime_ptr();
        }

        ~pxfs_file()
        {
            close();
        }

        int open_sync(std::string const& name, int const flag)
        {
            return open(name, flag).get();
        }

        lcos::future<int> open(std::string const& name, int const flag)
        {
            boost::intrusive_ptr<general_data> od_p(new general_data(rt_p_));
            {
                // Get a reference to one of the IO specific HPX io_service objects ...
                hpx::threads::executors::io_pool_executor scheduler;

                // ... and schedule the handler to run on one of its OS-threads.
                scheduler.add(hpx::util::bind(&pxfs_file::open_work, this,
                            boost::ref(name), flag, boost::ref(od_p)));

                // Note that the destructor of the scheduler object will wait for
                // the scheduled task to finish executing.
            }
            return od_p.detach()->get_future();
        }

        void open_work(std::string const& name, int const flag,
                boost::intrusive_ptr<general_data>& p)
        {
            if (fd_ >= 0)
            {
                close_sync();
            }
            file_name_ = name;
            if (flag & O_CREAT)
            {
                pxfs_open(file_name_.c_str(), flag, &fd_,
                        &set_promise_cb, p.get(), 0644);
            } else
            {
                pxfs_open(file_name_.c_str(), flag, &fd_,
                        &set_promise_cb, p.get());
            }
        }

        bool is_open()
        {
            return fd_ >= 0;
        }

        bool is_open_sync()
        {
            return is_open();
        }

        int close_sync()
        {
            return close().get();
        }

        lcos::future<int> close()
        {
            boost::intrusive_ptr<general_data> gd_p(new general_data(rt_p_));
            {
                // Get a reference to one of the IO specific HPX io_service objects ...
                hpx::threads::executors::io_pool_executor scheduler;

                // ... and schedule the handler to run on one of its OS-threads.
                scheduler.add(hpx::util::bind(&pxfs_file::close_work,
                            this, boost::ref(gd_p)));

                // Note that the destructor of the scheduler object will wait for
                // the scheduled task to finish executing.
            }
            return gd_p.detach()->get_future();
        }

        void close_work(boost::intrusive_ptr<general_data>& p)
        {
            file_name_.clear();
            if (fd_ >= 0)
            {
                pxfs_close(fd_, &set_promise_cb, p.get());
                fd_ = -1;
            } else
            {
                p->p_.set_value(0);
            }
        }

        int remove_file_sync(std::string const& file_name)
        {
            return remove_file(file_name).get();
        }

        lcos::future<int> remove_file(std::string const& file_name)
        {
            boost::intrusive_ptr<general_data> gd_p(new general_data(rt_p_));
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&pxfs_file::remove_file_work,
                            this, boost::ref(file_name), boost::ref(gd_p)));
            }
            return gd_p.detach()->get_future();
        }

        void remove_file_work(std::string const& file_name,
                boost::intrusive_ptr<general_data>& p)
        {
            pxfs_unlink(file_name.c_str(), &set_promise_cb, p.get());
        }

        std::vector<char> read_sync(size_t const count)
        {
            return read(count).get();
        }

        lcos::future<std::vector<char> > read(size_t const count)
        {
            boost::intrusive_ptr<read_data> rd_p(new read_data(rt_p_));
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&pxfs_file::read_work,
                            this, count, rd_p));
            }
            return rd_p.detach()->get_future();
        }

        void read_work(size_t const count,
                boost::intrusive_ptr<read_data> p)
        {
            std::vector<char> empty_vector;
            if (fd_ < 0 || count <= 0)
            {
                p->p_.set_value(empty_vector);
                return;
            }

            p->buf_.assign(count, 0);
            pxfs_read(fd_, p->buf_.data(), count, &p->len_,
                    &set_read_promise_cb, p.get());
        }

        std::vector<char> pread_sync(size_t const count, off_t const offset)
        {
            return pread(count, offset).get();
        }

        lcos::future<std::vector<char> > pread(ssize_t const count,
                off_t const offset)
        {
            boost::intrusive_ptr<read_data> rd_p(new read_data(rt_p_));
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&pxfs_file::pread_work,
                            this, count, offset, rd_p));
            }
            return rd_p.detach()->get_future();
        }

        void pread_work(size_t const count, off_t const offset,
                boost::intrusive_ptr<read_data> p)
        {
            std::vector<char> empty_vector;
            if (fd_ < 0 || count <= 0 || offset < 0)
            {
                p->p_.set_value(empty_vector);
                return;
            }

            p->buf_.assign(count, 0);
            pxfs_pread(fd_, p->buf_.data(), count, offset,
                    &p->len_,
                    &set_read_promise_cb, p.get());
        }

        ssize_t write_sync(std::vector<char> const& buf)
        {
            return write(buf).get();
        }

        lcos::future<ssize_t> write(std::vector<char> const& buf)
        {
            boost::intrusive_ptr<write_data> wd_p(new write_data(rt_p_));
            {
                // Get a reference to one of the IO specific HPX io_service objects ...
                hpx::threads::executors::io_pool_executor scheduler;

                // ... and schedule the handler to run on one of its OS-threads.
                scheduler.add(hpx::util::bind(&pxfs_file::write_work,
                            this, boost::ref(buf), wd_p));

                // Note that the destructor of the scheduler object will wait for
                // the scheduled task to finish executing.
            }
            return wd_p.detach()->get_future();
        }

        void write_work(std::vector<char> const& buf,
                boost::intrusive_ptr<write_data> p)
        {
            if (fd_ < 0 || buf.empty())
            {
                p->p_.set_value(0);
                return;
            }
            pxfs_write(fd_, buf.data(), buf.size(), &p->len_,
                    &set_write_promise_cb, p.get());
        }

        ssize_t pwrite_sync(std::vector<char> const& buf, off_t const offset)
        {
            return pwrite(buf, offset).get();
        }


        lcos::future<ssize_t> pwrite(std::vector<char> const& buf,
                off_t const offset)
        {
            boost::intrusive_ptr<write_data> wd_p(new write_data(rt_p_));
            {
                // Get a reference to one of the IO specific HPX io_service objects ...
                hpx::threads::executors::io_pool_executor scheduler;

                // ... and schedule the handler to run on one of its OS-threads.
                scheduler.add(hpx::util::bind(&pxfs_file::pwrite_work,
                            this, boost::ref(buf), offset, wd_p));

                // Note that the destructor of the scheduler object will wait for
                // the scheduled task to finish executing.
            }
            return wd_p.detach()->get_future();
        }

        void pwrite_work(std::vector<char> const& buf, off_t const offset,
                boost::intrusive_ptr<write_data> p)
        {
            if (fd_ < 0 || buf.empty() || offset < 0)
            {
                p->p_.set_value(0);
                return;
            }
            pxfs_pwrite(fd_, buf.data(), buf.size(), offset, &p->len_,
                    &set_write_promise_cb, p.get());
        }

        off_t lseek_sync(off_t const offset, int const whence)
        {
            return lseek(offset, whence).get();
        }

        lcos::future<off_t> lseek(off_t const offset, int const whence)
        {
            boost::intrusive_ptr<lseek_data> ld_p(new lseek_data(rt_p_));
            {
                // Get a reference to one of the IO specific HPX io_service objects ...
                hpx::threads::executors::io_pool_executor scheduler;

                // ... and schedule the handler to run on one of its OS-threads.
                scheduler.add(hpx::util::bind(&pxfs_file::lseek_work,
                            this, offset, whence, ld_p));

                // Note that the destructor of the scheduler object will wait for
                // the scheduled task to finish executing.
            }
            return ld_p.detach()->get_future();
        }

        void lseek_work(off_t const offset, int const whence,
                boost::intrusive_ptr<lseek_data> p)
        {
            if (fd_ < 0)
            {
                p->p_.set_value(-1);
                return;
            }
            pxfs_lseek(fd_, offset, whence, &p->offset_,
                    &set_lseek_promise_cb, p.get());
        }

      private:
        // PVFS2TAB_FILE env need to be set in the shell
        int fd_;
        std::string file_name_;
        hpx::runtime *rt_p_;
    };

}} // hpx::io


#endif
