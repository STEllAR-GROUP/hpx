//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_IO_SERVER_ORANGEFS_FILE_HPP_SEP_01_2014_1223PM)
#define HPX_COMPONENTS_IO_SERVER_ORANGEFS_FILE_HPP_SEP_01_2014_1223PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/util/locality_result.hpp>

/* ------------------------  added pvfs header stuff --------------- */

#ifdef __cplusplus
extern "C" {
#endif

#include <pvfs2-usrint.h>
#include <pvfs2.h>

#ifdef __cplusplus
} //extern "C" {
#endif

/* -------------------------  end pvfs header stuff --------------- */

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace io { namespace server
{

    class orangefs_file : public components::locking_hook<
                          components::managed_component_base<orangefs_file> >
    {
      public:
        orangefs_file() : fd_(-1)
        {
            file_name_.clear();
        }

        ~orangefs_file()
        {
            close();
        }

        void open(std::string const& name, int const flag)
        {
            // Get a reference to one of the IO specific HPX io_service objects ...
            hpx::threads::executors::io_pool_executor scheduler;

            // ... and schedule the handler to run on one of its OS-threads.
            scheduler.add(hpx::util::bind(&orangefs_file::open_work, this,
                        boost::ref(name), flag));

            // Note that the destructor of the scheduler object will wait for
            // the scheduled task to finish executing.
        }

        void open_work(std::string const& name, int const flag)
        {
            if (fd_ >= 0)
            {
                close();
            }
            if (flag & O_CREAT)
            {
                fd_ = pvfs_open(name.c_str(), flag, 0644);
            } else
            {
                fd_ = pvfs_open(name.c_str(), flag);
            }
            file_name_ = name;
        }

        bool is_open() const
        {
            return fd_ >= 0;
        }

        void close()
        {
            // Get a reference to one of the IO specific HPX io_service objects ...
            hpx::threads::executors::io_pool_executor scheduler;

            // ... and schedule the handler to run on one of its OS-threads.
            scheduler.add(hpx::util::bind(&orangefs_file::close_work, this));

            // Note that the destructor of the scheduler object will wait for
            // the scheduled task to finish executing.
        }

        void close_work()
        {
            if (fd_ >= 0)
            {
                pvfs_close(fd_);
                fd_ = -1;
            }
            file_name_.clear();
        }

        int remove_file(std::string const& file_name)
        {
            int result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::remove_file_work,
                            this, boost::ref(file_name), boost::ref(result)));
            }
            return result;
        }

        void remove_file_work(std::string const& file_name, int& result)
        {
            result = pvfs_unlink(file_name.c_str());
        }

        std::vector<char> read(size_t const count)
        {
            std::vector<char> result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::read_work,
                            this, count, boost::ref(result)));
            }
            return result;
        }

        void read_work(size_t const count, std::vector<char>& result)
        {
            if (fd_ < 0 || count <= 0)
            {
                return;
            }

            std::unique_ptr<char[]> sp(new char[count]);
            ssize_t len = pvfs_read(fd_, sp.get(), count);

            if (len == 0)
            {
                return;
            }

            result.assign(sp.get(), sp.get() + len);
        }

        std::vector<char> pread(size_t const count, off_t const offset)
        {
            std::vector<char> result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::pread_work,
                            this, count, offset, boost::ref(result)));
            }
            return result;
        }

        void pread_work(size_t const count, off_t const offset,
                std::vector<char>& result)
        {
            if (fd_ < 0 || count <= 0 || offset < 0)
            {
                return;
            }

            std::unique_ptr<char[]> sp(new char[count]);
            ssize_t len = pvfs_pread(fd_, sp.get(), count, offset);

            if (len == 0)
            {
                return;
            }

            result.assign(sp.get(), sp.get() + len);
        }

        ssize_t write(std::vector<char> const& buf)
        {
            ssize_t result = 0;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::write_work,
                            this, boost::ref(buf), boost::ref(result)));
            }
            return result;
        }

        void write_work(std::vector<char> const& buf, ssize_t& result)
        {
            if (fd_ < 0 || buf.empty())
            {
                return;
            }
            result = pvfs_write(fd_, buf.data(), buf.size());
        }

        ssize_t pwrite(std::vector<char> const& buf, off_t const offset)
        {
            ssize_t result = 0;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::pwrite_work,
                    this, boost::ref(buf), offset, boost::ref(result)));
            }
            return result;
        }

        void pwrite_work(std::vector<char> const& buf,
                off_t const offset, ssize_t& result)
        {
            if (fd_ < 0 || buf.empty() || offset < 0)
            {
                return;
            }
            result = pvfs_pwrite(fd_, buf.data(), buf.size(), offset);
        }

        off_t lseek(off_t const offset, int const whence)
        {
            off_t result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&orangefs_file::lseek_work,
                    this, offset, whence, boost::ref(result)));
            }
            return result;
        }

        void lseek_work(off_t const offset, int const whence, off_t& result)
        {
            if (fd_ < 0)
            {
                result = -1;
                return;
            }

            result = pvfs_lseek(fd_, offset, whence);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, open);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, is_open);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, close);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, remove_file);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, read);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, pread);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, write);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, pwrite);
        HPX_DEFINE_COMPONENT_ACTION(orangefs_file, lseek);

      private:
        typedef components::managed_component_base<orangefs_file> base_type;

        // PVFS2TAB_FILE env need to be set in the shell
        int fd_;
        std::string file_name_;
    };

}}} // hpx::io::server

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the orangefs_file actions
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::open_action,
        orangefs_file_open_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::is_open_action,
        orangefs_file_is_open_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::close_action,
        orangefs_file_close_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::remove_file_action,
        orangefs_file_remove_file_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::read_action,
        orangefs_file_read_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::pread_action,
        orangefs_file_pread_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::write_action,
        orangefs_file_write_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::pwrite_action,
        orangefs_file_pwrite_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::orangefs_file::lseek_action,
        orangefs_file_lseek_action)

#endif

