//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_IO_SERVER_LOCAL_FILE_HPP_AUG_27_2014_1200AM)
#define HPX_COMPONENTS_IO_SERVER_LOCAL_FILE_HPP_AUG_27_2014_1200AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <cstdio>
#include <vector>
#include <string>

#if defined(BOOST_MSVC)
#ifdef _WIN64
typedef __int64    ssize_t;
#else
typedef _W64 int   ssize_t;
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace io { namespace server
{
    // local file class
    // uses C style file APIs
    class local_file
      : public components::locking_hook<
            components::managed_component_base<local_file> >
    {
      public:

        local_file()
        {
            fp_ = NULL;
            file_name_.clear();
        }

        ~local_file()
        {
            close();
        }

        void open(std::string const& name, std::string const& mode)
        {
            // Get a reference to one of the IO specific HPX io_service objects ...
            hpx::threads::executors::io_pool_executor scheduler;

            // ... and schedule the handler to run on one of its OS-threads.
            scheduler.add(hpx::util::bind(&local_file::open_work, this,
                        boost::ref(name), boost::ref(mode)));

            // Note that the destructor of the scheduler object will wait for
            // the scheduled task to finish executing.
        }

        void open_work(std::string const& name, std::string const& mode)
        {
            if (fp_ != NULL)
            {
                close();
            }
            fp_ = fopen(name.c_str(), mode.c_str());
            file_name_ = name;
        }

        bool is_open() const
        {
            return fp_ != NULL;
        }

        void close()
        {
            // Get a reference to one of the IO specific HPX io_service objects ...
            hpx::threads::executors::io_pool_executor scheduler;

            // ... and schedule the handler to run on one of its OS-threads.
            scheduler.add(hpx::util::bind(&local_file::close_work, this));

            // Note that the destructor of the scheduler object will wait for
            // the scheduled task to finish executing.
        }

        void close_work()
        {
            if (fp_ != NULL)
            {
                std::fclose(fp_);
                fp_ = NULL;
            }
            file_name_.clear();
        }

        int remove_file(std::string const& file_name)
        {
            int result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&local_file::remove_file_work,
                            this, boost::ref(file_name), boost::ref(result)));
            }
            return result;
        }

        void remove_file_work(std::string const& file_name, int &result)
        {
            result = remove(file_name.c_str());
        }

        std::vector<char> read(size_t const count)
        {
            std::vector<char> result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&local_file::read_work,
                            this, count, boost::ref(result)));
            }
            return result;
        }

        void read_work(size_t const count, std::vector<char>& result)
        {
            if (fp_ == NULL || count <= 0)
            {
                return;
            }

            std::unique_ptr<char> sp(new char[count]);
            ssize_t len = fread(sp.get(), 1, count, fp_);

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
                scheduler.add(hpx::util::bind(&local_file::pread_work,
                            this, count, offset, boost::ref(result)));
            }
            return result;
        }

        void pread_work(size_t const count, off_t const offset,
                std::vector<char>& result)
        {
            if (fp_ == NULL || count <= 0 || offset < 0)
            {
                return;
            }

            fpos_t pos;
            if (fgetpos(fp_, &pos) != 0)
            {
                return;
            }

            if (fseek(fp_, offset, SEEK_SET) != 0)
            {
                fsetpos(fp_, &pos);
                return;
            }

            read_work(count, result);
            fsetpos(fp_, &pos);
        }

        ssize_t write(std::vector<char> const& buf)
        {
            ssize_t result = 0;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&local_file::write_work,
                            this, boost::ref(buf), boost::ref(result)));
            }
            return result;
        }

        void write_work(std::vector<char> const& buf, ssize_t& result)
        {
            if (fp_ == NULL || buf.empty())
            {
                return;
            }
            result = fwrite(buf.data(), 1, buf.size(), fp_);
        }

        ssize_t pwrite(std::vector<char> const& buf, off_t const offset)
        {
            ssize_t result = 0;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&local_file::pwrite_work,
                    this, boost::ref(buf), offset, boost::ref(result)));
            }
            return result;
        }

        void pwrite_work(std::vector<char> const& buf,
                off_t const offset, ssize_t& result)
        {
            if (fp_ == NULL || buf.empty() || offset < 0)
            {
                return;
            }

            fpos_t pos;
            if (fgetpos(fp_, &pos) != 0)
            {
                return;
            }

            if (fseek(fp_, offset, SEEK_SET) != 0)
            {
                fsetpos(fp_, &pos);
                return;
            }

            write_work(buf, result);
            fsetpos(fp_, &pos);
        }

        int lseek(off_t const offset, int const whence)
        {
            int result;
            {
                hpx::threads::executors::io_pool_executor scheduler;
                scheduler.add(hpx::util::bind(&local_file::lseek_work,
                    this, offset, whence, boost::ref(result)));
            }
            return result;
        }

        void lseek_work(off_t const offset, int const whence, int& result)
        {
            if (fp_ == NULL)
            {
                result = -1;
                return;
            }

            result = fseek(fp_, offset, whence);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(local_file, open);
        HPX_DEFINE_COMPONENT_ACTION(local_file, is_open);
        HPX_DEFINE_COMPONENT_ACTION(local_file, close);
        HPX_DEFINE_COMPONENT_ACTION(local_file, remove_file);
        HPX_DEFINE_COMPONENT_ACTION(local_file, read);
        HPX_DEFINE_COMPONENT_ACTION(local_file, pread);
        HPX_DEFINE_COMPONENT_ACTION(local_file, write);
        HPX_DEFINE_COMPONENT_ACTION(local_file, pwrite);
        HPX_DEFINE_COMPONENT_ACTION(local_file, lseek);

      private:
        typedef components::managed_component_base<local_file> base_type;

        std::FILE *fp_;
        std::string file_name_;
    };

}}} // hpx::io::server

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the local_file actions
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::open_action,
        local_file_open_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::is_open_action,
        local_file_is_open_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::close_action,
        local_file_close_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::remove_file_action,
        local_file_remove_file_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::read_action,
        local_file_read_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::pread_action,
        local_file_pread_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::write_action,
        local_file_write_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::pwrite_action,
        local_file_pwrite_action)
HPX_REGISTER_ACTION_DECLARATION(hpx::io::server::local_file::lseek_action,
        local_file_lseek_action)

#endif

