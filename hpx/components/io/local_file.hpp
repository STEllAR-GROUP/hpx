//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_IO_LOCAL_FILE_HPP_AUG_26_2014_1102AM)
#define HPX_COMPONENTS_IO_LOCAL_FILE_HPP_AUG_26_2014_1102AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/client.hpp>
#include <hpx/components/io/server/local_file.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace io
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a local_file class is the client side representation of a
    // concrete \a server#local_file component
    class local_file :
        public components::client_base<local_file, server::local_file>
    {
    private:
        typedef components::client_base<local_file, server::local_file>
            base_type;

    public:
        local_file(naming::id_type gid) : base_type(gid) {}

        local_file(hpx::future<naming::id_type> && gid)
          : base_type(std::move(gid))
        {}

        lcos::future<void> open(std::string const& name, std::string const& mode)
        {
            typedef server::local_file::open_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    name, mode);
        }

        void open_sync(std::string const& name, std::string const& mode)
        {
            return open(name, mode).get();
        }

        lcos::future<bool> is_open()
        {
            typedef server::local_file::is_open_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid());
        }

        bool is_open_sync()
        {
            return is_open().get();
        }

        lcos::future<void> close()
        {
            typedef server::local_file::close_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid());
        }

        void close_sync()
        {
            return close().get();
        }

        lcos::future<int> remove_file(std::string const& file_name)
        {
            typedef server::local_file::remove_file_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    file_name);
        }

        int remove_file_sync(std::string const& file_name)
        {
            return remove_file(file_name).get();
        }

        lcos::future<std::vector<char> > read(size_t const& count)
        {
            typedef server::local_file::read_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    count);
        }

        std::vector<char> read_sync(size_t const count)
        {
            return read(count).get();
        }

        lcos::future<std::vector<char> > pread(ssize_t const count,
                off_t const offset)
        {
            typedef server::local_file::pread_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    count, offset);
        }

        std::vector<char> pread_sync(size_t const count, off_t const offset)
        {
            return pread(count, offset).get();
        }

        lcos::future<ssize_t> write(std::vector<char> const& buf)
        {
            typedef server::local_file::write_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    buf);
        }

        ssize_t write_sync(std::vector<char> const& buf)
        {
            return write(buf).get();
        }

        lcos::future<ssize_t> pwrite(std::vector<char> const& buf,
                off_t const offset)
        {
            typedef server::local_file::pwrite_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    buf, offset);
        }

        ssize_t pwrite_sync(std::vector<char> const& buf, off_t const offset)
        {
            return pwrite(buf, offset).get();
        }

        lcos::future<int> lseek(off_t const offset, int const whence)
        {
            typedef server::local_file::lseek_action action_type;
            return hpx::async<action_type>(this->base_type::get_gid(),
                    offset, whence);
        }

        int lseek_sync(off_t const offset, int const whence)
        {
            return lseek(offset, whence).get();
        }
    };

}} // hpx::io


#endif

