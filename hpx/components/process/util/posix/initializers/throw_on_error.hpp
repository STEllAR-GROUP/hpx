// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_THROW_ON_ERROR_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_THROW_ON_ERROR_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/exception.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

class throw_on_error : public initializer_base
{
public:
    template <class PosixExecutor>
    void on_fork_setup(PosixExecutor&) const
    {
        if (::pipe(fds_) == -1)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "throw_on_error::on_fork_setup", "pipe(2) failed");
        }
        if (::fcntl(fds_[1], F_SETFD, FD_CLOEXEC) == -1)
        {
            ::close(fds_[0]);
            ::close(fds_[1]);

            HPX_THROW_EXCEPTION(kernel_error,
                "throw_on_error::on_fork_setup", "fcntl(2) failed");
        }
    }

    template <class PosixExecutor>
    void on_fork_error(PosixExecutor&) const
    {
        ::close(fds_[0]);
        ::close(fds_[1]);

        HPX_THROW_EXCEPTION(kernel_error,
            "throw_on_error::on_fork_error", "fork(2) failed");
    }

    template <class PosixExecutor>
    void on_fork_success(PosixExecutor&) const
    {
        ::close(fds_[1]);
        int code;
        if (::read(fds_[0], &code, sizeof(int)) > 0)
        {
            ::close(fds_[0]);

            HPX_THROW_EXCEPTION(kernel_error,
                "throw_on_error::on_fork_success", "execve(2) failed");
        }
        ::close(fds_[0]);
    }

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor&) const
    {
        ::close(fds_[0]);
    }

    template <class PosixExecutor>
    void on_exec_error(PosixExecutor&) const
    {
        int e = errno;
        while (::write(fds_[1], &e, sizeof(int)) == -1 && errno == EINTR)
            ;
        ::close(fds_[1]);
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive&, unsigned const) {}

    mutable int fds_[2];
};

}

}}}}

#endif
#endif
