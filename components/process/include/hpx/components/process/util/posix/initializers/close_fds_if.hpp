// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#if !defined(HPX_PROCESS_POSIX_MAX_FD)
#   define HPX_PROCESS_POSIX_MAX_FD 32
#endif

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

template <class Predicate>
class close_fds_if_ : public initializer_base
{
private:
    static void close(int fd)
    {
        ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    }

public:
    explicit close_fds_if_(const Predicate &pred) : pred_(pred) {}

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor&) const
    {
        for (int fd = 0; fd != upper_bound(); ++fd)
        {
            if (pred_(fd))
            {
                close(fd);
            }
        }
    }

private:
    static int upper_bound()
    {
        int up;
#if defined(F_MAXFD)
        do
        {
            up = ::fcntl(0, F_MAXFD);
        } while (up == -1 && errno == EINTR);
        if (up == -1)
#endif
            up = static_cast<int>(::sysconf(_SC_OPEN_MAX));
        if (up == -1)
            up = HPX_PROCESS_POSIX_MAX_FD;
        return up;
    }

    Predicate pred_;
};

template <class Predicate>
close_fds_if_<Predicate> close_fds_if(const Predicate &pred)
{
    return close_fds_if_<Predicate>(pred);
}

}

}}}}

#endif
