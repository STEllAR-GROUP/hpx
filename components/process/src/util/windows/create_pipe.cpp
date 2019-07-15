// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/exception.hpp>

#include <hpx/components/process/util/windows/create_pipe.hpp>

namespace hpx { namespace components { namespace process { namespace windows
{
    pipe create_pipe()
    {
        HANDLE handles[2];
        if (!::CreatePipe(&handles[0], &handles[1], nullptr, 0))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "posix::create_pipe", "CreatePipe() failed");
        }
        return make_pipe(handles[0], handles[1]);
    }

    pipe create_pipe(hpx::error_code &ec)
    {
        HANDLE handles[2] = { nullptr, nullptr };
        if (!::CreatePipe(&handles[0], &handles[1], nullptr, 0))
        {
            HPX_THROWS_IF(ec, invalid_status,
                "posix::create_pipe", "CreatePipe() failed");
        }
        else
        {
            ec = hpx::make_success_code();
        }
        return make_pipe(handles[0], handles[1]);
    }
}}}}

#endif
