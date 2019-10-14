// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/shell_path.hpp>
#include <hpx/errors.hpp>
#include <hpx/filesystem.hpp>

namespace hpx { namespace components { namespace process { namespace posix
{
    filesystem::path shell_path()
    {
        return "/bin/sh";
    }

    filesystem::path shell_path(hpx::error_code &ec)
    {
        ec = hpx::make_success_code();
        return "/bin/sh";
    }
}}}}

#endif
