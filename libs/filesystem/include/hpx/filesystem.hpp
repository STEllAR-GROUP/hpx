//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  hpxinspect:nodeprecatedinclude:boost/filesystem.hpp

/// \file
/// This file provides a compatibility layer using Boost.Filesystem for the
/// C++17 filesystem library. It is *not* intended to be a complete
/// compatibility layer. It only contains functions required by the HPX
/// codebase. It also provides some functions only available in Boost.Filesystem
/// when using C++17 filesystem.

#pragma once

#include <hpx/config.hpp>
#include <hpx/filesystem/config/defines.hpp>

#if !defined(HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY)
#include <filesystem>
#include <string>
#include <system_error>

namespace hpx { namespace filesystem {
    using namespace std::filesystem;
    using std::error_code;
    using std::filesystem::canonical;

    inline path initial_path()
    {
        static path ip = current_path();
        return ip;
    }

    inline std::string basename(path const& p)
    {
        return p.stem().string();
    }

    inline path canonical(path const& p, path const& base)
    {
        if (p.is_relative())
        {
            return canonical(base / p);
        }
        else
        {
            return canonical(p);
        }
    }

    inline path canonical(path const& p, path const& base, error_code& ec)
    {
        if (p.is_relative())
        {
            return canonical(base / p, ec);
        }
        else
        {
            return canonical(p, ec);
        }
    }

}}    // namespace hpx::filesystem
#else
#include <boost/filesystem.hpp>

static_assert(BOOST_FILESYSTEM_VERSION == 3,
    "HPX requires Boost.Filesystem version 3 (or support for the C++17 "
    "filesystem library)");

namespace hpx { namespace filesystem {
    using namespace boost::filesystem;
    using boost::system::error_code;
}}    // namespace hpx::filesystem
#endif
