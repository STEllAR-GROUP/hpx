//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_FILESYSTEM_COMPATIBILITY_JAN_28_2010_0833PM)
#define HPX_FILESYSTEM_COMPATIBILITY_JAN_28_2010_0833PM

#include <string>

#include <boost/version.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace hpx { namespace util
{
#if BOOST_VERSION >= 105000
#define HPX_FILESYSTEM3 filesystem
#else
#define HPX_FILESYSTEM3 filesystem3
#endif

///////////////////////////////////////////////////////////////////////////////
// filesystem wrappers allowing to handle different Boost versions
#if !defined(BOOST_FILESYSTEM_NO_DEPRECATED)
// interface wrappers for older Boost versions
    inline boost::filesystem::path initial_path()
    {
        return boost::filesystem::initial_path();
    }

    inline boost::filesystem::path current_path()
    {
        return boost::filesystem::current_path();
    }

    inline boost::filesystem::path create_path(std::string const& p)
    {
        char back = p.empty() ? 0 : p[p.length()-1];
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::filesystem::path(back == ':' ? p.substr(0, p.size()-1) : p);
#else
        return boost::filesystem::path(back == ':' ? p.substr(0, p.size()-1)
            : p, boost::filesystem::native);
#endif
    }

    inline std::string leaf(boost::filesystem::path const& p)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return p.leaf().string();
#else
        return p.leaf();
#endif
    }

    inline boost::filesystem::path branch_path(boost::filesystem::path const& p)
    {
        return p.branch_path();
    }

    inline boost::filesystem::path normalize(boost::filesystem::path& p)
    {
        return p.normalize();
    }

    inline std::string native_file_string(boost::filesystem::path const& p)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return p.string();
#else
        return p.native_file_string();
#endif
    }

    inline boost::filesystem::path complete_path(
        boost::filesystem::path const& p)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::HPX_FILESYSTEM3::complete(p, initial_path());
#else
        return boost::filesystem::complete(p, initial_path());
#endif
    }

    inline boost::filesystem::path complete_path(
        boost::filesystem::path const& p, boost::filesystem::path const& base)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::HPX_FILESYSTEM3::complete(p, base);
#else
        return boost::filesystem::complete(p, base);
#endif
    }

#else

// interface wrappers if deprecated functions do not exist
    inline boost::filesystem::path initial_path()
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::HPX_FILESYSTEM3::detail::initial_path();
#else
        return boost::filesystem::initial_path<boost::filesystem::path>();
#endif
    }

    inline boost::filesystem::path current_path()
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::HPX_FILESYSTEM3::current_path();
#else
        return boost::filesystem::current_path<boost::filesystem::path>();
#endif
    }

    template <typename String>
    inline boost::filesystem::path create_path(String const& p)
    {
        return boost::filesystem::path(p);
    }

    inline std::string leaf(boost::filesystem::path const& p)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return p.filename().string();
#else
        return p.filename();
#endif
    }

    inline boost::filesystem::path branch_path(boost::filesystem::path const& p)
    {
        return p.parent_path();
    }

    inline boost::filesystem::path normalize(boost::filesystem::path& p)
    {
        return p; // function doesn't exist anymore
    }

    inline std::string native_file_string(boost::filesystem::path const& p)
    {
        return p.string();
    }

    inline boost::filesystem::path complete_path(
        boost::filesystem::path const& p)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::filesystem::absolute(p, initial_path());
#else
        return boost::filesystem::complete(p, initial_path());
#endif
    }

    inline boost::filesystem::path complete_path(
        boost::filesystem::path const& p, boost::filesystem::path const& base)
    {
#if BOOST_FILESYSTEM_VERSION >= 3
        return boost::filesystem::absolute(p, base);
#else
        return boost::filesystem::complete(p, base);
#endif
    }
#endif

#if BOOST_VERSION <= 104900 || BOOST_FILESYSTEM_VERSION < 3
    inline boost::filesystem::path resolve(
        boost::filesystem::path const& p,
        boost::filesystem::path const& base = current_path())
    {
        boost::filesystem::path abspath = boost::filesystem::absolute(p, base);
        boost::filesystem::path result;
        for(boost::filesystem::path::iterator it = abspath.begin();
            it != abspath.end(); ++it)
        {
            if (*it == "..")
            {
                // /a/b/.. is not necessarily /a if b is a symbolic link
                if (boost::filesystem::is_symlink(result))
                    result /= *it;

                // /a/b/../.. is not /a/b/.. under most circumstances
                // We can end up with ..s in our result because of symbolic links
                else if (result.filename() == "..")
                    result /= *it;

                // Otherwise it should be safe to resolve the parent
                else
                    result = result.parent_path();
            }
            else if (*it == ".")
            {
                // Ignore
            }
            else
            {
                // Just cat other path entries
                result /= *it;
            }
        }
        return result;
    }
#endif

    inline boost::filesystem::path canonical_path(
        boost::filesystem::path const& p, boost::system::error_code& ec)
    {
#if BOOST_VERSION > 104900 && BOOST_FILESYSTEM_VERSION >= 3
        return boost::filesystem::canonical(p, initial_path(), ec);
#else
        return resolve(p, initial_path());
#endif
    }

#undef HPX_FILESYSTEM3
}}

#endif
