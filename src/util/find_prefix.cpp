////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if defined(__linux__)
    #include <link.h>
    #include <dlfcn.h>
    #include <limits.h>

    #include <hpx/config.hpp>

    #include <boost/filesystem/path.hpp>
#endif

namespace hpx { namespace util
{

#if defined(__linux__)

std::string find_prefix(
    std::string library
    )
{
    #if defined(HPX_DEBUG)
        std::string const library_file
            = std::string("lib") + library + "d" + HPX_SHARED_LIB_EXTENSION;
    #else
        std::string const library_file
            = std::string("lib") + library + HPX_SHARED_LIB_EXTENSION;
    #endif

    void* handle = ::dlopen(library_file.c_str(), RTLD_LAZY);

    if (!handle)
    {
        ::dlerror();
        return HPX_PREFIX;
    }

    char directory[PATH_MAX];

    if (::dlinfo(handle, RTLD_DI_ORIGIN, directory) < 0)
    {
        ::dlclose(handle);
        ::dlerror();
        return HPX_PREFIX;
    }

    using boost::filesystem::path;

    std::string const prefix
        = path(directory).parent_path().parent_path().string();

    if (prefix.empty())
    {
        ::dlclose(handle);
        ::dlerror();
        return HPX_PREFIX;
    }

    else
    {
        ::dlclose(handle);
        ::dlerror();
        return prefix;
    }
}

#elif defined(BOOST_WINDOWS)

// TODO: Windows

std::string find_prefix(
    std::string library
    )
{
    return HPX_PREFIX;
}

#else

std::string find_prefix(
    std::string library
    )
{
    return HPX_PREFIX;
}

#endif

}}

