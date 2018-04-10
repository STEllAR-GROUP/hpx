//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::find_here;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map&
    )
{
    {
#       define HPX_SIZEOF(type)                                               \
            hpx::util::format("{1:-40} {2}\n",                                \
                HPX_PP_STRINGIZE(type), sizeof(type))                         \
            /**/

        cout << HPX_SIZEOF(hpx::naming::gid_type)
             << HPX_SIZEOF(hpx::naming::id_type)
             << HPX_SIZEOF(hpx::naming::address)
             << HPX_SIZEOF(hpx::threads::thread_data)
             << flush;

#       undef HPX_SIZEOF
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

