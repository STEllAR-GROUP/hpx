//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/util.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/modules/format.hpp>

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;

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

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Initialize and run HPX.
    return hpx::init(argc, argv);
}
#endif
