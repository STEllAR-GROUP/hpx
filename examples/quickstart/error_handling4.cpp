//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>

using boost::program_options::variables_map;

using hpx::no_success;
using hpx::detail::diagnostic_information;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        HPX_THROW_EXCEPTION(no_success, "main", "unhandled exception"); 
    }
    catch (boost::exception const& be) {
        std::cout << diagnostic_information(be) << std::endl;
    }
}

