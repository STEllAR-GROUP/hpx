//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/security/certificate_authority.hpp>

int hpx_main(boost::program_options::variables_map &)
{
    {
        hpx::components::security::certificate_authority certificate_authority;

        certificate_authority.create(hpx::find_here());

        certificate_authority.get_certificate_signing_request();
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    boost::program_options::options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
