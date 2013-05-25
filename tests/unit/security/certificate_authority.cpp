//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/security/certificate_authority_base.hpp>
#include <hpx/components/security/server/certificate_store.hpp>
#include <hpx/components/security/server/key_pair.hpp>
#include <hpx/components/security/server/root_certificate_authority.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>
#include <hpx/util/lightweight_test.hpp>

int hpx_main(boost::program_options::variables_map &)
{
    {
        using namespace hpx::components::security;

        server::key_pair root_key_pair;

        certificate_authority_base root_certificate_authority(
            hpx::components::new_<
                server::root_certificate_authority
            >(hpx::find_here(), root_key_pair));

        server::signed_type<server::certificate> const & root_certificate =
            root_certificate_authority.get_certificate();

        server::certificate_store store(root_certificate);

        server::public_key const & root_public_key =
            root_certificate.get_type().get_subject_public_key();

        HPX_TEST(root_public_key.verify(root_certificate));


        server::key_pair subordinate_key_pair;

        certificate_authority_base subordinate_certificate_authority(
            hpx::components::new_<
                server::subordinate_certificate_authority
            >(hpx::find_here()
            , subordinate_key_pair
            , root_certificate_authority.get_gid()));

        server::signed_type<server::certificate> const & subordinate_certificate =
            subordinate_certificate_authority.get_certificate();

        store.insert(subordinate_certificate);

        HPX_TEST(root_public_key.verify(subordinate_certificate));
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    boost::program_options::options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
