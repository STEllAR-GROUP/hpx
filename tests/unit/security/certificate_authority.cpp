//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/security/server/certificate_store.hpp>
#include <hpx/components/security/server/public_key.hpp>
#include <hpx/components/security/server/signed_type.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/security/root_certificate_authority.hpp>
#include <hpx/util/security/subordinate_certificate_authority.hpp>

#include <iostream>

int hpx_main(boost::program_options::variables_map &)
{
    {
        using namespace hpx::components::security;

        hpx::util::security::root_certificate_authority
            root_certificate_authority;
        root_certificate_authority.initialize();

        server::signed_type<server::certificate> const & root_certificate =
            root_certificate_authority.get_certificate();

        std::cout << root_certificate << std::endl;

        server::certificate_store store(root_certificate);

        server::public_key const & root_public_key =
            root_certificate.get_type().get_subject_public_key();

        std::cout << root_public_key << std::endl;

        HPX_TEST(root_public_key.verify(root_certificate));


        hpx::util::security::subordinate_certificate_authority
            subordinate_certificate_authority;
        subordinate_certificate_authority.initialize();

        server::signed_type<server::certificate> const &
            subordinate_certificate =
                subordinate_certificate_authority.get_certificate();

        std::cout << subordinate_certificate << std::endl;

        store.insert(subordinate_certificate);

        HPX_TEST(root_public_key.verify(subordinate_certificate));


        /* server::hash hash;
        hash.update(
            reinterpret_cast<unsigned char const *>("Hello, world!"), 13);

        server::parcel parcel(0, hash);

        server::signed_type<server::parcel> signed_parcel =
            subordinate_key_pair.sign(parcel); */
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    boost::program_options::options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
