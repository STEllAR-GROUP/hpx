//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/make_shared.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/security/certificate_store.hpp>
#include <hpx/components/security/hash.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/public_key.hpp>
#include <hpx/components/security/signed_type.hpp>
#include <hpx/components/security/verify.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/security/root_certificate_authority.hpp>
#include <hpx/util/security/subordinate_certificate_authority.hpp>

#include <iostream>
#include <vector>

int hpx_main(boost::program_options::variables_map &)
{
    {
        using namespace hpx::components::security;

        hpx::util::security::root_certificate_authority
            root_certificate_authority;
        root_certificate_authority.initialize();

        signed_type<certificate> const & root_certificate =
            root_certificate_authority.get_certificate();

        std::cout << root_certificate << std::endl;

        certificate_store certificate_store(root_certificate);

        public_key const & root_public_key =
            root_certificate.get_type().get_subject_public_key();

        std::cout << root_public_key << std::endl;

        HPX_TEST(root_public_key.verify(root_certificate));


        hpx::util::security::subordinate_certificate_authority
            subordinate_certificate_authority;
        subordinate_certificate_authority.initialize();

        signed_type<certificate> const &
            subordinate_certificate =
                subordinate_certificate_authority.get_certificate();

        std::cout << subordinate_certificate << std::endl;

        certificate_store.insert(subordinate_certificate);

        HPX_TEST(root_public_key.verify(subordinate_certificate));


        hash hash;
        hash.update(reinterpret_cast<unsigned char const *>("Hello, world!"), 13);

        parcel_suffix parcel_suffix(
            hpx::parcelset::parcel::generate_unique_id(), hash);

        signed_type<parcel_suffix> signed_parcel_suffix =
            subordinate_certificate_authority.get_key_pair().sign(
                parcel_suffix);

        boost::shared_ptr<std::vector<char> > parcel_data =
            boost::make_shared<std::vector<char> >(
                signed_parcel_suffix.begin(), signed_parcel_suffix.end();

        HPX_TEST(verify(certificate_store, parcel_data));
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    boost::program_options::options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    HPX_TEST_EQ(hpx::init(desc_commandline, argc, argv), 0);
    return hpx::util::report_errors();
}
