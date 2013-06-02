//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(HPX_HAVE_SECURITY)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/security/root_ca.hpp>

namespace hpx { namespace util { namespace security
{
    void root_ca::init()
    {
        key_pair_ = new components::security::server::key_pair;

        // Bind the new_root_certificate_authority symbol dynamically and invoke it.
        typedef ca_type* (*function_type)(
            components::security::server::key_pair);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "new_root_certificate_authority");

        root_ca_ = (*function.first)(*key_pair_);
    }

    root_ca::~root_ca()
    {
        // Bind the delete_root_certificate_authority symbol dynamically and invoke it.
        typedef void (*function_type)(ca_type*);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "delete_root_certificate_authority");

        (*function.first)(root_ca_);

        delete key_pair_;
    }

    components::security::server::signed_type<
        components::security::server::certificate> root_ca::get_certificate()
    {
        BOOST_ASSERT(0 != root_ca_);

        // Bind the certificate_authority_get_certificate symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::server::signed_type<
                components::security::server::certificate
            >*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_get_certificate");

        components::security::server::signed_type<
            components::security::server::certificate
        > certificate;

        (*function.first)(root_ca_, &certificate);

        return certificate;
    }

    naming::gid_type root_ca::get_gid()
    {
        return naming::gid_type(
            HPX_ROOT_CERTIFICATE_AUTHORITY_MSB
          , HPX_ROOT_CERTIFICATE_AUTHORITY_LSB);
    }
}}}

#endif
