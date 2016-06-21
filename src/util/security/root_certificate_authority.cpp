//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SODIUM)

#include <hpx/util/function.hpp>
#include <hpx/util/security/root_certificate_authority.hpp>

#include <string>

namespace hpx { namespace util { namespace security
{
    root_certificate_authority::~root_certificate_authority()
    {
        // Bind the delete_root_certificate_authority symbol dynamically and invoke it.
        typedef void (*function_type)(certificate_authority_type*);
        typedef function_nonser<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "delete_root_certificate_authority");

        (*function.first)(root_certificate_authority_);

        delete key_pair_;
    }

    void root_certificate_authority::initialize()
    {
        HPX_ASSERT(0 == key_pair_);
        key_pair_ = new components::security::key_pair;

        // Bind the new_root_certificate_authority symbol dynamically and invoke it.
        typedef certificate_authority_type* (*function_type)(
            components::security::key_pair const &);
        typedef function_nonser<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "new_root_certificate_authority");

        HPX_ASSERT(0 == root_certificate_authority_);
        root_certificate_authority_ = (*function.first)(*key_pair_);
    }

    components::security::signed_certificate
        root_certificate_authority::sign_certificate_signing_request(
            components::security::signed_certificate_signing_request
            const & signed_csr) const
    {
        HPX_ASSERT(0 != root_certificate_authority_);

        // Bind the certificate_authority_sign_certificate_signing_request
        //  symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::signed_certificate_signing_request const &
          , components::security::signed_certificate*);

        typedef function_nonser<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_sign_certificate_signing_request");

        components::security::signed_certificate signed_certificate;

        (*function.first)(
            root_certificate_authority_
          , signed_csr
          , &signed_certificate);

        return signed_certificate;
    }

    components::security::signed_certificate
        root_certificate_authority::get_certificate(error_code& ec) const
    {
        if (0 == root_certificate_authority_)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "root_certificate_authority::get_certificate",
                "root_certificate_authority is not initialized yet");
            return components::security::signed_certificate::invalid_signed_type;
        }

        // Bind the certificate_authority_get_certificate symbol dynamically and
        // invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::signed_certificate*);

        typedef function_nonser<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_get_certificate");

        components::security::signed_certificate signed_certificate;

        (*function.first)(root_certificate_authority_, &signed_certificate);

        return signed_certificate;
    }

    naming::gid_type root_certificate_authority::get_gid()
    {
        return naming::gid_type(
            HPX_ROOT_CERTIFICATE_AUTHORITY_MSB
          , HPX_ROOT_CERTIFICATE_AUTHORITY_LSB);
    }

    bool root_certificate_authority::is_valid() const
    {
        HPX_ASSERT(0 != root_certificate_authority_);

        // Bind the certificate_authority_is_valid symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , bool*);

        typedef function_nonser<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_is_valid");

        bool valid;

        (*function.first)(root_certificate_authority_, &valid);

        return valid;
    }
}}}

#endif
