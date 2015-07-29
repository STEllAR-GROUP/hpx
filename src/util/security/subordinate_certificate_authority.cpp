//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_SODIUM)

#include <hpx/util/plugin.hpp>
#include <hpx/util/security/root_certificate_authority.hpp>
#include <hpx/util/security/subordinate_certificate_authority.hpp>

#include <boost/function.hpp>

namespace hpx { namespace util { namespace security
{
    subordinate_certificate_authority::~subordinate_certificate_authority()
    {
        // Bind the delete_subordinate_certificate_authority symbol dynamically
        // and invoke it.
        typedef void (*function_type)(certificate_authority_type*);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "delete_subordinate_certificate_authority");

        (*function.first)(subordinate_certificate_authority_);
    }

    void subordinate_certificate_authority::initialize()
    {
        // Bind the new_subordinate_certificate_authority symbol dynamically
        // and invoke it.
        typedef certificate_authority_type* (*function_type)(
            components::security::key_pair const&);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "new_subordinate_certificate_authority");

        HPX_ASSERT(0 == subordinate_certificate_authority_);
        subordinate_certificate_authority_ = (*function.first)(key_pair_);
    }

    components::security::signed_certificate_signing_request
        subordinate_certificate_authority::get_certificate_signing_request() const
    {
        HPX_ASSERT(0 != subordinate_certificate_authority_);

        // Bind the certificate_authority_sign_certificate_signing_request
        // symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::subordinate_certificate_authority*
          , components::security::signed_certificate_signing_request*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "subordinate_certificate_authority_get_certificate_signing_request");

        components::security::signed_certificate_signing_request signed_csr;

        (*function.first)(subordinate_certificate_authority_, &signed_csr);

        return signed_csr;
    }

    components::security::signed_certificate
        subordinate_certificate_authority::sign_certificate_signing_request(
            components::security::signed_certificate_signing_request const & signed_csr)
        const
    {
        HPX_ASSERT(0 != subordinate_certificate_authority_);

        // Bind the certificate_authority_sign_certificate_signing_request
        // symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::signed_certificate_signing_request const &
          , components::security::signed_certificate*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_sign_certificate_signing_request");

        components::security::signed_certificate signed_certificate;

        (*function.first)(
            subordinate_certificate_authority_
          , signed_csr
          , &signed_certificate);

        return signed_certificate;
    }

    void subordinate_certificate_authority::set_certificate(
        components::security::signed_certificate const & signed_certificate)
    {
        HPX_ASSERT(0 != subordinate_certificate_authority_);

        // Bind the subordinate_certificate_authority_set_certificate symbol
        // dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::subordinate_certificate_authority*
          , components::security::signed_certificate const &);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "subordinate_certificate_authority_set_certificate");

        (*function.first)(
            subordinate_certificate_authority_, signed_certificate);
    }

    components::security::signed_certificate
        subordinate_certificate_authority::get_certificate(error_code& ec) const
    {
        if (0 == subordinate_certificate_authority_)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "subordinate_certificate_authority::get_certificate",
                "subordinate_certificate_authority is not initialized yet");
            return components::security::signed_certificate::invalid_signed_type;
        }

        // Bind the certificate_authority_get_certificate symbol dynamically
        // and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::signed_certificate*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_get_certificate");

        components::security::signed_certificate signed_certificate;

        (*function.first)(
            subordinate_certificate_authority_, &signed_certificate);

        return signed_certificate;
    }

    naming::gid_type subordinate_certificate_authority::get_gid() const
    {
        HPX_ASSERT(0 != subordinate_certificate_authority_);

        // Bind the certificate_authority_get_gid symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , naming::gid_type*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_get_gid");

        naming::gid_type gid;

        (*function.first)(subordinate_certificate_authority_, &gid);

        return gid;
    }

    bool subordinate_certificate_authority::is_valid() const
    {
        HPX_ASSERT(0 != subordinate_certificate_authority_);

        // Bind the certificate_authority_is_valid symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , bool*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_is_valid");

        bool valid;

        (*function.first)(subordinate_certificate_authority_, &valid);

        return valid;
    }

    components::security::key_pair const &
        subordinate_certificate_authority::get_key_pair() const
    {
        return key_pair_;
    }
}}}

#endif
