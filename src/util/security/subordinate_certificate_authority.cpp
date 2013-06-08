//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(HPX_HAVE_SODIUM)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/plugin.hpp>
#include <hpx/util/security/root_certificate_authority.hpp>
#include <hpx/util/security/subordinate_certificate_authority.hpp>

#include <boost/function.hpp>

namespace hpx { namespace util { namespace security
{
    subordinate_certificate_authority::~subordinate_certificate_authority()
    {
        // Bind the delete_subordinate_certificate_authority symbol dynamically and invoke it.
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
        // Bind the new_subordinate_certificate_authority symbol dynamically and invoke it.
        typedef certificate_authority_type* (*function_type)(
            components::security::key_pair const&
          , naming::id_type const&);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "new_subordinate_certificate_authority");

        BOOST_ASSERT(0 == subordinate_certificate_authority_);
        subordinate_certificate_authority_ = (*function.first)(key_pair_,
            naming::id_type(root_certificate_authority::get_gid()
                          , naming::id_type::unmanaged));
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

        // Bind the certificate_authority_get_certificate symbol dynamically and invoke it.
        typedef void (*function_type)(
            components::security::server::certificate_authority_base*
          , components::security::signed_type<components::security::certificate>*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll dll(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> function =
            dll.get<function_type, deleter_type>(
                "certificate_authority_get_certificate");

        components::security::signed_type<
            components::security::certificate
        > certificate;

        (*function.first)(subordinate_certificate_authority_, &certificate);

        return certificate;
    }

    naming::gid_type subordinate_certificate_authority::get_gid() const
    {
        BOOST_ASSERT(0 != subordinate_certificate_authority_);

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

    components::security::key_pair const &
        subordinate_certificate_authority::get_key_pair() const
    {
        return key_pair_;
    }
}}}

#endif
