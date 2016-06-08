//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_ROOT_CERTIFICATE_AUTHORITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_ROOT_CERTIFICATE_AUTHORITY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/runtime/components/server/fixed_component_base.hpp>

#include <hpx/components/security/key_pair.hpp>
#include "certificate_authority_base.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class HPX_COMPONENT_EXPORT root_certificate_authority
      : public certificate_authority_base
      , public fixed_component_base<root_certificate_authority>
    {
    public:
        typedef root_certificate_authority type_holder;
        typedef certificate_authority_base base_type_holder;

        root_certificate_authority();
        root_certificate_authority(key_pair const &);

        signed_type<certificate> sign_certificate_signing_request(
            signed_type<certificate_signing_request> const &) const;

        naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const
        {
            typedef fixed_component_base<root_certificate_authority>
                component_base_type;
            return this->component_base_type::get_base_gid(assign_gid);
        }
    };
}}}}

#endif

#endif
