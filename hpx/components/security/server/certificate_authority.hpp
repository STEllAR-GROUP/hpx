//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include "certificate_signing_request.hpp"
#include "public_key.hpp"
#include "secret_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class HPX_COMPONENT_EXPORT certificate_authority
      : public simple_component_base<certificate_authority>
    {
    public:
        certificate_authority();

        signed_type<certificate_signing_request>
        get_certificate_signing_request() const;

        HPX_DEFINE_COMPONENT_CONST_ACTION(
            certificate_authority
          , get_certificate_signing_request);

    protected:
        certificate_authority(naming::id_type const &);

    private:
        naming::id_type subject_;

        public_key public_key_;
        secret_key secret_key_;
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority::get_certificate_signing_request_action
  , certificate_authority_get_certificate_signing_request_action);

#endif
