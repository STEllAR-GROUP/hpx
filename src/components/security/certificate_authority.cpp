//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/components/security/server/certificate_authority.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::security::server::certificate_authority
    certificate_authority_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<certificate_authority_type>
  , certificate_authority);

HPX_REGISTER_ACTION(
    certificate_authority_type::get_certificate_signing_request_action
  , certificate_authority_get_certificate_signing_request_action);
