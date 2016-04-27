//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/runtime/components/component_type.hpp>

#include <hpx/components/security/server/certificate_authority_base.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::security::server::certificate_authority_base
    certificate_authority_base_type;

typedef hpx::components::fixed_component<
    certificate_authority_base_type
> certificate_authority_base_component_type;

HPX_DEFINE_GET_COMPONENT_TYPE(certificate_authority_base_type);
HPX_DEFINE_GET_COMPONENT_TYPE(certificate_authority_base_component_type);

HPX_REGISTER_ACTION(
    certificate_authority_base_type::sign_certificate_signing_request_action
  , certificate_authority_base_sign_certificate_signing_request_action);

HPX_REGISTER_ACTION(
    certificate_authority_base_type::get_certificate_action
  , certificate_authority_base_get_certificate_action);

HPX_REGISTER_ACTION(
    certificate_authority_base_type::is_valid_action
  , certificate_authority_base_is_valid_action);
