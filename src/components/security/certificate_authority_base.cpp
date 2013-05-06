//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/security/server/certificate_authority_base.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::security::server::certificate_authority_base
    certificate_authority_base_type;

HPX_DEFINE_GET_COMPONENT_TYPE(certificate_authority_base_type);

HPX_REGISTER_ACTION(
    certificate_authority_base_type::test_action
  , certificate_authority_base_test_action);

HPX_REGISTER_ACTION(
    certificate_authority_base_type::get_certificate_action
  , certificate_authority_base_get_certificate_action);
