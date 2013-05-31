//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

typedef hpx::components::security::server::subordinate_certificate_authority
    subordinate_certificate_authority_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_4(
    hpx::components::simple_component<subordinate_certificate_authority_type>
  , subordinate_certificate_authority
  , "hpx::components::security::server::subordinate_certificate_authority"
  , ::hpx::components::factory_check);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    subordinate_certificate_authority_type
  , hpx::components::component_subordinate_certificate_authority)
