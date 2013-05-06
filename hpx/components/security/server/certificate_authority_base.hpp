//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_AUTHORITY_BASE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include "certificate.hpp"
#include "public_key.hpp"
#include "secret_key.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class HPX_COMPONENT_EXPORT certificate_authority_base
      : public abstract_simple_component_base<certificate_authority_base>
    {
    public:
        certificate_authority_base();
        virtual ~certificate_authority_base();

        virtual void test() const = 0;

        void test_nonvirt() const
        {
            test();
        }
        HPX_DEFINE_COMPONENT_CONST_ACTION(
            certificate_authority_base,
            test_nonvirt,
            test_action);

        signed_type<certificate> get_certificate() const
        {
            return certificate_;
        }
        HPX_DEFINE_COMPONENT_CONST_ACTION(
            certificate_authority_base,
            get_certificate);

    protected:
        public_key public_key_;
        secret_key secret_key_;

        signed_type<certificate> certificate_;
    };
}}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::test_action
  , certificate_authority_base_test_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::security::server::certificate_authority_base::get_certificate_action
  , certificate_authority_base_get_certificate_action);

#endif
