//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/security/root_ca.hpp>

namespace hpx { namespace util { namespace security
{
    root_ca::~root_ca()
    {
        // Bind the delete_root_ca symbol dynamically and invoke it.
        typedef void (*function_type)(ca_type*);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll module(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> p =
            module.get<function_type, deleter_type>("delete_root_ca");

        (*p.first)(root_ca_);

        delete key_pair_;
    }

    naming::gid_type root_ca::get_gid() const
    {
        BOOST_ASSERT(0 != root_ca_);

        // Bind the ca_get_gid symbol dynamically and invoke it.
        typedef naming::gid_type (*function_type)(
            components::security::server::certificate_authority_base*);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll module(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> p =
            module.get<function_type, deleter_type>("ca_get_gid");

        return (*p.first)(root_ca_);
    }

    void root_ca::init()
    {
        key_pair_ = new components::security::server::key_pair;

        // Bind the create_root_ca symbol dynamically and invoke it.
        typedef ca_type* (*function_type)(
            components::security::server::key_pair);
        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll module(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> p =
            module.get<function_type, deleter_type>("create_root_ca");

        root_ca_ = (*p.first)(*key_pair_);
    }

    components::security::server::signed_type<
        components::security::server::certificate> root_ca::get_certificate()
    {
        BOOST_ASSERT(0 != root_ca_);

        // Bind the ca_get_certificate symbol dynamically and invoke it.
        typedef components::security::server::signed_type<
            components::security::server::certificate
        > (*function_type)(
            components::security::server::certificate_authority_base*);

        typedef boost::function<void(function_type)> deleter_type;

        hpx::util::plugin::dll module(
            HPX_MAKE_DLL_STRING(std::string("security")));
        std::pair<function_type, deleter_type> p =
            module.get<function_type, deleter_type>("ca_get_certificate");

        return (*p.first)(root_ca_);
    }
}}}

