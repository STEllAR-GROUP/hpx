//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/lcos/promise.hpp>

HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::int64_t>, int64_t_promise,
    hpx::components::int64_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::uint64_t>, uint64_t_promise,
    hpx::components::uint64_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<bool>, bool_promise,
    hpx::components::bool_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<hpx::util::section>, section_promise,
    hpx::components::section_promise)

#if defined(HPX_SECURITY)

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/security/certificate_authority_base.hpp>
#include <hpx/components/security/server/subordinate_certificate_authority.hpp>

HPX_REGISTER_PROMISE(hpx::lcos::promise<
    hpx::components::security::signed_type<hpx::components::security::certificate>
>, signed_certificate_promise, hpx::components::signed_certificate_promise)

#endif
