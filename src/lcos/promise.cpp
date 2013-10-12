//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/lcos/promise.hpp>

#include <vector>

namespace hpx { namespace traits { namespace detail
{
    boost::detail::atomic_count unique_type(
        static_cast<long>(components::component_last));
}}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_PROMISE(hpx::lcos::promise<hpx::naming::gid_type>, gid_promise,
    hpx::components::gid_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<std::vector<hpx::naming::gid_type> >,
    vector_gid_romise, hpx::components::vector_gid_romise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<hpx::naming::id_type>, id_promise,
    hpx::components::id_promise)

typedef hpx::lcos::promise<
    hpx::naming::id_type, hpx::naming::gid_type
> id_gid_promise;
HPX_REGISTER_PROMISE(id_gid_promise, id_gid_promise,
    hpx::components::id_gid_promise)

HPX_REGISTER_PROMISE(hpx::lcos::promise<std::vector<hpx::naming::id_type> >,
    vector_id_promise, hpx::components::vector_id_promise)

typedef hpx::lcos::promise<
    std::vector<hpx::naming::id_type>, std::vector<hpx::naming::gid_type>
> id_vector_gid_vector_promise;
HPX_REGISTER_PROMISE(id_vector_gid_vector_promise, id_vector_gid_vector_promise,
    hpx::components::id_vector_gid_vector_promise)

