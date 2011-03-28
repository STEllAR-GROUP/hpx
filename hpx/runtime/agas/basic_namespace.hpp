////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_37AC2DAB_1D2D_458E_ABA4_562EA435B0C3)
#define HPX_37AC2DAB_1D2D_458E_ABA4_562EA435B0C3

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/stubs/basic_namespace.hpp>

namespace hpx { namespace components { namespace agas 
{

template <typename Tag>
struct basic_namespace
  : client_base<basic_namespace<Tag>, stubs::basic_namespace<Tag> >
{
    typedef client_base<basic_namespace<Tag>, stubs::basic_namespace<Tag> >
        base_type;

    typedef typename hpx::agas::traits::key_type<Tag>::type key_type;
    typedef typename hpx::agas::traits::mapped_type<Tag>::type mapped_type;
    
    typedef typename hpx::agas::traits::bind_hook<Tag>::result_type
        bind_result_type;
    typedef typename hpx::agas::traits::resolve_hook<Tag>::result_type
        resolve_result_type;
    typedef typename hpx::agas::traits::unbind_hook<Tag>::result_type
        unbind_result_type;

    explicit basic_namespace(naming::id_type const& gid = naming::invalid_id)
      : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    // Bind value to key. 
    lcos::future_value<bind_result_type>
    bind_async(key_type const& key, mapped_type const& value)
    { return this->base_type::bind_async(this->gid_, key, value); }

    bind_result_type
    bind(key_type const& key, mapped_type const& value)
    { return this->base_type::bind(this->gid_, key, value); }

    ///////////////////////////////////////////////////////////////////////////
    // Resolve key to value.
    lcos::future_value<resolve_result_type>
    resolve_async(key_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    resolve_result_type
    resolve(key_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
    
    ///////////////////////////////////////////////////////////////////////////
    // Remove key. 
    lcos::future_value<unbind_result_type>
    unbind_async(key_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    unbind_result_type
    unbind(key_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
};            

}}}

#endif // HPX_37AC2DAB_1D2D_458E_ABA4_562EA435B0C3

