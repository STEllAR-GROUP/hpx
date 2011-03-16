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

    explicit basic_namespace(naming::id_type const& gid = naming::invalid_id)
      : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    // Bind value to key. Behavior is Tag specific if the key is already bound. 
    lcos::future_value<key_type>
    bind_async(key_type const& key, mapped_type const& value)
    { return this->base_type::bind_async(this->gid_, key, value); }

    key_type
    bind(key_type const& key, mapped_type const& value)
    { return this->base_type::bind(this->gid_, key, value); }
    
    ///////////////////////////////////////////////////////////////////////////
    // Update key. Behavior is Tag specific if the key is not bound. 
    lcos::future_value<bool>
    update_async(key_type const& key, mapped_type const& value)
    { return this->base_type::update_async(this->gid_, key, value); }

    bool
    update(key_type const& key, mapped_type const& value)
    { return this->base_type::update(this->gid_, key, value); }

    ///////////////////////////////////////////////////////////////////////////
    // Resolve key to value. Returns an invalid/empty type if key is unbound.
    lcos::future_value<mapped_type>
    resolve_async(key_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    mapped_type
    resolve(key_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
    
    ///////////////////////////////////////////////////////////////////////////
    // Remove key. Returns false if the key wasn't bound, true otherwise. 
    lcos::future_value<bool>
    unbind_async(key_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    bool
    unbind(key_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
};            

}}}

#endif // HPX_37AC2DAB_1D2D_458E_ABA4_562EA435B0C3

