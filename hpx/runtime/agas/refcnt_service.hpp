////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1A970276_D1AA_4DDD_BA1E_87C424FDDAD0)
#define HPX_1A970276_D1AA_4DDD_BA1E_87C424FDDAD0

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/refcnt_service.hpp>

namespace hpx { namespace components { namespace agas 
{

struct refcnt_service
  : client_base<refcnt_service, stubs::refcnt_service>
{
    typedef client_base<refcnt_service, stubs::refcnt_service>
        base_type;

    typedef stubs::refcnt_service::registry_type::key_type key_type;
    typedef stubs::refcnt_service::registry_type::mapped_type mapped_type; 

    typedef stubs::refcnt_service::decrement_result_type decrement_result_type;

    explicit refcnt_service(naming::id_type const& gid = naming::invalid_id)
      : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<key_type>
    increment_async(key_type const& key, mapped_type count = 1)
    { return this->base_type::increment_async(this->gid_, key, count); }
 
    key_type
    increment(key_type const& key, mapped_type count = 1)
    { return this->base_type::increment(this->gid_, key, count); }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<decrement_result_type>
    decrement_async(key_type const& key, mapped_type count = 1)
    { return this->base_type::decrement_async(this->gid_, key, count); }
    
    decrement_result_type
    decrement(key_type const& key, mapped_type count = 1)
    { return this->base_type::decrement(this->gid_, key, count); }
};            

}}}

#endif // HPX_1A970276_D1AA_4DDD_BA1E_87C424FDDAD0

