////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B5E64F65_83E3_4BE0_A83A_B49BCC7C0C3A)
#define HPX_B5E64F65_83E3_4BE0_A83A_B49BCC7C0C3A

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/lcos/codelet.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/plain_actions.hpp>

// TODO: error codes

namespace hpx { namespace lcos
{

template <
    typename Signature
>
inline naming::gid_type make_local_codelet(
    typename codelet<Signature>::action_type const& f
  , typename codelet<Signature>::callback_type const& cb
    )
{
    components::component_type const ctype
        = components::get_component_type<codelet<Signature> >();

    naming::gid_type gid
        = get_runtime_support_ptr()->create_component(ctype, 1);

    naming::address addr;

    if (naming::get_agas_client().resolve(gid, addr))
    {
        get_lva<codelet<Signature> >::call(addr.address_)->initialize(f, cb); 
    }

    else
    {
        HPX_THROW_EXCEPTION(
            hpx::unknown_component_address
          , "make_codelet"
          , "failed to resolve newly created codelet"); 
    }

    return gid; 
}

template <
    typename Signature
>
inline naming::id_type make_codelet(
    typename codelet<Signature>::action_type const& f
  , typename codelet<Signature>::callback_type const& cb
    )
{
    return naming::id_type(make_local_codelet<Signature>(f, cb)
                         , naming::id_type::managed);
}

template <
    typename Signature
>
inline naming::id_type make_codelet(
    naming::id_type const& id
  , typename codelet<Signature>::action_type const& f
  , typename codelet<Signature>::callback_type const& cb
    )
{
    return make_codelet(id.get_gid(), f, cb);  
}

template <
    typename Signature
>
inline naming::id_type make_codelet(
    naming::gid_type const& gid
  , typename codelet<Signature>::action_type const& f
  , typename codelet<Signature>::callback_type const& cb
    )
{
    typedef hpx::actions::plain_result_action2<
        naming::gid_type
      , typename codelet<Signature>::action_type const&
      , typename codelet<Signature>::callback_type const&
      , &make_local_codelet<Signature> 
    > action_type; 

    return eager_future<action_type, naming::id_type>(gid, f, cb).get(); 
}

}}

#endif

