////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A16135FC_AA32_444F_BB46_549AD456A661)
#define HPX_A16135FC_AA32_444F_BB46_549AD456A661

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace components { namespace agas { namespace server
{

template <typename Tag>
struct HPX_COMPONENT_EXPORT basic_namespace
  : simple_component_base<basic_namespace<Tag> >
{
    typedef typename hpx::agas::traits::mutex_type<Tag>::type mutex_type;
    typedef typename hpx::agas::traits::registry_type<Tag>::type registry_type;
    typedef typename hpx::agas::traits::key_type<Tag>::type key_type;
    typedef typename hpx::agas::traits::mapped_type<Tag>::type mapped_type;

    typedef typename hpx::agas::traits::bind_hook<Tag>::result_type
        bind_result_type;
    typedef typename hpx::agas::traits::resolve_hook<Tag>::result_type
        resolve_result_type;
    typedef typename hpx::agas::traits::unbind_hook<Tag>::result_type
        unbind_result_type;

    enum actions
    {
        namespace_bind,
        namespace_resolve,
        namespace_unbind
    };
  
  private:
    mutex_type _mutex;
    registry_type _registry;
  
  public:
    basic_namespace()
    { hpx::agas::traits::initialize_mutex(_mutex); }

    bind_result_type bind(key_type const& key, mapped_type const& value)
    {
        typename mutex_type::scoped_lock l(_mutex);
        return hpx::agas::traits::bind<Tag>(_registry, key, value);
    }

    resolve_result_type resolve(key_type const& key)
    {
        typename mutex_type::scoped_lock l(_mutex);
        return hpx::agas::traits::resolve<Tag>(_registry, key);
    } 
    
    unbind_result_type unbind(key_type const& key)
    {
        typename mutex_type::scoped_lock l(_mutex);
        return hpx::agas::traits::unbind<Tag>(_registry, key);
    } 

    typedef hpx::actions::result_action2<
        basic_namespace<Tag>,
        bind_result_type,                    // return type
        namespace_bind,                      // action type
        key_type const&, mapped_type const&, // arguments 
        &basic_namespace<Tag>::bind
    > bind_action;
    
    typedef hpx::actions::result_action1<
        basic_namespace<Tag>,
        resolve_result_type,                 // return type
        namespace_resolve,                   // action type
        key_type const&,                     // arguments 
        &basic_namespace<Tag>::resolve
    > resolve_action;
    
    typedef hpx::actions::result_action1<
        basic_namespace<Tag>,
        unbind_result_type,                  // return type
        namespace_unbind,                    // action type
        key_type const&,                     // arguments 
        &basic_namespace<Tag>::unbind
    > unbind_action;
};

}}}}

#endif // HPX_A16135FC_AA32_444F_BB46_549AD456A661

