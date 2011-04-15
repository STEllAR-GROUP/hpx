////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <boost/fusion/include/vector.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/network/full_gva.hpp>
#include <hpx/runtime/agas/partition.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct HPX_COMPONENT_EXPORT primary_namespace
  : simple_component_base<primary_namespace<Database, Protocol> >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef typename traits::network::endpoint_type<Protocol>::type
        endpoint_type;

    typedef gva<Protocol> gva_type;
    typedef full_gva<Protocol> full_gva_type;
    typedef boost::uint64_t count_type;
    typedef components::component_type component_type;

    typedef table<Database, naming::gid_type, full_gva_type>
        gva_table_type; 

    typedef table<Database, endpoint_type, partition>
        locality_table_type;
    
    typedef table<Database, naming::gid_type, refcnt>
        refcnt_table_type;
    // }}}
 
  private:
    database_mutex_type mutex_;
    gva_table_type gvas_;
    locality_table_type localities_;
    refcnt_table_type refcnts_;
  
  public:
    primary_namespace()
      : mutex_(),
        gvas_("hpx.agas.primary_namespace.gva"),
        localities_("hpx.agas.primary_namespace.locality"),
        refcnts_("hpx.agas.primary_namespace.refcnt")
    { traits::initialize_mutex(mutex_); }

    partition bind(gva_type const& gva, count_type count)
    { // {{{ bind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}

    partition rebind(endpoint_type const& ep, count_type count)
    { // {{{ rebind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}

    partition resolve_endpoint(endpoint_type const& ep)
    { // {{{ resolve_endpoint implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}

    gva_type resolve_gid(naming::gid_type const& gid)
    { // {{{ resolve_gid implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}

    bool unbind(endpoint_type const& ep, count_type count)
    { // {{{ unbind implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}

    count_type increment(naming::gid_type const& gid, count_type credits)
    { // {{{ increment implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}
    
    boost::fusion::vector2<count_type, component_type>
    decrement(naming::gid_type const& gid, count_type credits)
    { // {{{ decrement implementation
        typename database_mutex_type::scoped_lock l(mutex_);
        // IMPLEMENT
    } // }}}
 
    // {{{ action types
    enum actions 
    {
        namespace_bind,
        namespace_rebind,
        namespace_resolve_endpoint,
        namespace_resolve_gid,
        namespace_unbind,
        namespace_increment,
        namespace_decrement
    };

    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ partition,
        /* enum value */  namespace_bind,
        /* arguments */   gva_type const&,
        &primary_namespace<Database, Protocol>::bind
    > bind_action; 
   
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ partition,
        /* enum value */  namespace_rebind,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace<Database, Protocol>::rebind
    > rebind_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ partition,
        /* enum value */  namespace_resolve_endpoint,
        /* arguments */   endpoint_type const&,
        &primary_namespace<Database, Protocol>::resolve_endpoint
    > resolve_endpoint_action;

    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ gva_type,
        /* enum value */  namespace_resolve_gid,
        /* arguments */   naming::gid_type const&,
        &primary_namespace<Database, Protocol>::resolve_gid
    > resolve_gid_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ bool, 
        /* enum value */  namespace_unbind,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace<Database, Protocol>::unbind
    > unbind_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ count_type,  
        /* enum value */  namespace_increment,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::increment
    > increment_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ boost::fusion::vector2<count_type, component_type>,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::decrement
    > decrement_action;
    // }}}
};

}}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

