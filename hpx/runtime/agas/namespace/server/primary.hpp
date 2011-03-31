////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <hpx/hpx_fwd.hpp>
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
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef typename traits::network::endpoint_type<Protocol>::type
        endpoint_type;

    typedef gva<Protocol> gva_type;
    typedef full_gva<Protocol> full_gva_type;
    typedef boost::uint64_t refcnt_type;

    typedef table<Database, naming::gid_type, full_gva_type>
        gva_table_type; 

    typedef table<Database, endpoint_type, partition>
        locality_table_type;
    
    typedef table<Database, naming::gid_type, refcnt>
        refcnt_table_type;

    enum actions
    {
        namespace_bind_gid_range,
        namespace_bind_locality,
        namespace_resolve_locality,
        namespace_resolve_gid,
        namespace_unbind_locality,
        namespace_unbind_gid_range,
        namespace_increment,
        namespace_decrement
    };
  
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

    // TODO: implement actions
};

}}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

