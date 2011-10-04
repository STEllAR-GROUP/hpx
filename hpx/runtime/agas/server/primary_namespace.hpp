////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <boost/format.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/response.hpp>

namespace hpx { namespace agas { namespace server
{

/// \brief AGAS's primary namespace maps 128-bit global identifiers (GIDs) to
/// resolved addresses.
///
/// \note The layout of the address space is implementation defined, and
/// subject to change. Never write application code that relies on the internal
/// layout of GIDs. AGAS only guarantees that all assigned GIDs will be unique.
/// 
/// The following is the canonical description of the partitioning of AGAS's
/// primary namespace.
///
///     |-----MSB------||------LSB-----|
///     BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
///     |prefix||RC||----identifier----|
///     
///     MSB        - Most significant bits (bit 64 to bit 127)
///     LSB        - Least significant bits (bit 0 to bit 63)
///     prefix     - Highest 32 bits (bit 96 to bit 127) of the MSB. Each
///                  locality is assigned a prefix. This creates a 96-bit
///                  address space for each locality.
///     RC         - Bit 80 to bit 95 of the MSB. This is the number of
///                  reference counting credits on the GID.
///     identifier - Bit 64 to bit 80 of the MSB, and the entire LSB. The
///                  content of these bits depends on the component type of
///                  the underlying object. For all user-defined components,
///                  these bits contain a unique 80-bit number which is
///                  assigned sequentially for each locality. For
///                  \a hpx#components#component_runtime_support and
///                  \a hpx#components#component_memory, the high 16 bits are
///                  zeroed and the low 64 bits hold the LVA of the component.
///
/// The following address ranges are reserved. Some are either explicitly or 
/// implicitly protected by AGAS. The letter x represents a single-byte
/// wildcard.
///
///     00000000xxxxxxxxxxxxxxxxxxxxxxxx
///         Historically unused address space reserved for future use.
///     xxxxxxxxxxxx0000xxxxxxxxxxxxxxxx
///         Address space for LVA-encoded GIDs.
///     00000001xxxxxxxxxxxxxxxxxxxxxxxx
///         Prefix of the bootstrap AGAS locality.
///     00000001000000010000000000000001
///         Address of the primary_namespace component on the bootstrap AGAS
///         locality.
///     00000001000000010000000000000002 
///         Address of the component_namespace component on the bootstrap AGAS
///         locality.
///     00000001000000010000000000000003
///         Address of the symbol_namespace component on the bootstrap AGAS
///         locality.
///
struct primary_namespace : 
  components::fixed_component_base<
    HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
    primary_namespace
  >
{
    // {{{ nested types
    typedef util::spinlock database_mutex_type;

    typedef naming::locality endpoint_type;

    typedef gva gva_type;
    typedef gva_type::count_type count_type;
    typedef gva_type::offset_type offset_type;
    typedef boost::int32_t component_type;
    typedef boost::uint32_t prefix_type;

    typedef boost::fusion::vector2<prefix_type, naming::gid_type>
        partition_type;

    typedef std::map<naming::gid_type, gva_type>
        gva_table_type; 

    typedef std::map<endpoint_type, partition_type>
        partition_table_type;
    
    typedef std::map<naming::gid_type, count_type>
        refcnt_table_type;
    // }}}
 
  private:
    database_mutex_type mutex_;
    gva_table_type gvas_;
    partition_table_type partitions_;
    refcnt_table_type refcnts_;
    boost::uint32_t prefix_counter_; 

  public:
    primary_namespace()
      : mutex_()
      , gvas_()
      , partitions_()
      , refcnts_()
      , prefix_counter_(0)
    {}

    response bind_locality(
        endpoint_type const& ep
      , count_type count
        )
    { 
        return bind_locality(ep, count, throws);
    } 

    response bind_locality(
        endpoint_type const& ep
      , count_type count
      , error_code& ec
        );

    response bind_gid(
        naming::gid_type const& gid
      , gva_type const& gva
        )
    {
        return bind_gid(gid, gva, throws);
    }

    response bind_gid(
        naming::gid_type const& gid
      , gva_type const& gva
      , error_code& ec
        );

    response page_fault(
        naming::gid_type const& gid
        )
    {
        return page_fault(gid, throws);
    }

    response page_fault(
        naming::gid_type const& gid
      , error_code& ec
        );

    response unbind_locality(
        endpoint_type const& ep
        )
    {
        return unbind_locality(ep, throws);
    }

    response unbind_locality(
        endpoint_type const& ep
      , error_code& ec
        );

    response unbind_gid(
        naming::gid_type const& gid
      , count_type count
        )
    {
        return unbind_gid(gid, count, throws);
    }

    response unbind_gid(
        naming::gid_type const& gid
      , count_type count
      , error_code& ec
        );

    response increment(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return increment(gid, credits, throws);
    }

    response increment(
        naming::gid_type const& gid
      , count_type credits
      , error_code& ec
        );
    
    response decrement(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return decrement(gid, credits, throws); 
    }
       
    response decrement(
        naming::gid_type const& gid
      , count_type credits
      , error_code& ec
        );
 
    response localities()
    {
        return localities(throws);
    }

    response localities(
        error_code& ec
        );

    enum actions 
    { // {{{ action enum
        namespace_bind_locality    = BOOST_BINARY_U(1000000),
        namespace_bind_gid         = BOOST_BINARY_U(1000001),
        namespace_page_fault      = BOOST_BINARY_U(1000010),
        namespace_unbind_locality  = BOOST_BINARY_U(1000011),
        namespace_unbind_gid       = BOOST_BINARY_U(1000100),
        namespace_increment        = BOOST_BINARY_U(1000101),
        namespace_decrement        = BOOST_BINARY_U(1000110),
        namespace_localities       = BOOST_BINARY_U(1000111),
    }; // }}}
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_bind_locality,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace::bind_locality
      , threads::thread_priority_critical
    > bind_locality_action; 
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_bind_gid,
        /* arguments */   naming::gid_type const&, gva_type const&,
        &primary_namespace::bind_gid
      , threads::thread_priority_critical
    > bind_gid_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_page_fault,
        /* arguments */   naming::gid_type const&,
        &primary_namespace::page_fault
      , threads::thread_priority_critical
    > page_fault_action;

    typedef hpx::actions::result_action1<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_unbind_locality,
        /* arguments */   endpoint_type const&,
        &primary_namespace::unbind_locality
      , threads::thread_priority_critical
    > unbind_locality_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_unbind_gid,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::unbind_gid
      , threads::thread_priority_critical
    > unbind_gid_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response,  
        /* enum value */  namespace_increment,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::increment
      , threads::thread_priority_critical
    > increment_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::decrement
      , threads::thread_priority_critical
    > decrement_action;
    
    typedef hpx::actions::result_action0<
        primary_namespace,
        /* return type */ response,
        /* enum value */  namespace_localities,
        &primary_namespace::localities
      , threads::thread_priority_critical
    > localities_action;
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

