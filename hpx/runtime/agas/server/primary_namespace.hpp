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

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/spinlock.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_primary_namespace_gid(); 
HPX_EXPORT naming::id_type bootstrap_primary_namespace_id(); 

namespace server
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

    response service(
        request const& req
        )
    {
        return service(req, throws);
    }

    response service(
        request const& req
      , error_code& ec
        );

    response bind_locality(
        request const& req
      , error_code& ec = throws
        );

    response bind_gid(
        request const& req
      , error_code& ec = throws
        );

    response page_fault(
        request const& req
      , error_code& ec = throws
        );

    response unbind_locality(
        request const& req
      , error_code& ec = throws
        );

    response unbind_gid(
        request const& req
      , error_code& ec = throws
        );

    response increment(
        request const& req
      , error_code& ec = throws
        );
    
    response decrement(
        request const& req
      , error_code& ec = throws
        );

    response localities(
        request const& req
      , error_code& ec = throws
        );

    enum actions 
    { // {{{ action enum
        // Actual actions
        namespace_service          = BOOST_BINARY_U(1000000)

        // Pseudo-actions
      , namespace_bind_locality    = BOOST_BINARY_U(1000001)
      , namespace_bind_gid         = BOOST_BINARY_U(1000010)
      , namespace_page_fault       = BOOST_BINARY_U(1000011)
      , namespace_unbind_locality  = BOOST_BINARY_U(1000100)
      , namespace_unbind_gid       = BOOST_BINARY_U(1000101)
      , namespace_increment        = BOOST_BINARY_U(1000110)
      , namespace_decrement        = BOOST_BINARY_U(1000111)
      , namespace_localities       = BOOST_BINARY_U(1001000)
    }; // }}}
    
    typedef hpx::actions::result_action1<
        primary_namespace
      , /* return type */ response
      , /* enum value */  namespace_service
      , /* arguments */   request const&
      , &primary_namespace::service
      , threads::thread_priority_critical
    > service_action;
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

