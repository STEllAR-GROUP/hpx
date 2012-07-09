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
#include <boost/utility/binary.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/merging_map.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/lcos/local/mutex.hpp>

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
struct HPX_EXPORT primary_namespace :
    components::fixed_component_base<
        HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
        primary_namespace
    >
{
    // {{{ nested types
    typedef lcos::local::mutex mutex_type;

    typedef boost::int32_t component_type;

    typedef boost::fusion::vector2<boost::uint32_t, naming::gid_type>
        partition_type;

    typedef std::map<naming::gid_type, gva>
        gva_table_type;

    typedef std::map<naming::locality, partition_type>
        partition_table_type;

    typedef util::merging_map<naming::gid_type, boost::int64_t>
        refcnt_table_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;
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
      , prefix_counter_(HPX_AGAS_BOOTSTRAP_PREFIX)
    {}

    bool route(
        parcelset::parcel const& p
        )
    {
        return route(p, throws);
    }

    bool route(
        parcelset::parcel const& p
      , error_code& ec
        );

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

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
        )
    {
        return bulk_service(reqs, throws);
    }

    // register all performance counter types exposed by this component
    void register_counter_types(
        char const* servicename
      , error_code& ec = throws);

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , error_code& ec
        );

    response allocate(
        request const& req
      , error_code& ec = throws
        );

    response bind_gid(
        request const& req
      , error_code& ec = throws
        );

    response resolve_gid(
        request const& req
      , error_code& ec = throws
        );

    response resolve_locality(
        request const& req
      , error_code& ec = throws
        );

    response free(
        request const& req
      , error_code& ec = throws
        );

    response unbind_gid(
        request const& req
      , error_code& ec = throws
        );

    response change_credit_non_blocking(
        request const& req
      , error_code& ec = throws
        );

    response change_credit_sync(
        request const& req
      , error_code& ec = throws
        );

    response localities(
        request const& req
      , error_code& ec = throws
        );

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

  private:
    boost::fusion::vector2<naming::gid_type, gva> resolve_gid_locked(
        naming::gid_type const& gid
      , error_code& ec
        );

    void increment(
        naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t credits
      , error_code& ec
        );

    /// TODO/REVIEW: Do we ensure that a GID doesn't get reinserted into the
    /// table after it's been decremented to 0 and destroyed? How do we do this
    /// efficiently?
    ///
    /// The new decrement algorithm (decrement_sweep handles 0-2,
    /// kill_non_blocking or kill_sync handles 3):
    ///
    ///    0.) Apply the decrement across the entire keyspace.
    ///    1.) Search for dead objects (e.g. objects with a reference count of
    ///        0) by iterating over the keyspace.
    ///    2.) Resolve the dead objects (retrieve the GVA, adjust for partial
    ///        matches) and remove them from the reference counting table.
    ///    3.) Kill the dead objects (fire-and-forget semantics).

    typedef boost::fusion::vector3<
        gva                 // gva
      , naming::gid_type    // gid
      , naming::gid_type    // count
    > free_entry;

    void decrement_sweep(
        std::list<free_entry>& free_list
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t credits
      , error_code& ec
        );

    void kill_non_blocking(
        std::list<free_entry>& free_list
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , error_code& ec
        );

    void kill_sync(
        std::list<free_entry>& free_list
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , error_code& ec
        );

  public:
    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service                       = primary_ns_service
      , namespace_bulk_service                  = primary_ns_bulk_service
      , namespace_route                         = primary_ns_route

        // Pseudo-actions
      , namespace_allocate                      = primary_ns_allocate
      , namespace_bind_gid                      = primary_ns_bind_gid
      , namespace_resolve_gid                   = primary_ns_resolve_gid
      , namespace_resolve_locality              = primary_ns_resolve_locality
      , namespace_free                          = primary_ns_free
      , namespace_unbind_gid                    = primary_ns_unbind_gid
      , namespace_change_credit_non_blocking    = primary_ns_change_credit_non_blocking
      , namespace_change_credit_sync            = primary_ns_change_credit_sync
      , namespace_localities                    = primary_ns_localities
      , namespace_statistics_counter            = primary_ns_statistics_counter
    }; // }}}

    typedef hpx::actions::result_action1<
        primary_namespace
      , /* return type */ response
      , /* enum value */  namespace_service
      , /* arguments */   request const&
      , &primary_namespace::service
      , threads::thread_priority_critical
    > service_action;

    typedef hpx::actions::result_action1<
        primary_namespace
      , /* return type */ std::vector<response>
      , /* enum value */  namespace_bulk_service
      , /* arguments */   std::vector<request> const&
      , &primary_namespace::bulk_service
      , threads::thread_priority_critical
    > bulk_service_action;

    typedef hpx::actions::result_action1<
        primary_namespace
      , /* return type */ bool
      , /* enum value */  namespace_route
      , /* arguments */   parcelset::parcel const&
      , &primary_namespace::route
      , threads::thread_priority_critical
    > route_action;
};

}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::server::primary_namespace::service_action,
    primary_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::server::primary_namespace::bulk_service_action,
    primary_namespace_bulk_service_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::server::primary_namespace::route_action,
    primary_namespace_route_action)

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

