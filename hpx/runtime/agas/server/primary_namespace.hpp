////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/lcos/local/condition_variable.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 408000
#include <boost/shared_ptr.hpp>
#endif

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_primary_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_primary_namespace_id();

namespace server
{

// Base name used to register the component
char const* const primary_namespace_service_name = "primary/";

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
///     RC         - Bit 88 to bit 92 of the MSB. This is the log2 of the number
///                  of reference counting credits on the GID.
///                  Bit 93 is used by the locking scheme for gid_types.
///                  Bit 94 is a flag which is set if the credit value is valid.
///                  Bit 95 is a flag that is set if a GID's credit count is
///                  ever split (e.g. if the GID is ever passed to another
///                  locality).
///                - Bit 87 marks the gid such that it will not be stored in
///                  any of the AGAS caches. This is used mainly for ids
///                  which represent 'one-shot' objects (like promises).
///     identifier - Bit 64 to bit 86 of the MSB, and the entire LSB. The
///                  content of these bits depends on the component type of
///                  the underlying object. For all user-defined components,
///                  these bits contain a unique 88-bit number which is
///                  assigned sequentially for each locality. For
///                  \a hpx#components#component_runtime_support and
///                  \a hpx#components#component_memory, the high 24 bits are
///                  zeroed and the low 64 bits hold the LVA of the component.
///
/// The following address ranges are reserved. Some are either explicitly or
/// implicitly protected by AGAS. The letter x represents a single-byte
/// wild card.
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
///     00000001000000010000000000000004
///         Address of the locality_namespace component on the bootstrap AGAS
///         locality.
///     00000001000000010000000000000005
///         Address of the root-CA component
///     xxxxxxxx000000010000000000000006
///         Address of the locality based sub-CA, xxxxxxxx is replaced with the
///         correct locality id
///
struct HPX_EXPORT primary_namespace
  : components::fixed_component_base<primary_namespace>
{
    // {{{ nested types
    typedef lcos::local::spinlock mutex_type;
    typedef components::fixed_component_base<primary_namespace> base_type;

    typedef boost::int32_t component_type;

    typedef std::pair<gva, naming::gid_type> gva_table_data_type;
    typedef std::map<naming::gid_type, gva_table_data_type> gva_table_type;
    typedef std::map<naming::gid_type, boost::int64_t> refcnt_table_type;

    typedef hpx::util::tuple<naming::gid_type, gva, naming::gid_type>
        resolved_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;

    gva_table_type gvas_;
    refcnt_table_type refcnts_;
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
    typedef std::map<
            naming::gid_type,
            hpx::util::tuple<bool, std::size_t, lcos::local::condition_variable_any>
        > migration_table_type;
#else
    typedef std::map<
            naming::gid_type,
            hpx::util::tuple<
                bool, std::size_t,
                boost::shared_ptr<lcos::local::condition_variable_any>
            >
        > migration_table_type;
#endif

    std::string instance_name_;
    naming::gid_type next_id_;      // next available gid
    naming::gid_type locality_;     // our locality id
    migration_table_type migrating_objects_;

    struct update_time_on_exit;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data
    {
    private:
        HPX_NON_COPYABLE(counter_data);

    public:
        struct api_counter_data
        {
            api_counter_data()
              : count_(0)
              , time_(0)
            {}

            boost::atomic<boost::int64_t> count_;
            boost::atomic<boost::int64_t> time_;
        };

        counter_data()
        {}

    public:
        // access current counter values
        boost::int64_t get_route_count(bool);
        boost::int64_t get_bind_gid_count(bool);
        boost::int64_t get_resolve_gid_count(bool);
        boost::int64_t get_unbind_gid_count(bool);
        boost::int64_t get_increment_credit_count(bool);
        boost::int64_t get_decrement_credit_count(bool);
        boost::int64_t get_allocate_count(bool);
        boost::int64_t get_begin_migration_count(bool);
        boost::int64_t get_end_migration_count(bool);
        boost::int64_t get_overall_count(bool);

        boost::int64_t get_route_time(bool);
        boost::int64_t get_bind_gid_time(bool);
        boost::int64_t get_resolve_gid_time(bool);
        boost::int64_t get_unbind_gid_time(bool);
        boost::int64_t get_increment_credit_time(bool);
        boost::int64_t get_decrement_credit_time(bool);
        boost::int64_t get_allocate_time(bool);
        boost::int64_t get_begin_migration_time(bool);
        boost::int64_t get_end_migration_time(bool);
        boost::int64_t get_overall_time(bool);

        // increment counter values
        void increment_route_count();
        void increment_bind_gid_count();
        void increment_resolve_gid_count();
        void increment_unbind_gid_count();
        void increment_increment_credit_count();
        void increment_decrement_credit_count();
        void increment_allocate_count();
        void increment_begin_migration_count();
        void increment_end_migration_count();

    private:
        friend struct update_time_on_exit;
        friend struct primary_namespace;

        api_counter_data route_;                // primary_ns_
        api_counter_data bind_gid_;             // primary_ns_bind_gid
        api_counter_data resolve_gid_;          // primary_ns_resolve_gid
        api_counter_data unbind_gid_;           // primary_ns_unbind_gid
        api_counter_data increment_credit_;     // primary_ns_increment_credit
        api_counter_data decrement_credit_;     // primary_ns_decrement_credit
        api_counter_data allocate_;             // primary_ns_allocate
        api_counter_data begin_migration_;      // primary_ns_begin_migration
        api_counter_data end_migration_;        // primary_ns_end_migration
    };
    counter_data counter_data_;

    struct update_time_on_exit
    {
        update_time_on_exit(boost::atomic<boost::int64_t>& t)
          : started_at_(hpx::util::high_resolution_clock::now())
          , t_(t)
        {}

        ~update_time_on_exit()
        {
            t_ += (hpx::util::high_resolution_clock::now() - started_at_);
        }

        boost::uint64_t started_at_;
        boost::atomic<boost::int64_t>& t_;
    };

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
    /// Dump the credit counts of all matching ranges. Expects that \p l
    /// is locked.
    void dump_refcnt_matches(
        refcnt_table_type::iterator lower_it
      , refcnt_table_type::iterator upper_it
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , std::unique_lock<mutex_type>& l
      , const char* func_name
        );
#endif

    // API
    response begin_migration(
        request const& req
      , error_code& ec);
    response end_migration(
        request const& req
      , error_code& ec);

    // helper function
    void wait_for_migration_locked(
        std::unique_lock<mutex_type>& l
      , naming::gid_type id
      , error_code& ec);

  public:
    primary_namespace()
      : base_type(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB)
      , locality_(naming::invalid_gid)
    {}

    void finalize();

    void set_local_locality(naming::gid_type const& g)
    {
        locality_ = g;
        next_id_ = naming::gid_type(g.get_msb() + 1, 0x1000);
    }

    response remote_service(
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
    std::vector<response> remote_bulk_service(
        std::vector<request> const& reqs
        )
    {
        return bulk_service(reqs, throws);
    }

    /// Maps \a service over \p reqs in parallel.
    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , error_code& ec
        );

    /// Register all performance counter types exposed by this component.
    static void register_counter_types(
        error_code& ec = throws
        );
    static void register_global_counter_types(
        error_code& ec = throws
        );

    void register_server_instance(
        char const* servicename
      , boost::uint32_t locality_id = naming::invalid_locality_id
      , error_code& ec = throws
        );

    void unregister_server_instance(
        error_code& ec = throws
        );

    response route(
        parcelset::parcel && p
        );

    response bind_gid(
        request const& req
      , error_code& ec = throws
        );

    response resolve_gid(
        request const& req
      , error_code& ec = throws
        );

    response unbind_gid(
        request const& req
      , error_code& ec = throws
        );

    response increment_credit(
        request const& req
      , error_code& ec = throws
        );

    response decrement_credit(
        request const& req
      , error_code& ec = throws
        );

    response allocate(
        request const& req
      , error_code& ec = throws
        );

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

  private:
    resolved_type resolve_gid_locked(
        std::unique_lock<mutex_type>& l
      , naming::gid_type const& gid
      , error_code& ec
        );

    void increment(
        naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t& credits
      , error_code& ec
        );

    ///////////////////////////////////////////////////////////////////////////
    struct free_entry
    {
        free_entry(agas::gva gva, naming::gid_type const& gid,
                naming::gid_type const& loc)
          : gva_(gva), gid_(gid), locality_(loc)
        {}

        agas::gva gva_;
        naming::gid_type gid_;
        naming::gid_type locality_;
    };

    void resolve_free_list(
        std::unique_lock<mutex_type>& l
      , std::list<refcnt_table_type::iterator> const& free_list
      , std::list<free_entry>& free_entry_list
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , error_code& ec
        );

    void decrement_sweep(
        std::list<free_entry>& free_list
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t credits
      , error_code& ec
        );

    void free_components_sync(
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

        // Pseudo-actions
      , namespace_route                         = primary_ns_route
      , namespace_bind_gid                      = primary_ns_bind_gid
      , namespace_resolve_gid                   = primary_ns_resolve_gid
      , namespace_unbind_gid                    = primary_ns_unbind_gid
      , namespace_increment_credit              = primary_ns_increment_credit
      , namespace_decrement_credit              = primary_ns_decrement_credit
      , namespace_allocate                      = primary_ns_allocate
      , namespace_begin_migration               = primary_ns_begin_migration
      , namespace_end_migration                 = primary_ns_end_migration
      , namespace_statistics_counter            = primary_ns_statistics_counter
    }; // }}}

    HPX_DEFINE_COMPONENT_ACTION(primary_namespace, remote_service, service_action);
    HPX_DEFINE_COMPONENT_ACTION(primary_namespace, remote_bulk_service,
        bulk_service_action);

    HPX_DEFINE_COMPONENT_ACTION(primary_namespace, route, route_action);

    static parcelset::policies::message_handler* get_message_handler(
        parcelset::parcelhandler* ph
      , parcelset::locality const& loc
      , parcelset::parcel const& p
        );

    static serialization::binary_filter* get_serialization_filter(
        parcelset::parcel const& p
        );
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::primary_namespace::service_action,
    primary_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::primary_namespace::bulk_service_action,
    primary_namespace_bulk_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::primary_namespace::route_action,
    primary_namespace_route_action)

namespace hpx { namespace traits
{
    // Parcel routing forwards the message handler request to the routed action
    template <>
    struct action_message_handler<agas::server::primary_namespace::route_action>
    {
        static parcelset::policies::message_handler* call(
            parcelset::parcelhandler* ph
          , parcelset::locality const& loc
          , parcelset::parcel const& p
            )
        {
            return agas::server::primary_namespace::get_message_handler(
                ph, loc, p);
        }
    };

    // Parcel routing forwards the binary filter request to the routed action
    template <>
    struct action_serialization_filter<
        agas::server::primary_namespace::route_action>
    {
        static serialization::binary_filter* call(parcelset::parcel const& p)
        {
            return agas::server::primary_namespace::get_serialization_filter(p);
        }
    };
}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

