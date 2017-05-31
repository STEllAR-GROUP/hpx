////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AGAS_LOCALITY_NAMESPACE_APR_04_2013_1107AM)
#define HPX_AGAS_LOCALITY_NAMESPACE_APR_04_2013_1107AM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/parcelset/locality.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <boost/atomic.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_locality_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_locality_namespace_id();

namespace server
{

// Base name used to register the component
char const* const locality_namespace_service_name = "locality/";

struct HPX_EXPORT locality_namespace
  : components::fixed_component_base<locality_namespace>
{
    // {{{ nested types
    typedef lcos::local::spinlock mutex_type;
    typedef components::fixed_component_base<locality_namespace> base_type;

    typedef std::int32_t component_type;

    // stores the locality endpoints, and number of OS-threads running on this locality
    typedef hpx::util::tuple<
        parcelset::endpoints_type, std::uint32_t>
    partition_type;

    typedef std::map<std::uint32_t, partition_type> partition_table_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;
    std::string instance_name_;

    partition_table_type partitions_;
    std::uint32_t prefix_counter_;
    primary_namespace* primary_;

    struct update_time_on_exit;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data
    {
    public:
        HPX_NON_COPYABLE(counter_data);

    public:
        typedef lcos::local::spinlock mutex_type;

        struct api_counter_data
        {
            api_counter_data()
              : count_(0)
              , time_(0)
            {}

            boost::atomic<std::int64_t> count_;
            boost::atomic<std::int64_t> time_;
        };

        counter_data()
        {}

    public:
        // access current counter values
        std::int64_t get_allocate_count(bool);
        std::int64_t get_resolve_locality_count(bool);
        std::int64_t get_free_count(bool);
        std::int64_t get_localities_count(bool);
        std::int64_t get_num_localities_count(bool);
        std::int64_t get_num_threads_count(bool);
        std::int64_t get_resolved_localities_count(bool);
        std::int64_t get_overall_count(bool);

        std::int64_t get_allocate_time(bool);
        std::int64_t get_resolve_locality_time(bool);
        std::int64_t get_free_time(bool);
        std::int64_t get_localities_time(bool);
        std::int64_t get_num_localities_time(bool);
        std::int64_t get_num_threads_time(bool);
        std::int64_t get_resolved_localities_time(bool);
        std::int64_t get_overall_time(bool);

        // increment counter values
        void increment_allocate_count();
        void increment_resolve_locality_count();
        void increment_free_count();
        void increment_localities_count();
        void increment_num_localities_count();
        void increment_num_threads_count();

    private:
        friend struct update_time_on_exit;
        friend struct locality_namespace;

        api_counter_data allocate_;             // locality_ns_allocate
        api_counter_data resolve_locality_;     // locality_ns_resolve_locality
        api_counter_data free_;                 // locality_ns_free
        api_counter_data localities_;           // locality_ns_localities
        api_counter_data num_localities_;       // locality_ns_num_localities
        api_counter_data num_threads_;          // locality_ns_num_threads
    };
    counter_data counter_data_;

  public:
    locality_namespace(primary_namespace* primary)
      : base_type(HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB)
      , prefix_counter_(HPX_AGAS_BOOTSTRAP_PREFIX)
      , primary_(primary)
    {}

    void finalize();

    // register all performance counter types exposed by this component
    static void register_counter_types(
        error_code& ec = throws
        );
    static void register_global_counter_types(
        error_code& ec = throws
        );

    void register_server_instance(
        char const* servicename
      , error_code& ec = throws
        );

    void unregister_server_instance(
        error_code& ec = throws
        );

    std::uint32_t allocate(
        parcelset::endpoints_type const& endpoints
      , std::uint64_t count
      , std::uint32_t num_threads
      , naming::gid_type suggested_prefix
        );

    parcelset::endpoints_type resolve_locality(
        naming::gid_type locality);

    void free(naming::gid_type locality);

    std::vector<std::uint32_t> localities();

    std::uint32_t get_num_localities();

    std::vector<std::uint32_t> get_num_threads();

    std::uint32_t get_num_overall_threads();

    naming::gid_type statistics_counter(std::string name);

  public:
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, allocate);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, free);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, localities);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, resolve_locality);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, get_num_localities);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, get_num_threads);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, get_num_overall_threads);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, statistics_counter);
};

}}}

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::allocate_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::allocate_action,
    locality_namespace_allocate_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::free_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::free_action,
    locality_namespace_allocate_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::localities_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::localities_action,
    locality_namespace_localities_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::resolve_locality_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::resolve_locality_action,
    locality_namespace_resolve_locality_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::get_num_localities_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::get_num_localities_action,
    locality_namespace_get_num_localities_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::get_num_threads_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::get_num_threads_action,
    locality_namespace_get_num_threads_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::get_num_overall_threads_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::get_num_overall_threads_action,
    locality_namespace_get_num_overall_threads_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::server::locality_namespace::statistics_counter_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::statistics_counter_action,
    locality_namespace_statistics_counter_action)

#include <hpx/config/warnings_suffix.hpp>

#endif

