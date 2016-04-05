////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AGAS_LOCALITY_NAMESPACE_APR_04_2013_1107AM)
#define HPX_AGAS_LOCALITY_NAMESPACE_APR_04_2013_1107AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <map>

#include <boost/format.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/atomic.hpp>

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

    typedef boost::int32_t component_type;

    // stores the locality endpoints, and number of OS-threads running on this locality
    typedef boost::fusion::vector2<
        parcelset::endpoints_type, boost::uint32_t>
    partition_type;

    typedef std::map<boost::uint32_t, partition_type> partition_table_type;
    // }}}

  private:
    // REVIEW: Separate mutexes might reduce contention here. This has to be
    // investigated carefully.
    mutex_type mutex_;
    std::string instance_name_;

    partition_table_type partitions_;
    boost::uint32_t prefix_counter_;
    primary_namespace* primary_;

    struct update_time_on_exit;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data
    {
    private:
        HPX_NON_COPYABLE(counter_data);

    public:
        typedef lcos::local::spinlock mutex_type;

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
        boost::int64_t get_allocate_count(bool);
        boost::int64_t get_resolve_locality_count(bool);
        boost::int64_t get_free_count(bool);
        boost::int64_t get_localities_count(bool);
        boost::int64_t get_num_localities_count(bool);
        boost::int64_t get_num_threads_count(bool);
        boost::int64_t get_resolved_localities_count(bool);
        boost::int64_t get_overall_count(bool);

        boost::int64_t get_allocate_time(bool);
        boost::int64_t get_resolve_locality_time(bool);
        boost::int64_t get_free_time(bool);
        boost::int64_t get_localities_time(bool);
        boost::int64_t get_num_localities_time(bool);
        boost::int64_t get_num_threads_time(bool);
        boost::int64_t get_resolved_localities_time(bool);
        boost::int64_t get_overall_time(bool);

        // increment counter values
        void increment_allocate_count();
        void increment_resolve_locality_count();
        void increment_free_count();
        void increment_localities_count();
        void increment_num_localities_count();
        void increment_num_threads_count();
        void increment_resolved_localities_count();

    private:
        friend struct update_time_on_exit;
        friend struct locality_namespace;

        api_counter_data allocate_;             // locality_ns_allocate
        api_counter_data resolve_locality_;     // locality_ns_resolve_locality
        api_counter_data free_;                 // locality_ns_free
        api_counter_data localities_;           // locality_ns_localities
        api_counter_data num_localities_;       // locality_ns_num_localities
        api_counter_data num_threads_;          // locality_ns_num_threads
        api_counter_data resolved_localities_;  // locality_ns_resolved_localities
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

  public:
    locality_namespace(primary_namespace* primary)
      : base_type(HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB)
      , prefix_counter_(HPX_AGAS_BOOTSTRAP_PREFIX)
      , primary_(primary)
    {}

    void finalize();

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

    response allocate(
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

    response localities(
        request const& req
      , error_code& ec = throws
        );

    response resolved_localities(
        request const& req
      , error_code& ec = throws
        );

    response get_num_localities(
        request const& req
      , error_code& ec = throws
        );

    response get_num_threads(
        request const& req
      , error_code& ec = throws
        );

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

  public:
    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service                       = locality_ns_service
      , namespace_bulk_service                  = locality_ns_bulk_service

        // Pseudo-actions
      , namespace_allocate                      = locality_ns_allocate
      , namespace_resolve_locality              = locality_ns_resolve_locality
      , namespace_free                          = locality_ns_free
      , namespace_localities                    = locality_ns_localities
      , namespace_num_localities                = locality_ns_num_localities
      , namespace_num_threads                   = locality_ns_num_threads
      , namespace_statistics_counter            = locality_ns_statistics_counter
      , namespace_resolved_localities           = locality_ns_resolved_localities
    }; // }}}

    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, remote_service, service_action);
    HPX_DEFINE_COMPONENT_ACTION(locality_namespace, remote_bulk_service,
        bulk_service_action);
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::service_action,
    locality_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::locality_namespace::bulk_service_action,
    locality_namespace_bulk_service_action)

#endif

