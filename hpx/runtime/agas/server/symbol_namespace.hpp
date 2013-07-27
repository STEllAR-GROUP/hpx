////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB)
#define HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <map>

#include <boost/format.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_symbol_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_symbol_namespace_id();

namespace server
{

// Base name used to register the component
char const* const symbol_namespace_service_name = "symbol/";

struct HPX_EXPORT symbol_namespace 
  : components::fixed_component_base<symbol_namespace>
{
    // {{{ nested types
    typedef lcos::local::spinlock mutex_type;
    typedef components::fixed_component_base<symbol_namespace> base_type;

    // FIXME: This signature should use id_type, not gid_type
    typedef hpx::util::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;

    typedef std::map<std::string, naming::gid_type> gid_table_type;
    // }}}

  private:
    mutex_type mutex_;
    gid_table_type gids_;
    std::string instance_name_;

    struct update_time_on_exit;

    // data structure holding all counters for the omponent_namespace component
    struct counter_data :  boost::noncopyable
    {
        typedef lcos::local::spinlock mutex_type;

        struct api_counter_data
        {
            api_counter_data()
                : count_(0)
                , time_(0)
            {}

            boost::int64_t count_;
            boost::int64_t time_;
        };

        counter_data()
        {}

    public:
        // access current counter values
        boost::int64_t get_bind_count(bool);
        boost::int64_t get_resolve_count(bool);
        boost::int64_t get_unbind_count(bool);
        boost::int64_t get_iterate_names_count(bool);
        boost::int64_t get_overall_count(bool);

        boost::int64_t get_bind_time(bool);
        boost::int64_t get_resolve_time(bool);
        boost::int64_t get_unbind_time(bool);
        boost::int64_t get_iterate_names_time(bool);
        boost::int64_t get_overall_time(bool);

        // increment counter values
        void increment_bind_count();
        void increment_resolve_count();
        void increment_unbind_count();
        void increment_iterate_names_count();

    private:
        friend struct update_time_on_exit;
        friend struct symbol_namespace;

        mutable mutex_type mtx_;
        api_counter_data bind_;               // symbol_ns_bind
        api_counter_data resolve_;            // symbol_ns_resolve
        api_counter_data unbind_;             // symbol_ns_unbind
        api_counter_data iterate_names_;      // symbol_ns_iterate_names
    };
    counter_data counter_data_;

    struct update_time_on_exit
    {
        update_time_on_exit(counter_data& data, boost::int64_t& t)
          : started_at_(hpx::util::high_resolution_clock::now())
          , data_(data)
          , t_(t)
        {}

        ~update_time_on_exit()
        {
            counter_data::mutex_type::scoped_lock l(data_.mtx_);
            t_ += (hpx::util::high_resolution_clock::now() - started_at_);
        }

        boost::uint64_t started_at_;
        symbol_namespace::counter_data& data_;
        boost::int64_t& t_;
    };

  public:
    symbol_namespace()
      : base_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB)
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

    void register_server_instance(
        char const* servicename
      , error_code& ec = throws
        );

    void unregister_server_instance(
        error_code& ec = throws
        );

    response bind(
        request const& req
      , error_code& ec = throws
        );

    response resolve(
        request const& req
      , error_code& ec = throws
        );

    response unbind(
        request const& req
      , error_code& ec = throws
        );

    response iterate(
        request const& req
      , error_code& ec = throws
        );

    response statistics_counter(
        request const& req
      , error_code& ec = throws
        );

    enum actions
    { // {{{ action enum
        // Actual actions
        namespace_service            = symbol_ns_service
      , namespace_bulk_service       = symbol_ns_bulk_service

        // Pseudo-actions
      , namespace_bind               = symbol_ns_bind
      , namespace_resolve            = symbol_ns_resolve
      , namespace_unbind             = symbol_ns_unbind
      , namespace_iterate_names      = symbol_ns_iterate_names
      , namespace_statistics_counter = symbol_ns_statistics_counter
    }; // }}}

    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, remote_service, service_action);
    HPX_DEFINE_COMPONENT_ACTION(symbol_namespace, remote_bulk_service, bulk_service_action);

    /// This is the default hook implementation for decorate_action which 
    /// does no hooking at all.
    static HPX_STD_FUNCTION<threads::thread_function_type> 
    wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
        naming::address::address_type)
    {
        return boost::move(f);
    }
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::bulk_service_action,
    symbol_namespace_bulk_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::server::symbol_namespace::service_action,
    symbol_namespace_service_action)

#endif // HPX_D69CE952_C5D9_4545_B83E_BA3DCFD812EB

