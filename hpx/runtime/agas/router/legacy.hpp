////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_15D904C7_CD18_46E1_A54A_65059966A34F)
#define HPX_15D904C7_CD18_46E1_A54A_65059966A34F

#include <hpx/config.hpp>

#include <vector>

#include <boost/atomic.hpp>
#include <boost/lockfree/fifo.hpp>
#include <boost/make_shared.hpp>
#include <boost/cache/entries/lfu_entry.hpp>
#include <boost/cache/local_cache.hpp>
#include <boost/cstdint.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/noncopyable.hpp>

#include <hpx/exception.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/lcos/local_counting_semaphore.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/agas/router/big_boot_barrier.hpp>
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/runtime_configuration.hpp>

// TODO: use the response pools for GID range allocation and bind requests.
// TODO: pass error codes once they're implemented in AGAS.

namespace hpx { namespace agas
{

struct HPX_EXPORT legacy_router : boost::noncopyable
{
    // {{{ types 
    typedef primary_namespace<tag::database::stdmap, tag::network::tcpip>
        primary_namespace_type;

    typedef component_namespace<tag::database::stdmap, tag::network::tcpip>
        component_namespace_type;

    typedef symbol_namespace<tag::database::stdmap, tag::network::tcpip>
        symbol_namespace_type;

    typedef primary_namespace_type::server_type
        primary_namespace_server_type; 

    typedef component_namespace_type::server_type
        component_namespace_server_type; 

    typedef symbol_namespace_type::server_type
        symbol_namespace_server_type; 

    typedef component_namespace_type::component_id_type component_id_type;

    typedef response<tag::network::tcpip> response_type; 

    typedef primary_namespace_type::gva_type gva_type;
    typedef primary_namespace_type::count_type count_type;
    typedef primary_namespace_type::offset_type offset_type;
    typedef primary_namespace_type::endpoint_type endpoint_type;
    typedef component_namespace_type::prefix_type prefix_type;

    typedef symbol_namespace_type::iterate_function_type
        iterateids_function_type;

    typedef hpx::lcos::mutex cache_mutex_type;

    typedef boost::atomic<boost::uint32_t> console_cache_type;
    // }}}

    // {{{ gva cache
    struct gva_cache_key
    { // {{{ gva_cache_key implementation
        gva_cache_key()
          : id(), count(0)
        {}

        explicit gva_cache_key(naming::gid_type const& id_,
                               count_type count_ = 1)
          : id(id_), count(count_)
        {}

        naming::gid_type id;
        count_type count;

        friend bool
        operator<(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id + (lhs.count - 1)) < rhs.id; }

        friend bool
        operator==(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id == rhs.id) && (lhs.count == rhs.count); }
    }; // }}}
    
    struct gva_erase_policy
    { // {{{ gva_erase_policy implementation
        gva_erase_policy(naming::gid_type const& id, count_type count)
          : entry(id, count)
        {}

        typedef std::pair<
            gva_cache_key, boost::cache::entries::lfu_entry<gva_type>
        > entry_type;

        bool operator()(entry_type const& p) const
        { return p.first == entry; }

        gva_cache_key entry;
    }; // }}}

    typedef boost::cache::entries::lfu_entry<gva_type> gva_entry_type;

    typedef boost::cache::local_cache<
        gva_cache_key, gva_entry_type, 
        std::less<gva_entry_type>,
        boost::cache::policies::always<gva_entry_type>,
        std::map<gva_cache_key, gva_entry_type>
    > gva_cache_type;
    // }}}

    // {{{ locality cache 
    typedef boost::cache::entries::lfu_entry<naming::gid_type>
        locality_entry_type;

    typedef boost::cache::local_cache<naming::locality, locality_entry_type>
        locality_cache_type;
    // }}}

    // {{{ future freelists
    typedef boost::lockfree::fifo<
        lcos::eager_future<
            primary_namespace_server_type::bind_locality_action,
            response_type
        >*
    > allocate_response_pool_type;

    typedef boost::lockfree::fifo<
        lcos::eager_future<
            primary_namespace_server_type::bind_gid_action,
            response_type
        >*
    > bind_response_pool_type;
    // }}}

    struct bootstrap_data_type
    { // {{{
        primary_namespace_server_type primary_ns_server;
        component_namespace_server_type component_ns_server;
        symbol_namespace_server_type symbol_ns_server;
    }; // }}}

    struct hosted_data_type
    { // {{{
        hosted_data_type()
          : console_cache_(0)
        {}

        primary_namespace_type primary_ns_;
        component_namespace_type component_ns_;
        symbol_namespace_type symbol_ns_;

        cache_mutex_type gva_cache_mtx_;
        gva_cache_type gva_cache_;

        cache_mutex_type locality_cache_mtx_;
        locality_cache_type locality_cache_;

        console_cache_type console_cache_;

        hpx::lcos::local_counting_semaphore allocate_response_sema_;
        allocate_response_pool_type allocate_response_pool_;

        hpx::lcos::local_counting_semaphore bind_response_sema_;
        bind_response_pool_type bind_response_pool_;

        naming::address primary_ns_addr_;
        naming::address component_ns_addr_;
        naming::address symbol_ns_addr_;
    }; // }}}

    const router_mode router_type;
    const runtime_mode runtime_type;

    boost::shared_ptr<bootstrap_data_type> bootstrap;
    boost::shared_ptr<hosted_data_type> hosted;

    atomic_state state_;
    naming::gid_type prefix_;

    legacy_router(
        parcelset::parcelport& pp 
      , util::runtime_configuration const& ini_
      , runtime_mode runtime_type_
    );

    ~legacy_router()
    { destroy_big_boot_barrier(); }

    void launch_bootstrap(
        parcelset::parcelport& pp 
      , util::runtime_configuration const& ini_
    );

    void launch_hosted(
        parcelset::parcelport& pp 
      , util::runtime_configuration const& ini_
    );

    state status() const
    {
        if (!hosted && !bootstrap)
            return stopping;
        else
            return state_.load();
    }
    
    void status(state new_state) 
    { state_.store(new_state); }

    naming::gid_type const& local_prefix() const
    {
        BOOST_ASSERT(prefix_ != naming::invalid_gid);
        return prefix_;
    }

    void local_prefix(naming::gid_type const& g)
    { prefix_ = g; }

    bool is_bootstrap() const
    { return router_type == router_mode_bootstrap; } 

    bool is_console() const
    { return runtime_type == runtime_mode_console; }
 
    bool is_smp_mode() const
    { return false; } 

    bool get_prefix(naming::locality const& l, naming::gid_type& prefix,
                    bool self = true, bool try_cache = true,
                    error_code& ec = throws);
    
    bool get_prefix_cached(naming::locality const& l, naming::gid_type& prefix,
                           bool self = true, error_code& ec = throws);

    bool remove_prefix(naming::locality const& l, error_code& ec = throws);

    bool get_console_prefix(naming::gid_type& prefix,
                            bool try_cache = true, error_code& ec = throws);

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      components::component_type type, error_code& ec = throws);

    // forwarder
    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      error_code& ec = throws) 
    { return get_prefixes(prefixes, components::component_invalid, ec); }

    components::component_type
    get_component_id(std::string const& name, error_code& ec = throws);

    components::component_type
    register_factory(naming::gid_type const& prefix, std::string const& name,
                     error_code& ec = throws);

    bool get_id_range(naming::locality const& l, count_type count, 
                      naming::gid_type& lower_bound,
                      naming::gid_type& upper_bound, error_code& ec = throws);

    // forwarder
    bool bind(naming::gid_type const& id, naming::address const& addr,
              error_code& ec = throws) 
    { return bind_range(id, 1, addr, 0, ec); }

    bool bind_range(naming::gid_type const& lower_id, count_type count, 
                    naming::address const& baseaddr, offset_type offset,
                    error_code& ec = throws);

    count_type
    incref(naming::gid_type const& id, count_type credits = 1,
           error_code& ec = throws);

    count_type
    decref(naming::gid_type const& id, components::component_type& t,
           count_type credits = 1, error_code& ec = throws);

    // forwarder
    bool unbind(naming::gid_type const& id, error_code& ec = throws) 
    { return unbind_range(id, 1, ec); } 
        
    // forwarder
    bool unbind(naming::gid_type const& id, naming::address& addr,
                error_code& ec = throws) 
    { return unbind_range(id, 1, addr, ec); }

    // forwarder
    bool unbind_range(naming::gid_type const& lower_id, count_type count,
                      error_code& ec = throws) 
    {
        naming::address addr; 
        return unbind_range(lower_id, count, addr, ec);
    } 

    bool unbind_range(naming::gid_type const& lower_id, count_type count, 
                      naming::address& addr, error_code& ec = throws);

    bool resolve(naming::gid_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws);

    // forwarder
    bool resolve(naming::id_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { return resolve(id.get_gid(), addr, try_cache, ec); }

    bool resolve_cached(naming::gid_type const& id, naming::address& addr,
                        error_code& ec = throws);

    bool registerid(std::string const& name, naming::gid_type const& id,
                    error_code& ec = throws);

    bool unregisterid(std::string const& name, error_code& ec = throws);

    bool queryid(std::string const& ns_name, naming::gid_type& id,
                 error_code& ec = throws);

    bool iterateids(iterateids_function_type const& f);
};

}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

