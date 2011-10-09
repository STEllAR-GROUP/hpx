////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/logging.hpp>

namespace hpx { namespace agas
{

addressing_service::addressing_service(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
    )
  : service_type(ini_.get_agas_service_mode())
  , runtime_type(runtime_type_)
  , here_(get_runtime().here())
  , rts_lva_(get_runtime().get_runtime_support_lva())
  , state_(starting)
  , prefix_()
{ // {{{
    // boot the parcel port
    pp.run(false);

    create_big_boot_barrier(pp, ini_, runtime_type_);

    if (service_type == service_mode_bootstrap)
        launch_bootstrap(pp, ini_);
    else
        launch_hosted(pp, ini_);
} // }}}

void addressing_service::launch_bootstrap(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
    )
{ // {{{
    bootstrap = boost::make_shared<bootstrap_data_type>();
    
    const naming::locality ep = ini_.get_agas_locality();
    const naming::gid_type here
        = naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX);

    const naming::gid_type primary_gid = bootstrap_primary_namespace_gid();
    const naming::gid_type component_gid = bootstrap_component_namespace_gid();
    const naming::gid_type symbol_gid = bootstrap_symbol_namespace_gid();

    gva primary_gva(ep,
        primary_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->primary_ns_server));
    gva component_gva(ep,
        component_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->component_ns_server));
    gva symbol_gva(ep,
        symbol_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->symbol_ns_server));

    local_prefix(here);

    request reqs[] =
    {
        request(primary_ns_bind_locality, ep, 3)
      , request(primary_ns_bind_gid, primary_gid, primary_gva)
      , request(primary_ns_bind_gid, component_gid, component_gva)
      , request(primary_ns_bind_gid, symbol_gid, symbol_gva)
    };

    for (std::size_t i = 0; i < (sizeof(reqs) / sizeof(request)); ++i)
        bootstrap->primary_ns_server.service(reqs[i]);

    bootstrap->symbol_ns_server.service(
        request(symbol_ns_bind, "/locality(agas#0)", here));

    if (runtime_type == runtime_mode_console)
        bootstrap->symbol_ns_server.service(
            request(symbol_ns_bind, "/locality(console)", here));

    naming::gid_type lower, upper;
    get_id_range(ep, HPX_INITIAL_GID_RANGE, lower, upper);
    get_runtime().get_id_pool().set_range(lower, upper);

    get_big_boot_barrier().wait();

    state_.store(running);
} // }}}

void addressing_service::launch_hosted(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
    )
{ // {{{
    hosted = boost::make_shared<hosted_data_type>();

    hosted->gva_cache_.reserve(ini_.get_agas_gva_cache_size());

    get_big_boot_barrier().wait();

    state_.store(running);
} // }}}

response addressing_service::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    if (req.get_action_code() & primary_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->primary_ns_server.service(req, ec);
        else
            return hosted->primary_ns_.service(req, ec);
    }        

    else if (req.get_action_code() & component_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->component_ns_server.service(req, ec);
        else
            return hosted->component_ns_.service(req, ec);
    }        

    else if (req.get_action_code() & symbol_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->symbol_ns_server.service(req, ec);
        else
            return hosted->symbol_ns_.service(req, ec);
    }        

    HPX_THROWS_IF(ec, bad_action_code
        , "addressing_service::service"
        , "invalid action code encountered in request")
    return response();
} // }}}

bool addressing_service::register_locality(
    naming::locality const& ep
  , naming::gid_type& prefix
  , error_code& ec
    )
{ // {{{
    try {
        request req(primary_ns_bind_locality, ep, 0);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
            rep = hosted->primary_ns_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return false;
   
        prefix = naming::get_gid_from_prefix(rep.get_prefix());
 
        return true;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::bind_locality", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

// TODO: We need to ensure that the locality isn't unbound while it still holds
// referenced objects.
bool addressing_service::unregister_locality(
    naming::locality const& ep
  , error_code& ec
    )
{ // {{{
    try {
        request req(primary_ns_unbind_locality, ep);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
            rep = hosted->primary_ns_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return false;
    
        return true;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::unbind_locality", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::get_console_prefix(
    naming::gid_type& prefix
  , bool try_cache
  , error_code& ec
    )
{ // {{{
    try {
        if (status() != running)
        {
            if (&ec != &throws)
                ec = make_success_code();
            return false;
        }

        if (is_console())
        {
            prefix = local_prefix();
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }
     
        if (try_cache && !is_bootstrap())
        {
            if (hosted->console_cache_)
            {
                prefix = naming::get_gid_from_prefix(hosted->console_cache_);
                if (&ec != &throws)
                    ec = make_success_code();
                return true;
            }
        }
    
        if (is_bootstrap()) {
            request req(symbol_ns_resolve, "/locality(console)");
            response rep = bootstrap->symbol_ns_server.service(req, ec);
    
            if (!ec &&
                (rep.get_gid() != naming::invalid_gid) &&
                (rep.get_status() == success))
            {
                prefix = rep.get_gid();
                return true;
            }
        }
    
        else {
            request req(symbol_ns_resolve, "/locality(console)");
            response rep = hosted->symbol_ns_.service(req, ec);
    
            if (!ec &&
                (rep.get_gid() != naming::invalid_gid) &&
                (rep.get_status() == success))
            {
                prefix = rep.get_gid();
                hosted->console_cache_.store
                    (naming::get_prefix_from_gid(prefix));
                return true;
            }
        }
    
        return false;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::get_console_prefix", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::get_prefixes(
    std::vector<naming::gid_type>& prefixes
  , components::component_type type
  , error_code& ec 
    )
{ // {{{ get_prefixes implementation
    try {
        if (type != components::component_invalid)
        {
            request req(component_ns_resolve_id, type);
            response rep;
   
            if (is_bootstrap())
                rep = bootstrap->component_ns_server.service(req, ec);
            else
                rep = hosted->component_ns_.service(req, ec);

            if (ec || (success != rep.get_status()))
                return false;
    
            const std::vector<boost::uint32_t> p = rep.get_localities();
    
            if (!p.size())
                return false;
    
            for (boost::uint64_t i = 0; i < p.size(); ++i) 
                prefixes.push_back(naming::get_gid_from_prefix(p[i]));
    
            return true; 
        }
    
        else
        {
            request req(primary_ns_localities);
            response rep;
    
            if (is_bootstrap())
                rep = bootstrap->primary_ns_server.service(req, ec);
            else
                rep = hosted->primary_ns_.service(req, ec);
           
            if (ec || (success != rep.get_status()))
                return false;

            const std::vector<boost::uint32_t> p = rep.get_localities();
    
            if (!p.size())
                return false;
    
            for (boost::uint64_t i = 0; i < p.size(); ++i) 
                prefixes.push_back(naming::get_gid_from_prefix(p[i]));
        
            return true;
        }
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::get_prefixes", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}} 

components::component_type addressing_service::get_component_id(
    std::string const& name
  , error_code& ec 
    )
{ /// {{{
    try {
        request req(component_ns_bind_name, name); 
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->component_ns_server.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return components::component_invalid;
    
        return (components::component_type) rep.get_component_type();
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::get_component_id", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return components::component_invalid;
    }
} // }}} 

components::component_type addressing_service::register_factory(
    boost::uint32_t prefix
  , std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        request req(component_ns_bind_prefix, name, prefix); 
        response rep;

        if (is_bootstrap())
            rep = bootstrap->component_ns_server.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, ec);
        
        if (ec || (success != rep.get_status()))
            return components::component_invalid;

        return (components::component_type) rep.get_component_type();
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(),
                "addressing_service::register_factory", e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return components::component_invalid;
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
struct lock_semaphore
{
    lock_semaphore(lcos::local_counting_semaphore& sem) 
      : sem_(sem)
    {
        // this needs to be invoked from a px-thread
        BOOST_ASSERT(NULL != threads::get_self_ptr());
        sem_.wait(1);
    }

    ~lock_semaphore() 
    {
        sem_.signal(1);
    }

    lcos::local_counting_semaphore& sem_;
};

template <typename Pool, typename Future>
struct checkout_future
{
    checkout_future(Pool& pool, Future*& future) 
      : result_ok_(false), pool_(pool), future_(future)
    {
        pool_.dequeue(&future_);
        BOOST_ASSERT(future_);    
        
        future_->reset();     // reset the future
    }
    ~checkout_future()
    {
        // return the future to the pool
        if (result_ok_) 
            pool_.enqueue(future_);
    }

    Future* operator->() { return future_; }
    void set_ok() { result_ok_ = true; }

private:
    bool result_ok_;
    Pool& pool_;
    Future*& future_;
};

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::get_id_range(
    naming::locality const& ep
  , boost::uint64_t count
  , naming::gid_type& lower_bound
  , naming::gid_type& upper_bound
  , error_code& ec
    )
{ // {{{ get_id_range implementation
    typedef lcos::eager_future<
        primary_namespace_server_type::service_action,
        response
    > allocate_response_future_type;

    allocate_response_future_type* f = 0;

    try {
        request req(primary_ns_bind_locality, ep, count);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
 
        else
        {
            // WARNING: this deadlocks if AGAS is unresponsive and all response
            // futures are checked out and pending.
    
            // wait for the semaphore to become available 
            lock_semaphore lock(hosted->allocate_response_sema_);

            // get a future
            typedef checkout_future<allocate_response_pool_type, 
                allocate_response_future_type> checkout_future_type;
            checkout_future_type cf(hosted->allocate_response_pool_, f);

            // execute the action (synchronously)
            cf->apply(
                naming::id_type(primary_namespace_server_type::fixed_gid()
                              , naming::id_type::unmanaged),
                req);
            rep = cf->get(ec);

            cf.set_ok();
        }

        const error s = rep.get_status();

        if (ec || (success != s && repeated_request != s))
            return false;
    
        lower_bound = rep.get_lower_bound(); 
        upper_bound = rep.get_upper_bound();

        return repeated_request != s;
    }
    catch (hpx::exception const& e) {
        // Replace the future in the pool. To be able to return the future to
        // the pool, we'd have to ensure that all threads (pending, suspended,
        // active, or in flight) that might read/write from it are aborted.
        // There's no guarantee that the future isn't corrupted in some other
        // way, and the aforementioned code would be lengthy, and would have to
        // be meticulously exception-free. So, for now, we just allocate a new
        // future for the pool, and let the old future stay in memory.
        if (!is_bootstrap() && f) 
        {
            hosted->allocate_response_pool_.enqueue 
                (new allocate_response_future_type);
            f->invalidate(boost::current_exception());
        }

        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error()
              , "addressing_service::get_id_range" 
              , e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::bind_range(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address const& baseaddr
  , boost::uint64_t offset
  , error_code& ec
    ) 
{ // {{{ bind_range implementation
    typedef lcos::eager_future<
        primary_namespace_server_type::service_action,
        response
    > bind_response_future_type;

    bind_response_future_type* f = 0;
    
    try {
        naming::locality const& ep = baseaddr.locality_;
       
        // Create a global virtual address from the legacy calling convention
        // parameters.
        const gva g(ep, baseaddr.type_, count, baseaddr.address_, offset);
        
        request req(primary_ns_bind_gid, lower_id, g);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
    
        else
        {
            // WARNING: this deadlocks if AGAS is unresponsive and all response
            // futures are checked out and pending.

            // wait for the semaphore to become available 
            lock_semaphore lock(hosted->bind_response_sema_);

            // get a future
            typedef checkout_future<bind_response_pool_type, 
                bind_response_future_type> checkout_future_type;
            checkout_future_type cf(hosted->bind_response_pool_, f);
    
            // execute the action (synchronously)
            cf->apply(
                naming::id_type(primary_namespace_server_type::fixed_gid()
                              , naming::id_type::unmanaged),
                req);
            rep = cf->get(ec);

            cf.set_ok();
        }
    
        const error s = rep.get_status();

        if (ec || (success != s && repeated_request != s))
            return false;
    
        if (!is_bootstrap())
        { 
            cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            gva_cache_key key(lower_id, count);
            hosted->gva_cache_.insert(key, g);
        }

        return repeated_request != s;
    }
    catch (hpx::exception const& e) {
        // Replace the future in the pool. See get_id_range above for an
        // explanation. 
        if (!is_bootstrap() && f) 
        {
            hosted->bind_response_pool_.enqueue(new bind_response_future_type);
            f->invalidate(boost::current_exception());
        }

        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error()
              , "addressing_service::bind_range" 
              , e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::unbind_range(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address& addr
  , error_code& ec 
    )
{ // {{{ unbind_range implementation
    try {
        request req(primary_ns_unbind_gid, lower_id, count);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
            rep = hosted->primary_ns_.service(req, ec);

        if (ec)
            return false;
 
        if (!is_bootstrap() && (success == rep.get_status()))
        {
            // I'm afraid that this will break the first form of paged caching,
            // so it's commented out for now.
            //cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            //gva_erase_policy ep(lower_id, count);
            //hosted->gva_cache_.erase(ep);
            addr.locality_ = rep.get_gva().endpoint;
            addr.type_ = rep.get_gva().type;
            addr.address_ = rep.get_gva().lva();
        }
    
        return success == rep.get_status(); 
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error()
              , "addressing_service::unbind_range" 
              , e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::resolve(
    naming::gid_type const& id
  , naming::address& addr
  , bool try_cache
  , error_code& ec
    )
{ // {{{ resolve implementation
    try {
        // {{{ special cases

        // LVA-encoded GIDs 
        if (naming::strip_credit_from_gid(id.get_msb())
            == local_prefix().get_msb())  
        {
            addr.locality_ = here_;  
    
            // An LSB of 0 references the runtime support component
            if (0 == id.get_lsb() || id.get_lsb() == rts_lva_)
            {
                addr.type_ = components::component_runtime_support;
                addr.address_ = rts_lva_;
            }
    
            else
            {
                addr.type_ = components::component_memory;
                addr.address_ = id.get_lsb();
            }
    
            if (&ec != &throws)
                ec = make_success_code();
    
            return true;
        }

        // authoritative AGAS component address resolution
        else if (id == primary_namespace_server_type::fixed_gid())
        {
            addr = hosted->primary_ns_addr_;
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }
    
        else if (id == component_namespace_server_type::fixed_gid())
        {
            addr = hosted->component_ns_addr_;
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }
    
        else if (id == symbol_namespace_server_type::fixed_gid())
        {
            addr = hosted->symbol_ns_addr_;
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }
        // }}}

        // Try the cache if applicable.
        if (!is_bootstrap() && try_cache)
        {
            if (resolve_cached(id, addr, ec))
                return true;

            if (ec)
                return false;
        }
 
        request req(primary_ns_page_fault, id);
        response rep; 
    
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
        {
            LHPX_(info, "  [AC] ") <<
                (boost::format("soft page fault, faulting address %1%") % id);
            rep = hosted->primary_ns_.service(req, ec);
        } 

        if (ec || (success != rep.get_status()))
            return false;

        // Resolve the page to the real resolved address (which is just a page
        // with as fully resolved LVA and an offset of zero).
        const gva g = rep.get_gva().resolve(id, rep.get_base_gid());

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva();
    
        if (!is_bootstrap())
        {
            // Put the page into the cache.
            cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            gva_cache_key key(rep.get_base_gid(), rep.get_gva().count);
            hosted->gva_cache_.insert(key, rep.get_gva());
        }

        return true;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::resolve", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::resolve_cached(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ resolve_cached implementation
    if (is_bootstrap())
        return resolve(id, addr, false, ec);

    // {{{ special cases

    // LVA-encoded GIDs 
    if (naming::strip_credit_from_gid(id.get_msb()) == local_prefix().get_msb())  
    {
        addr.locality_ = here_;  

        // An LSB of 0 references the runtime support component
        if (0 == id.get_lsb() || id.get_lsb() == rts_lva_)
        {
            addr.type_ = components::component_runtime_support;
            addr.address_ = rts_lva_;
        }

        else
        {
            addr.type_ = components::component_memory;
            addr.address_ = id.get_lsb();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    // authoritative AGAS component address resolution
    else if (id == primary_namespace_server_type::fixed_gid())
    {
        addr = hosted->primary_ns_addr_;
        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }
    
    else if (id == component_namespace_server_type::fixed_gid())
    {
        addr = hosted->component_ns_addr_;
        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }
    
    else if (id == symbol_namespace_server_type::fixed_gid())
    {
        addr = hosted->symbol_ns_addr_;
        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }
    // }}}

    // first look up the requested item in the cache
    gva_cache_key k(id);
    cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
    gva_cache_key idbase;
    gva_cache_type::entry_type e;

    // Check if the entry is currently in the cache
    if (hosted->gva_cache_.get_entry(k, idbase, e))
    {
        const boost::uint64_t id_msb
            = naming::strip_credit_from_gid(id.get_msb());

        if (HPX_UNLIKELY(id_msb != idbase.get_gid().get_msb()))
        {
            HPX_THROWS_IF(ec, invalid_page_fault
              , "addressing_service::resolve_cached" 
              , "bad page in cache, MSBs of GID base and GID do not match");
            return false;
        }

        gva const& g = e.get();

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva(id, idbase.get_gid());

        if (&ec != &throws)
            ec = make_success_code();

        LHPX_(debug, "  [AC] ") <<
            ( boost::format("cache hit for address %1% (base %2%)")
            % id % idbase.get_gid());
    
        return true;
    }

    if (&ec != &throws)
        ec = make_success_code();

    LHPX_(debug, "  [AC] ") <<
        (boost::format("cache miss for address %1%") % id);

    return false;
} // }}}

boost::uint64_t addressing_service::incref(
    naming::gid_type const& id
  , boost::uint64_t credits
  , error_code& ec 
    )
{ // {{{ incref implementation
    try {
        request req(primary_ns_increment, id, credits);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
            rep = hosted->primary_ns_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return 0;

        return rep.get_count();
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::incref", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return 0;
    }
} // }}}

boost::uint64_t addressing_service::decref(
    naming::gid_type const& id
  , components::component_type& t
  , boost::uint64_t credits
  , error_code& ec
    )
{ // {{{ decref implementation
    try {
        request req(primary_ns_decrement, id, credits);
        response rep;
        
        if (is_bootstrap())
            rep = bootstrap->primary_ns_server.service(req, ec);
        else
            rep = hosted->primary_ns_.service(req, ec);
   
        if (ec || (success != rep.get_status()))
            return 0;
 
        if (0 == rep.get_count())
        {
            t = (components::component_type) rep.get_component_type();

            if (HPX_UNLIKELY(components::component_invalid != t))
            {
                HPX_THROWS_IF(ec, bad_component_type
                  , "addressing_service::decref"
                  , boost::str(boost::format(
                    "received invalid component type when decrementing last "
                    "GID to 0, gid(%1%)")
                    % id));
                return 0;
            }
        }
    
        return rep.get_count();
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::decref", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::registerid(
    std::string const& name
  , naming::gid_type const& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_bind, name, id);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->symbol_ns_server.service(req, ec);
        else
            rep = hosted->symbol_ns_.service(req, ec);
   
        // Check if we evicted another entry or if an exception occured. 
        return !ec && (rep.get_gid() == id);
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::registerid", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::unregisterid(
    std::string const& name
  , naming::gid_type& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_unbind, name);
        response rep;
    
        if (is_bootstrap())
            rep = bootstrap->symbol_ns_server.service(req, ec);
        else  
            rep = hosted->symbol_ns_.service(req, ec);
    
        if (!ec && (success == rep.get_status()))
        {
            id = rep.get_gid();
            return true;
        }

        return false;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error()
              , "addressing_service::unregisterid"
              , e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::queryid(
    std::string const& name
  , naming::gid_type& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_resolve, name);
        response rep;
  
        if (is_bootstrap())
            rep = bootstrap->symbol_ns_server.service(req, ec);
        else
            rep = hosted->symbol_ns_.service(req, ec);
    
        if (!ec && (success == rep.get_status()))
        {
            id = rep.get_gid();
            return true;
        }
    
        else
            return false;
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::queryid", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

/// Invoke the supplied hpx::function for every registered global name
bool addressing_service::iterateids(
    iterateids_function_type const& f
  , error_code& ec
    ) 
{ // {{{
    try {
        request req(symbol_ns_iterate, f);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->symbol_ns_server.service(req, ec);
        else
            rep = hosted->symbol_ns_.service(req, ec);

        return !ec && (success == rep.get_status());
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error()
              , "addressing_service::iterateids" 
              , e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

}}

