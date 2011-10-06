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
    
    naming::locality ep = ini_.get_agas_locality();

    gva_type primary_gva(ep,
        primary_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->primary_ns_server));
    gva_type component_gva(ep,
        component_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->component_ns_server));
    gva_type symbol_gva(ep,
        symbol_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->symbol_ns_server));
    
    bootstrap->primary_ns_server.bind_locality(ep, 3);
    bootstrap->symbol_ns_server.bind("/locality(agas#0)",
        naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 

    local_prefix(naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX));

    bootstrap->primary_ns_server.bind_gid
        (primary_namespace_server_type::fixed_gid(), primary_gva);
    bootstrap->primary_ns_server.bind_gid
        (component_namespace_server_type::fixed_gid(), component_gva);
    bootstrap->primary_ns_server.bind_gid
        (symbol_namespace_server_type::fixed_gid(), symbol_gva);

    naming::gid_type lower, upper;
    get_id_range(ep, HPX_INITIAL_GID_RANGE, lower, upper);
    get_runtime().get_id_pool().set_range(lower, upper);

    if (runtime_type == runtime_mode_console)
        bootstrap->symbol_ns_server.bind("/locality(console)",
            naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 

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

bool addressing_service::register_locality(
    naming::locality const& ep
  , naming::gid_type& prefix
  , error_code& ec
    )
{ // {{{
    try {
        response r;
    
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.bind_locality(ep, 0, ec);
        else
            r = hosted->primary_ns_.bind(ep, 0, ec);

        if (ec || (success != r.get_status()))
            return false;
   
        prefix = naming::get_gid_from_prefix(r.get_prefix());
 
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
        response r;
    
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.unbind_locality(ep, ec);
        else
            r = hosted->primary_ns_.unbind(ep, ec);

        if (ec || (success != r.get_status()))
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
            response r = 
                bootstrap->symbol_ns_server.resolve("/locality(console)", ec);
    
            if (!ec &&
                (r.get_gid() != naming::invalid_gid) &&
                (r.get_status() == success))
            {
                prefix = r.get_gid();
                return true;
            }
        }
    
        else {
            response r = hosted->symbol_ns_.resolve
                ("/locality(console)", ec);
    
            if (!ec &&
                (r.get_gid() != naming::invalid_gid) &&
                (r.get_status() == success))
            {
                prefix = r.get_gid();
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
            response r;
    
            if (is_bootstrap())
                r = bootstrap->component_ns_server.resolve_id(type, ec);
            else
                r = hosted->component_ns_.resolve(type, ec);

            if (ec)
                return false;
    
            const std::vector<boost::uint32_t> p = r.get_localities();
    
            // REVIEW: Check response status too? 
            if (!p.size())
                return false;
    
            for (count_type i = 0; i < p.size(); ++i) 
                prefixes.push_back(naming::get_gid_from_prefix(p[i]));
    
            return true; 
        }
    
        else
        {
            response r;
    
            if (is_bootstrap())
                r = bootstrap->primary_ns_server.localities(ec);
            else
                r = hosted->primary_ns_.localities(ec);
           
            if (ec)
                return false;

            const std::vector<boost::uint32_t> p = r.get_localities();
    
            // REVIEW: Check response status too? 
            if (!p.size())
                return false;
    
            for (count_type i = 0; i < p.size(); ++i) 
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
        response r;
    
        if (is_bootstrap())
            r = bootstrap->component_ns_server.bind_name(name, ec);
        else
            r = hosted->component_ns_.bind(name, ec);

        if (ec)
            return components::component_invalid;
    
        // REVIEW: Check response status?
        return (components::component_type) r.get_component_type();
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
    naming::gid_type const& prefix
  , std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        response r;

        if (is_bootstrap())
            r = bootstrap->component_ns_server.bind_prefix(name, prefix, ec);
        else
            r = hosted->component_ns_.bind(name, prefix, ec);
        
        if (ec)
            return components::component_invalid;

        // REVIEW: Check response status?
        return (components::component_type) r.get_component_type();
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
  , count_type count
  , naming::gid_type& lower_bound
  , naming::gid_type& upper_bound
  , error_code& ec
    )
{ // {{{ get_id_range implementation
    typedef lcos::eager_future<
        primary_namespace_server_type::bind_locality_action,
        response
    > allocate_response_future_type;

    allocate_response_future_type* f = 0;

    try {
        response r;
    
        if (is_bootstrap())
        {
            r = bootstrap->primary_ns_server.bind_locality(ep, count, ec);

            if (ec) 
                return false;
        }
 
        else
        {
            // WARNING: this deadlocks if AGAS is unresponsive and all response
            // futures are checked out and pending.
    
            // get a future
            lock_semaphore lock(hosted->allocate_response_sema_);
            
            typedef checkout_future<allocate_response_pool_type, 
                allocate_response_future_type> checkout_future_type;
            checkout_future_type cf(hosted->allocate_response_pool_, f);

            // execute the action (synchronously)
            cf->apply(
                naming::id_type(primary_namespace_server_type::fixed_gid()
                              , naming::id_type::unmanaged),
                ep, count);
            r = cf->get(ec);

            cf.set_ok();

            if (ec)
                return false;
        }
    
        lower_bound = r.get_lower_bound(); 
        upper_bound = r.get_upper_bound();
        BOOST_ASSERT(lower_bound != upper_bound);

        return lower_bound && upper_bound;
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
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::get_id_range", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::bind_range(
    naming::gid_type const& lower_id
  , count_type count
  , naming::address const& baseaddr
  , offset_type offset
  , error_code& ec
    ) 
{ // {{{ bind_range implementation
    typedef lcos::eager_future<
        primary_namespace_server_type::bind_gid_action,
        response
    > bind_response_future_type;

    bind_response_future_type* f = 0;
    
    try {
        naming::locality const& ep = baseaddr.locality_;
       
        // Create a global virtual address from the legacy calling convention
        // parameters.
        gva_type gva(ep, baseaddr.type_, count, baseaddr.address_, offset);
        
        if (is_bootstrap())
        {
            response r
                = bootstrap->primary_ns_server.bind_gid(lower_id, gva, ec);
    
            if (!ec && success == r.get_status()) 
                return true;
        }
    
        else
        {
            // WARNING: this deadlocks if AGAS is unresponsive and all response
            // futures are checked out and pending.
            // get a future

            // wait for the semaphore to get available 
            lock_semaphore lock(hosted->bind_response_sema_);

            typedef checkout_future<bind_response_pool_type, 
                bind_response_future_type> checkout_future_type;
            checkout_future_type cf(hosted->bind_response_pool_, f);
    
            // execute the action (synchronously)
            cf->apply(
                naming::id_type(primary_namespace_server_type::fixed_gid()
                              , naming::id_type::unmanaged),
                lower_id, gva);
            response r = cf->get(ec);

            cf.set_ok();

            if (ec)
                return false;
         
            if (success == r.get_status()) 
            { 
                cache_mutex_type::scoped_lock
                    lock(hosted->gva_cache_mtx_);
                gva_cache_key key(lower_id, count);
                hosted->gva_cache_.insert(key, gva);
                return true;
            }
        }
    
        return false; 
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
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::bind_range", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::unbind_range(
    naming::gid_type const& lower_id
  , count_type count
  , naming::address& addr
  , error_code& ec 
    )
{ // {{{ unbind_range implementation
    try {
        response r;
    
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.unbind_gid(lower_id, count, ec);
        else
            r = hosted->primary_ns_.unbind(lower_id, count, ec);

        if (ec)
            return false;
 
        if (!is_bootstrap() && (success == r.get_status()))
        {
            // I'm afraid that this will break the first form of paged caching,
            // so it's commented out for now.
            //cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            //gva_erase_policy ep(lower_id, count);
            //hosted->gva_cache_.erase(ep);
            addr.locality_ = r.get_gva().endpoint;
            addr.type_ = r.get_gva().type;
            addr.address_ = r.get_gva().lva();
        }
    
        return success == r.get_status(); 
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::unbind_range", 
                e.what());
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
 
        response r; 
    
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.page_fault(id, ec);
        else
        {
            LHPX_(info, "  [AC] ") <<
                (boost::format("soft page fault, faulting address %1%") % id);
            r = hosted->primary_ns_.page_fault(id, ec);
        } 

        if (ec || (success != r.get_status()))
            return false;

        // Resolve the page to the real resolved address (which is just a page
        // with as fully resolved LVA and an offset of zero).
        gva_type g = r.get_gva().resolve(id, r.get_base_gid());

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva();
    
        if (!is_bootstrap())
        {
            // Put the page into the cache.
            cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            gva_cache_key key(r.get_base_gid(), r.get_gva().count);
            hosted->gva_cache_.insert(key, r.get_gva());
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
        const boost::uint64_t idbase_msb
            = naming::strip_credit_from_gid(idbase.id.get_msb());

        if (HPX_UNLIKELY(id_msb != idbase_msb))
        {
            HPX_THROWS_IF(ec, invalid_page_fault
              , "addressing_service::resolve_cached" 
              , "bad page in cache, MSBs of GID base and GID do not match");
            return false;
        }

        gva_type const& gva = e.get();

        addr.locality_ = gva.endpoint;
        addr.type_ = gva.type;
        addr.address_ = gva.lva(id, idbase.id);

        if (&ec != &throws)
            ec = make_success_code();

        LHPX_(debug, "  [AC] ") <<
            ( boost::format("cache hit for address %1% (base %2%)")
            % id % idbase.id);
    
        return true;
    }

    if (&ec != &throws)
        ec = make_success_code();

    LHPX_(debug, "  [AC] ") <<
        (boost::format("cache miss for address %1%") % id);

    return false;
} // }}}

addressing_service::count_type addressing_service::incref(
    naming::gid_type const& id
  , count_type credits
  , error_code& ec 
    )
{ // {{{ incref implementation
    try {
        response r;

        if (is_bootstrap())
            r = bootstrap->primary_ns_server.increment(id, credits, ec);
        else
            r = hosted->primary_ns_.increment(id, credits, ec);

        if (ec)
            return 0;

        return r.get_count();
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

addressing_service::count_type addressing_service::decref(
    naming::gid_type const& id
  , components::component_type& t
  , count_type credits
  , error_code& ec
    )
{ // {{{ decref implementation
    try {
        response r;
        
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.decrement(id, credits, ec);
        else
            r = hosted->primary_ns_.decrement(id, credits, ec);
   
        if (ec)
            return 0;
 
        if (0 == r.get_count())
        {
            t = (components::component_type) r.get_component_type();

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
    
        return r.get_count();
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
        response r;
    
        if (is_bootstrap())
            r = bootstrap->symbol_ns_server.bind(name, id, ec);
        else
            r = hosted->symbol_ns_.bind(name, id, ec);
   
        // Check if we evicted another entry or if an exception occured. 
        return !ec && (r.get_gid() == id);
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
  , error_code& ec
    )
{ // {{{
    try {
        response r;
    
        if (is_bootstrap())
            r = bootstrap->symbol_ns_server.unbind(name, ec);
        else  
            r = hosted->symbol_ns_.unbind(name, ec);
    
        return !ec && (success == r.get_status());
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::unregisterid", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

bool addressing_service::queryid(
    std::string const& ns_name
  , naming::gid_type& id
  , error_code& ec
    )
{ // {{{
    try {
        response r;
    
        if (is_bootstrap())
            r = bootstrap->symbol_ns_server.resolve(ns_name, ec);
        else
            r = hosted->symbol_ns_.resolve(ns_name, ec);
    
        if (!ec && (success == r.get_status()))
        {
            id = r.get_gid();
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
        response r;

        if (is_bootstrap())
            r = bootstrap->symbol_ns_server.iterate(f, ec);
        else
            r = hosted->symbol_ns_.iterate(f, ec);

        return !ec && (success == r.get_status());
    }
    catch (hpx::exception const& e) {
        if (&ec == &throws) {
            HPX_RETHROW_EXCEPTION(e.get_error(), "addressing_service::iterateids", 
                e.what());
        }
        else {
            ec = e.get_error_code(hpx::rethrow); 
        }

        return false;
    }
} // }}}

}}

