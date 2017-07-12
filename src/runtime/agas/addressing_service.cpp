////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime.hpp>
#include <hpx/apply.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/runtime/agas/detail/bootstrap_component_namespace.hpp>
#include <hpx/runtime/agas/detail/bootstrap_locality_namespace.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/naming/split_gid.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <boost/format.hpp>
#include <boost/icl/closed_interval.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas
{
struct addressing_service::gva_cache_key
{ // {{{ gva_cache_key implementation
  private:
    typedef boost::icl::closed_interval<naming::gid_type, std::less>
        key_type;

    key_type key_;

  public:
    gva_cache_key()
      : key_()
    {}

    explicit gva_cache_key(
        naming::gid_type const& id_
      , std::uint64_t count_ = 1
        )
      : key_(naming::detail::get_stripped_gid(id_)
           , naming::detail::get_stripped_gid(id_) + (count_ - 1))
    {
        HPX_ASSERT(count_);
    }

    naming::gid_type get_gid() const
    {
        return boost::icl::lower(key_);
    }

    std::uint64_t get_count() const
    {
        naming::gid_type const size = boost::icl::length(key_);
        HPX_ASSERT(size.get_msb() == 0);
        return size.get_lsb();
    }

    friend bool operator<(
        gva_cache_key const& lhs
      , gva_cache_key const& rhs
        )
    {
        return boost::icl::exclusive_less(lhs.key_, rhs.key_);
    }

    friend bool operator==(
        gva_cache_key const& lhs
      , gva_cache_key const& rhs
        )
    {
        // Direct hit
        if(lhs.key_ == rhs.key_)
            return true;

        // Is lhs in rhs?
        if (1 == lhs.get_count() && 1 != rhs.get_count())
            return boost::icl::contains(rhs.key_, lhs.key_);

        // Is rhs in lhs?
        else if (1 != lhs.get_count() && 1 == rhs.get_count())
            return boost::icl::contains(lhs.key_, rhs.key_);

        return false;
    }
}; // }}}

addressing_service::addressing_service(
    parcelset::parcelhandler& ph
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
    )
  : gva_cache_(new gva_cache_type)
  , console_cache_(naming::invalid_locality_id)
  , max_refcnt_requests_(ini_.get_agas_max_pending_refcnt_requests())
  , refcnt_requests_count_(0)
  , enable_refcnt_caching_(true)
  , refcnt_requests_(new refcnt_requests_type)
  , service_type(ini_.get_agas_service_mode())
  , runtime_type(runtime_type_)
  , caching_(ini_.get_agas_caching_mode())
  , range_caching_(caching_ ? ini_.get_agas_range_caching_mode() : false)
  , action_priority_(threads::thread_priority_boost)
  , rts_lva_(0)
  , mem_lva_(0)
  , state_(state_starting)
  , locality_()
{ // {{{
    LPROGRESS_;

    if (caching_)
        gva_cache_->reserve(ini_.get_agas_local_cache_size());

#if defined(HPX_HAVE_NETWORKING)
    std::shared_ptr<parcelset::parcelport> pp = ph.get_bootstrap_parcelport();
    create_big_boot_barrier(pp ? pp.get() : nullptr, ph.endpoints(), ini_);
    if (service_type == service_mode_bootstrap)
    {
        launch_bootstrap(pp, ph.endpoints(), ini_);
    }
#else
    create_big_boot_barrier(nullptr, ph.endpoints(), ini_);
    HPX_ASSERT(service_type == service_mode_bootstrap);
    launch_bootstrap(nullptr, ph.endpoints(), ini_);
#endif
} // }}}

void addressing_service::initialize(parcelset::parcelhandler& ph,
    std::uint64_t rts_lva, std::uint64_t mem_lva)
{ // {{{
    rts_lva_ = rts_lva;
    mem_lva_ = mem_lva;

#if defined(HPX_HAVE_NETWORKING)
    // now, boot the parcel port
    std::shared_ptr<parcelset::parcelport> pp = ph.get_bootstrap_parcelport();
    if(pp)
        pp->run(false);

    if (service_type == service_mode_bootstrap)
    {
        get_big_boot_barrier().wait_bootstrap();
    }
    else
    {
        launch_hosted();
        get_big_boot_barrier().wait_hosted(
            pp ? pp->get_locality_name() : "<console>",
            primary_ns_.ptr(), symbol_ns_.ptr());
    }
#else
    HPX_ASSERT(service_type == service_mode_bootstrap);
    get_big_boot_barrier().wait_bootstrap();
#endif

    set_status(state_running);
} // }}}

namespace detail
{
    std::uint32_t get_number_of_pus_in_cores(std::uint32_t num_cores);
}

void addressing_service::launch_bootstrap(
    std::shared_ptr<parcelset::parcelport> const& pp
  , parcelset::endpoints_type const & endpoints
  , util::runtime_configuration const& ini_
    )
{ // {{{
    component_ns_.reset(new detail::bootstrap_component_namespace);
    locality_ns_.reset(new detail::bootstrap_locality_namespace(
        reinterpret_cast<server::primary_namespace *>(primary_ns_.ptr())));

    runtime& rt = get_runtime();

    naming::gid_type const here =
        naming::get_gid_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX);
    set_local_locality(here);

    // store number of cores used by other processes
    std::uint32_t cores_needed = rt.assign_cores();
    std::uint32_t first_used_core = rt.assign_cores(
        pp ? pp->get_locality_name() : "<console>", cores_needed);

    util::runtime_configuration& cfg = rt.get_config();
    cfg.set_first_used_core(first_used_core);
    HPX_ASSERT(pp ? pp->here() == pp->agas_locality(cfg) : true);

    naming::id_type const locality_gid = locality_ns_->gid();
    gva locality_gva(here,
        server::locality_namespace::get_component_type(), 1U,
            locality_ns_->ptr());

    naming::id_type const primary_gid = primary_ns_.gid();
    gva primary_gva(here,
        server::primary_namespace::get_component_type(), 1U,
            primary_ns_.ptr());

    naming::id_type const component_gid = component_ns_->gid();
    gva component_gva(here,
         server::component_namespace::get_component_type(), 1U,
            component_ns_->ptr());

    naming::id_type const symbol_gid = symbol_ns_.gid();
    gva symbol_gva(here,
        server::symbol_namespace::get_component_type(), 1U,
            symbol_ns_.ptr());

    rt.get_config().parse("assigned locality",
        boost::str(boost::format("hpx.locality!=%1%")
                  % naming::get_locality_id_from_gid(here)));

    std::uint32_t num_threads = hpx::util::get_entry_as<std::uint32_t>(
        ini_, "hpx.os_threads", 1u);
    locality_ns_->allocate(endpoints, 0, num_threads, naming::invalid_gid);

    naming::gid_type runtime_support_gid1(here);
    runtime_support_gid1.set_lsb(rt.get_runtime_support_lva());
    naming::gid_type runtime_support_gid2(here);
    runtime_support_gid2.set_lsb(std::uint64_t(0));

    gva runtime_support_address(here
      , components::get_component_type<components::server::runtime_support>()
      , 1U, rt.get_runtime_support_lva());

    register_name("/0/agas/locality#0", here);
    if (is_console())
        register_name("/0/locality#console", here);

    naming::gid_type lower, upper;
    get_id_range(HPX_INITIAL_GID_RANGE, lower, upper);
    rt.get_id_pool().set_range(lower, upper);
} // }}}

void addressing_service::launch_hosted()
{
}

void addressing_service::adjust_local_cache_size(std::size_t cache_size)
{ // {{{
    // adjust the local AGAS cache size for the number of worker threads and
    // create the hierarchy based on the topology
    if (caching_)
    {
        std::size_t previous = gva_cache_->size();
            gva_cache_->reserve(cache_size);

        LAGAS_(info) << (boost::format(
            "addressing_service::adjust_local_cache_size, previous size: %1%, "
            "new size: %3%")
            % previous % cache_size);
    }
} // }}}

void addressing_service::set_local_locality(naming::gid_type const& g)
{
    locality_ = g;
    primary_ns_.set_local_locality(g);
}

bool addressing_service::register_locality(
    parcelset::endpoints_type const & endpoints
  , naming::gid_type& prefix
  , std::uint32_t num_threads
  , error_code& ec
    )
{ // {{{
    try {
        prefix = naming::get_gid_from_locality_id(
            locality_ns_->allocate(endpoints, 0, num_threads, prefix));

        {
            std::unique_lock<mutex_type> l(resolved_localities_mtx_);
            std::pair<resolved_localities_type::iterator, bool> res
                = resolved_localities_.insert(std::make_pair(
                    prefix
                  , endpoints
                ));

            if (!res.second)
            {
                l.unlock();
                HPX_THROWS_IF(ec, bad_parameter,
                    "addressing_service::register_locality",
                    "locality insertion failed because of a duplicate");
                return false;
            }
        }

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_locality");
        return false;
    }
} // }}}

void addressing_service::register_console(parcelset::endpoints_type const & eps)
{
    std::lock_guard<mutex_type> l(resolved_localities_mtx_);
    std::pair<resolved_localities_type::iterator, bool> res
        = resolved_localities_.insert(std::make_pair(
            naming::get_gid_from_locality_id(0)
          , eps
        ));
    HPX_ASSERT(res.second);
}

bool addressing_service::has_resolved_locality(
    naming::gid_type const & gid
    )
{ // {{{
    std::unique_lock<mutex_type> l(resolved_localities_mtx_);
    return resolved_localities_.find(gid) != resolved_localities_.end();
} // }}}

void addressing_service::pre_cache_endpoints(
    std::vector<parcelset::endpoints_type> const& endpoints)
{ // {{{
    std::unique_lock<mutex_type> l(resolved_localities_mtx_);
    std::uint32_t locality_id = 0;
    for (parcelset::endpoints_type const& endpoint : endpoints)
    {
        resolved_localities_.insert(
            resolved_localities_type::value_type(
                naming::get_gid_from_locality_id(locality_id),
                endpoint));
        ++locality_id;
    }
} // }}}

parcelset::endpoints_type const & addressing_service::resolve_locality(
    naming::gid_type const & gid
  , error_code& ec
    )
{ // {{{
    std::unique_lock<mutex_type> l(resolved_localities_mtx_);
    resolved_localities_type::iterator it = resolved_localities_.find(gid);
    if (it == resolved_localities_.end() || it->second.empty())
    {
        // The locality hasn't been requested to be resolved yet. Do it now.
        parcelset::endpoints_type endpoints;
        {
            hpx::util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
            endpoints = locality_ns_->resolve_locality(gid);
            if (endpoints.empty())
            {
                std::stringstream strm;
                strm << "couldn't resolve the given target locality ("
                     << gid << ")";
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter,
                    "addressing_service::resolve_locality",
                    strm.str());
                return resolved_localities_[naming::invalid_gid];
            }
        }

        // Search again ... might have been added by a different thread already
        it = resolved_localities_.find(gid);
        if (it == resolved_localities_.end())
        {
            if(HPX_UNLIKELY(!util::insert_checked(resolved_localities_.insert(
                    std::make_pair(
                        gid
                      , endpoints
                    )
                ), it)))
            {
                l.unlock();

                HPX_THROWS_IF(ec, internal_server_error
                  , "addressing_service::resolve_locality"
                  , "resolved locality insertion failed "
                    "due to a locking error or memory corruption");
                return resolved_localities_[naming::invalid_gid];
            }
        }
        else if (it->second.empty() && !endpoints.empty())
        {
            resolved_localities_[gid] = endpoints;
        }
    }
    return it->second;
} // }}}

// TODO: We need to ensure that the locality isn't unbound while it still holds
// referenced objects.
bool addressing_service::unregister_locality(
    naming::gid_type const & gid
  , error_code& ec
    )
{ // {{{
    try {
        locality_ns_->free(gid);
        component_ns_->unregister_server_instance(ec);
        symbol_ns_.unregister_server_instance(ec);

        remove_resolved_locality(gid);
        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unregister_locality");
        return false;
    }
} // }}}

void addressing_service::remove_resolved_locality(naming::gid_type const& gid)
{
    std::lock_guard<mutex_type> l(resolved_localities_mtx_);
    resolved_localities_type::iterator it = resolved_localities_.find(gid);
    if(it != resolved_localities_.end())
        resolved_localities_.erase(it);
}


bool addressing_service::get_console_locality(
    naming::gid_type& prefix
  , error_code& ec
    )
{ // {{{
    try {
        if (get_status() != state_running)
        {
            if (&ec != &throws)
                ec = make_success_code();
            return false;
        }

        if (is_console())
        {
            prefix = get_local_locality();
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }

        {
            std::lock_guard<mutex_type> lock(console_cache_mtx_);

            if (console_cache_ != naming::invalid_locality_id)
            {
                prefix = naming::get_gid_from_locality_id(console_cache_);
                if (&ec != &throws)
                    ec = make_success_code();
                return true;
            }
        }

        std::string key("/0/locality#console");

        hpx::id_type resolved_prefix = resolve_name(key);
        if (resolved_prefix != naming::invalid_id)
        {
            std::uint32_t console = naming::get_locality_id_from_id(resolved_prefix);
            prefix = resolved_prefix.get_gid();

            {
                std::lock_guard<mutex_type> lock(console_cache_mtx_);
                if (console_cache_ == naming::invalid_locality_id) {
                    console_cache_ = console;
                }
                else {
                    HPX_ASSERT(console_cache_ == console);
                }
            }

            LAGAS_(debug) <<
                ( boost::format(
                  "addressing_server::get_console_locality, "
                  "caching console locality, prefix(%1%)")
                % console);

            return true;
        }

        return false;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_console_locality");
        return false;
    }
} // }}}

bool addressing_service::get_localities(
    std::vector<naming::gid_type>& locality_ids
  , components::component_type type
  , error_code& ec
    )
{ // {{{ get_locality_ids implementation
    try {
        if (type != components::component_invalid)
        {
            const std::vector<std::uint32_t> p = component_ns_->resolve_id(type);

            if (!p.size())
                return false;

            locality_ids.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                locality_ids.push_back(naming::get_gid_from_locality_id(p[i]));

            return true;
        }

        else
        {
            const std::vector<std::uint32_t> p = locality_ns_->localities();

            if (!p.size())
                return false;

            locality_ids.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                locality_ids.push_back(naming::get_gid_from_locality_id(p[i]));

            return true;
        }
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_locality_ids");
        return false;
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
std::uint32_t addressing_service::get_num_localities(
    components::component_type type
  , error_code& ec
    )
{ // {{{ get_num_localities implementation
    try {
        if (type == components::component_invalid)
        {
            return locality_ns_->get_num_localities();
        }

        return component_ns_->get_num_localities(type).get();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_localities");
    }
    return std::uint32_t(-1);
} // }}}

lcos::future<std::uint32_t> addressing_service::get_num_localities_async(
    components::component_type type
    )
{ // {{{ get_num_localities implementation
    if (type == components::component_invalid)
    {
        return locality_ns_->get_num_localities_async();
    }

    return component_ns_->get_num_localities(type);
} // }}}

///////////////////////////////////////////////////////////////////////////////
std::uint32_t addressing_service::get_num_overall_threads(
    error_code& ec
    )
{ // {{{ get_num_overall_threads implementation
    try {
        return locality_ns_->get_num_overall_threads();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_overall_threads");
    }
    return std::uint32_t(0);
} // }}}

lcos::future<std::uint32_t> addressing_service::get_num_overall_threads_async()
{ // {{{
    return locality_ns_->get_num_overall_threads_async();
} // }}}

std::vector<std::uint32_t> addressing_service::get_num_threads(
    error_code& ec
    )
{ // {{{ get_num_threads implementation
    try {
        return locality_ns_->get_num_threads();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_threads");
    }
    return std::vector<std::uint32_t>();
} // }}}

lcos::future<std::vector<std::uint32_t> > addressing_service::get_num_threads_async()
{ // {{{
    return locality_ns_->get_num_threads_async();
} // }}}

///////////////////////////////////////////////////////////////////////////////
components::component_type addressing_service::get_component_id(
    std::string const& name
  , error_code& ec
    )
{ /// {{{
    try {
        return component_ns_->bind_name(name);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_component_id");
        return components::component_invalid;
    }
} // }}}

void addressing_service::iterate_types(
    iterate_types_function_type const& f
  , error_code& ec
    )
{ // {{{
    try {
        return component_ns_->iterate_types(f);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_types");
    }
} // }}}

std::string addressing_service::get_component_type_name(
    components::component_type id
  , error_code& ec
    )
{ // {{{
    try {
        return component_ns_->get_component_type_name(id);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_types");
    }
    return "<unknown>";
} // }}}

components::component_type addressing_service::register_factory(
    std::uint32_t prefix
  , std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        return component_ns_->bind_prefix(name, prefix);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_factory");
        return components::component_invalid;
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::get_id_range(
    std::uint64_t count
  , naming::gid_type& lower_bound
  , naming::gid_type& upper_bound
  , error_code& ec
    )
{ // {{{ get_id_range implementation
    try {
        // parcelset::endpoints_type() is an obsolete, dummy argument

        std::pair<naming::gid_type, naming::gid_type> rep(
            primary_ns_.allocate(count));

        if(rep.first == naming::invalid_gid || rep.second == naming::invalid_gid)
            return false;

        lower_bound = rep.first;
        upper_bound = rep.second;

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_id_range");
        return false;
    }
} // }}}

bool addressing_service::bind_range_local(
    naming::gid_type const& lower_id
  , std::uint64_t count
  , naming::address const& baseaddr
  , std::uint64_t offset
  , error_code& ec
    )
{ // {{{ bind_range implementation
    try {
        naming::gid_type const& prefix = baseaddr.locality_;

        // Create a global virtual address from the legacy calling convention
        // parameters
        gva const g(prefix, baseaddr.type_, count, baseaddr.address_, offset);

        primary_ns_.bind_gid(g, lower_id, naming::get_locality_from_gid(lower_id));

        if (range_caching_)
        {
            // Put the range into the cache.
            update_cache_entry(lower_id, g, ec);
        }
        else
        {
            // Only put the first GID in the range into the cache
            gva const first_g = g.resolve(lower_id, lower_id);
            update_cache_entry(lower_id, first_g, ec);
        }

        if (ec)
            return false;

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::bind_range_local");
        return false;
    }
} // }}}

bool addressing_service::bind_postproc(
    future<bool> f, naming::gid_type const& lower_id, gva const& g
    )
{
    f.get();

    if(range_caching_)
    {
        // Put the range into the cache.
        update_cache_entry(lower_id, g);
    }
    else
    {
        // Only put the first GID in the range into the cache
        gva const first_g = g.resolve(lower_id, lower_id);
        update_cache_entry(lower_id, first_g);
    }

    return true;
}

hpx::future<bool> addressing_service::bind_range_async(
    naming::gid_type const& lower_id
  , std::uint64_t count
  , naming::address const& baseaddr
  , std::uint64_t offset
  , naming::gid_type const& locality
    )
{
    // ask server
    naming::gid_type const& prefix = baseaddr.locality_;

    // Create a global virtual address from the legacy calling convention
    // parameters.
    gva const g(prefix, baseaddr.type_, count, baseaddr.address_, offset);

    naming::gid_type id(
        naming::detail::get_stripped_gid_except_dont_cache(lower_id));

    future<bool> f = primary_ns_.bind_gid_async(g, id, locality);

    return f.then(util::bind(
            util::one_shot(&addressing_service::bind_postproc),
            this, _1, id, g
        ));
}

hpx::future<naming::address> addressing_service::unbind_range_async(
    naming::gid_type const& lower_id
  , std::uint64_t count
    )
{
    return primary_ns_.unbind_gid_async(count, lower_id);
}

bool addressing_service::unbind_range_local(
    naming::gid_type const& lower_id
  , std::uint64_t count
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ unbind_range implementation
    try {

        naming::gid_type gid = naming::detail::get_stripped_gid(lower_id);

        addr = primary_ns_.unbind_gid(count, lower_id);

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unbind_range_local");
        return false;
    }
} // }}}

/// This function will test whether the given address refers to an object
/// living on the locality of the caller. We rely completely on the local AGAS
/// cache and local AGAS instance, assuming that everything which is not in
/// the cache is not local.

// bool addressing_service::is_local_address(
//     naming::gid_type const& id
//   , naming::address& addr
//   , error_code& ec
//     )
// {
//     // Resolve the address of the GID.
//
//     // NOTE: We do not throw here for a reason; it is perfectly valid for the
//     // GID to not be found in the local AGAS instance.
//     if (!resolve(id, addr, ec) || ec)
//         return false;
//
//     return addr.locality_ == get_here();
// }

bool addressing_service::is_local_address_cached(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{
    // Assume non-local operation if the gid is known to have been migrated
    naming::gid_type id(naming::detail::get_stripped_gid_except_dont_cache(gid));

    {
        std::lock_guard<mutex_type> lock(migrated_objects_mtx_);
        if (was_object_migrated_locked(id))
        {
            if (&ec != &throws)
                ec = make_success_code();
            return false;
        }
    }

    // Try to resolve the address of the GID from the locally available
    // information.

    // NOTE: We do not throw here for a reason; it is perfectly valid for the
    // GID to not be found in the cache.
    if (!resolve_cached(id, addr, ec) || ec)
    {
        if (ec) return false;

        // try also the local part of AGAS before giving up
        if (!resolve_full_local(id, addr, ec) || ec)
            return false;
    }

    return addr.locality_ == get_local_locality();
}

// Return true if at least one address is local.
// bool addressing_service::is_local_address(
//     naming::gid_type const* gids
//   , naming::address* addrs
//   , std::size_t size
//   , boost::dynamic_bitset<>& locals
//   , error_code& ec
//     )
// {
//     // Try the cache
//     if (caching_)
//     {
//         bool all_resolved = resolve_cached(gids, addrs, size, locals, ec);
//         if (ec)
//             return false;
//         if (all_resolved)
//             return locals.any();      // all destinations resolved
//     }
//
//     if (!resolve_full(gids, addrs, size, locals, ec) || ec)
//         return false;
//
//     return locals.any();
// }

bool addressing_service::is_local_lva_encoded_address(
    std::uint64_t msb
    )
{
    // NOTE: This should still be migration safe.
    return naming::detail::strip_internal_bits_from_gid(msb) ==
        get_local_locality().get_msb();
}

bool addressing_service::resolve_locally_known_addresses(
    naming::gid_type const& id
  , naming::address& addr
    )
{
    // LVA-encoded GIDs (located on this machine)
    std::uint64_t lsb = id.get_lsb();
    std::uint64_t msb =
        naming::detail::strip_internal_bits_from_gid(id.get_msb());

    if (is_local_lva_encoded_address(msb))
    {
        addr.locality_ = get_local_locality();

        // An LSB of 0 references the runtime support component
        HPX_ASSERT(rts_lva_);

        if (0 == lsb || lsb == rts_lva_)
        {
            addr.type_ = components::component_runtime_support;
            addr.address_ = rts_lva_;
        }
        else
        {
            HPX_ASSERT(mem_lva_);

            addr.type_ = components::component_memory;
            addr.address_ = mem_lva_;
        }

        return true;
    }

    // explicitly resolve localities
    if (naming::is_locality(id))
    {
        addr.locality_ = id;
        addr.type_ = components::component_runtime_support;
        // addr.address_ will be supplied on the target locality
        return true;
    }

    // authoritative AGAS component address resolution
    if (HPX_AGAS_LOCALITY_NS_MSB == msb && HPX_AGAS_LOCALITY_NS_LSB == lsb)
    {
        addr = locality_ns_->addr();
        return true;
    }
    if (HPX_AGAS_COMPONENT_NS_MSB == msb && HPX_AGAS_COMPONENT_NS_LSB == lsb)
    {
        addr = component_ns_->addr();
        return true;
    }

    naming::gid_type dest = naming::get_locality_from_gid(id);
    if (HPX_AGAS_PRIMARY_NS_LSB == lsb)
    {
        // primary AGAS service on locality 0?
        if(dest == get_local_locality())
        {
            addr = primary_ns_.addr();
        }
        // primary AGAS service on any locality
        else
        {
            addr.locality_ = dest;
            addr.type_ = server::primary_namespace::get_component_type();
            // addr.address_ will be supplied on the target locality
            return true;
        }
    }

    if (HPX_AGAS_SYMBOL_NS_LSB == lsb)
    {
        // symbol AGAS service on this locality?
        if(dest == get_local_locality())
        {
            addr = symbol_ns_.addr();
        }
        // symbol AGAS service on any locality
        else
        {
            addr.locality_ = dest;
            addr.type_ = server::symbol_namespace::get_component_type();
            // addr.address_ will be supplied on the target locality
        }

        return true;
    }

    return false;
} // }}}

bool addressing_service::resolve_full_local(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ resolve implementation
    try {
        auto rep = primary_ns_.resolve_gid(id);

        using hpx::util::get;

        if (get<0>(rep) == naming::invalid_gid || get<2>(rep) == naming::invalid_gid)
            return false;

        // Resolve the gva to the real resolved address (which is just a gva
        // with as fully resolved LVA and an offset of zero).
        naming::gid_type base_gid = get<0>(rep);
        gva const base_gva = get<1>(rep);

        gva const g = base_gva.resolve(id, base_gid);

        addr.locality_ = g.prefix;
        addr.type_ = g.type;
        addr.address_ = g.lva();

        if (naming::detail::store_in_cache(id))
        {
            HPX_ASSERT(addr.address_);
            if(range_caching_)
            {
                // Put the range into the cache.
                update_cache_entry(base_gid, base_gva, ec);
            }
            else
            {
                // Put the fully resolved gva into the cache.
                update_cache_entry(id, g, ec);
            }
        }

        if (ec)
            return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_full_local");
        return false;
    }
} // }}}

bool addressing_service::resolve_cached(
    naming::gid_type const& gid
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ resolve_cached implementation

    naming::gid_type id = naming::detail::get_stripped_gid_except_dont_cache(gid);

    // special cases
    if (resolve_locally_known_addresses(id, addr))
        return true;

    // If caching is disabled, bail
    if (!caching_)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return false;
    }

    // don't look at cache if id is marked as non-cache-able
    if (!naming::detail::store_in_cache(id))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return false;
    }

    // don't look at the cache if the id is locally managed
    if (naming::get_locality_id_from_gid(id) ==
        naming::get_locality_id_from_gid(locality_))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return false;
    }

    // force routing if target object was migrated
    {
        std::lock_guard<mutex_type> lock(migrated_objects_mtx_);
        if (was_object_migrated_locked(id))
        {
            if (&ec != &throws)
                ec = make_success_code();
            return false;
        }
    }

    // first look up the requested item in the cache
    gva g;
    naming::gid_type idbase;
    if (get_cache_entry(id, g, idbase, ec))
    {
        addr.locality_ = g.prefix;
        addr.type_ = g.type;
        addr.address_ = g.lva(id, idbase);

        if (&ec != &throws)
            ec = make_success_code();

/*
        LAGAS_(debug) <<
            ( boost::format(
                "addressing_service::resolve_cached, "
                "cache hit for address %1%, lva %2% (base %3%, lva %4%)")
            % id
            % reinterpret_cast<void*>(addr.address_)
            % idbase.get_gid()
            % reinterpret_cast<void*>(g.lva()));
*/

        return true;
    }

    if (&ec != &throws)
        ec = make_success_code();

    LAGAS_(debug) <<
        ( boost::format(
            "addressing_service::resolve_cached, "
            "cache miss for address %1%")
        % id);

    return false;
} // }}}

hpx::future<naming::address> addressing_service::resolve_async(
    naming::gid_type const& gid
    )
{
    if (!gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_async",
            "invalid reference id");
        return make_ready_future(naming::address());
    }

    // Try the cache.
    if (caching_)
    {
        naming::address addr;
        error_code ec;
        if (resolve_cached(gid, addr, ec))
            return make_ready_future(addr);

        if (ec)
        {
            return hpx::make_exceptional_future<naming::address>(
                hpx::detail::access_exception(ec));
        }
    }

    // now try the AGAS service
    return resolve_full_async(gid);
}

hpx::future<naming::id_type> addressing_service::get_colocation_id_async(
    naming::id_type const& id
    )
{
    if (!id)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::get_colocation_id_async",
            "invalid reference id");
        return make_ready_future(naming::invalid_id);
    }

    return primary_ns_.colocate(id.get_gid());
}

///////////////////////////////////////////////////////////////////////////////
naming::address addressing_service::resolve_full_postproc(
    future<primary_namespace::resolved_type> f, naming::gid_type const& id
    )
{
    using hpx::util::get;

    naming::address addr;

    auto rep = f.get();
    if (get<0>(rep) == naming::invalid_gid || get<2>(rep) == naming::invalid_gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_full_postproc",
            "could no resolve global id");
        return addr;
    }

    // Resolve the gva to the real resolved address (which is just a gva
    // with as fully resolved LVA and and offset of zero).
    naming::gid_type base_gid = get<0>(rep);
    gva const base_gva = get<1>(rep);

    gva const g = base_gva.resolve(id, base_gid);

    addr.locality_ = g.prefix;
    addr.type_ = g.type;
    addr.address_ = g.lva();

    if (naming::detail::store_in_cache(id))
    {
        if (range_caching_)
        {
            // Put the range into the cache.
            update_cache_entry(base_gid, base_gva);
        }
        else
        {
            // Put the fully resolved gva into the cache.
            update_cache_entry(id, g);
        }
    }

    return addr;
}

hpx::future<naming::address> addressing_service::resolve_full_async(
    naming::gid_type const& gid
    )
{
    if (!gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_full_async",
            "invalid reference id");
        return make_ready_future(naming::address());
    }

    // ask server
    future<primary_namespace::resolved_type> f =
        primary_ns_.resolve_full(gid);

    using util::placeholders::_1;
    return f.then(util::bind(
            util::one_shot(&addressing_service::resolve_full_postproc),
            this, _1, gid
        ));
}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::resolve_full_local(
    naming::gid_type const* gids
  , naming::address* addrs
  , std::size_t count
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    locals.resize(count);

    try {
        using hpx::util::get;

        // special cases
        for (std::size_t i = 0; i != count; ++i)
        {
            if (addrs[i])
            {
                locals.set(i, true);
                continue;
            }

            HPX_ASSERT(!locals.test(i));

            if (!addrs[i] && !locals.test(i))
            {
                auto rep = primary_ns_.resolve_gid(gids[i]);

                if (get<0>(rep) == naming::invalid_gid ||
                    get<2>(rep) == naming::invalid_gid)
                    return false;
                // Resolve the gva to the real resolved address (which is just a gva
                // with as fully resolved LVA and and offset of zero).
                naming::gid_type base_gid = get<0>(rep);
                gva const base_gva = get<1>(rep);

                gva const g = base_gva.resolve(gids[i], base_gid);

                naming::address& addr = addrs[i];
                addr.locality_ = g.prefix;
                addr.type_ = g.type;
                addr.address_ = g.lva();

                hpx::error_code ec;
                if (naming::detail::store_in_cache(gids[i]))
                {
                    if (range_caching_)
                    {
                        // Put the range into the cache.
                        update_cache_entry(base_gid, base_gva, ec);
                    }
                    else
                    {
                        // Put the fully resolved gva into the cache.
                        update_cache_entry(gids[i], g, ec);
                    }
                }

                if (ec)
                    return false;
            }
        }

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_full");
        return false;
    }
}

bool addressing_service::resolve_cached(
    naming::gid_type const* gids
  , naming::address* addrs
  , std::size_t count
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    locals.resize(count);

    std::size_t resolved = 0;
    for (std::size_t i = 0; i != count; ++i)
    {
        if (!addrs[i] && !locals.test(i))
        {
            bool was_resolved = resolve_cached(gids[i], addrs[i], ec);
            if (ec)
                return false;
            if (was_resolved)
                ++resolved;

            if (addrs[i].locality_ == get_local_locality())
                locals.set(i, true);
        }

        else if (addrs[i].locality_ == get_local_locality())
        {
            ++resolved;
            locals.set(i, true);
        }
    }

    return resolved == count;   // returns whether all have been resolved
}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::route(
    parcelset::parcel p
  , util::function_nonser<void(boost::system::error_code const&,
        parcelset::parcel const&)> && f
  , threads::thread_priority local_priority
    )
{
    if (HPX_UNLIKELY(nullptr == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        void (addressing_service::*route_ptr)(
            parcelset::parcel,
            util::function_nonser<void(boost::system::error_code const&,
                parcelset::parcel const&)> &&,
            threads::thread_priority
        ) = &addressing_service::route;

        threads::register_thread_nullary(
            util::deferred_call(
                route_ptr, this, std::move(p), std::move(f), local_priority),
            "addressing_service::route", threads::pending, true,
            threads::thread_priority_normal, std::size_t(-1),
            threads::thread_stacksize_default);
        return;
    }

    primary_ns_.route(std::move(p), std::move(f));
}

///////////////////////////////////////////////////////////////////////////////
// The parameter 'compensated_credit' holds the amount of credits to be added
// to the acknowledged number of credits. The compensated credits are non-zero
// if there was a pending decref request at the point when the incref was sent.
// The pending decref was subtracted from the amount of credits to incref.
std::int64_t addressing_service::synchronize_with_async_incref(
    hpx::future<std::int64_t> fut
  , naming::id_type const& id
  , std::int64_t compensated_credit
    )
{
    return fut.get() + compensated_credit;
}

lcos::future<std::int64_t> addressing_service::incref_async(
    naming::gid_type const& id
  , std::int64_t credit
  , naming::id_type const& keep_alive
    )
{ // {{{ incref implementation
    naming::gid_type raw(naming::detail::get_stripped_gid(id));

    if (HPX_UNLIKELY(nullptr == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        lcos::future<std::int64_t> (
                addressing_service::*incref_async_ptr)(
            naming::gid_type const&
          , std::int64_t
          , naming::id_type const&
        ) = &addressing_service::incref_async;

        return async(incref_async_ptr, this, raw, credit, keep_alive);
    }

    if (HPX_UNLIKELY(0 >= credit))
    {
        HPX_THROW_EXCEPTION(bad_parameter
          , "addressing_service::incref_async"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return lcos::future<std::int64_t>();
    }

    HPX_ASSERT(keep_alive != naming::invalid_id);

    typedef refcnt_requests_type::value_type mapping;

    // Some examples of calculating the compensated credits below
    //
    //  case   pending   credits   remaining   sent to   compensated
    //  no     decref              decrefs     AGAS      credits
    // ------+---------+---------+------------+--------+-------------
    //   1         0        10        0           0        10
    //   2        10         9        1           0        10
    //   3        10        10        0           0        10
    //   4        10        11        0           1        10

    std::pair<naming::gid_type, std::int64_t> pending_incref;
    bool has_pending_incref = false;
    std::int64_t pending_decrefs = 0;

    {
        std::lock_guard<mutex_type> l(refcnt_requests_mtx_);

        typedef refcnt_requests_type::iterator iterator;

        iterator matches = refcnt_requests_->find(raw);
        if (matches != refcnt_requests_->end())
        {
            pending_decrefs = matches->second;
            matches->second += credit;

            // Increment requests need to be handled immediately.

            // If the given incref was fully compensated by a pending decref
            // (i.e. match_data is less than 0) then there is no need
            // to do anything more.
            if (matches->second > 0)
            {
                // credit > decrefs (case no 4): store the remaining incref to
                // be handled below.
                pending_incref = mapping(matches->first, matches->second);
                has_pending_incref = true;

                refcnt_requests_->erase(matches);
            }
            else if (matches->second == 0)
            {
                // credit == decref (case no. 3): if the incref offsets any
                // pending decref, just remove the pending decref request.
                refcnt_requests_->erase(matches);
            }
            else
            {
                // credit < decref (case no. 2): do nothing
            }
        }
        else
        {
            // case no. 1
            pending_incref = mapping(raw, credit);
            has_pending_incref = true;
        }
    }

    if (!has_pending_incref)
    {
        // no need to talk to AGAS, acknowledge the incref immediately
        return hpx::make_ready_future(pending_decrefs);
    }

    naming::gid_type const e_lower = pending_incref.first;

    lcos::future<std::int64_t> f =
        primary_ns_.increment_credit(pending_incref.second, e_lower, e_lower);

    // pass the amount of compensated decrefs to the callback
    using util::placeholders::_1;
    return f.then(util::bind(
            util::one_shot(&addressing_service::synchronize_with_async_incref),
            this, _1, keep_alive, pending_decrefs
        ));
} // }}}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::decref(
    naming::gid_type const& gid
  , std::int64_t credit
  , error_code& ec
    )
{ // {{{ decref implementation
    naming::gid_type raw(naming::detail::get_stripped_gid(gid));

    if (HPX_UNLIKELY(nullptr == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        void (addressing_service::*decref_ptr)(
            naming::gid_type const&
          , std::int64_t
          , error_code&
        ) = &addressing_service::decref;

        threads::register_thread_nullary(
            util::deferred_call(decref_ptr, this, raw, credit, std::ref(throws)),
            "addressing_service::decref", threads::pending, true,
            threads::thread_priority_normal, std::size_t(-1),
            threads::thread_stacksize_default, ec);

        return;
    }

    if (HPX_UNLIKELY(credit <= 0))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "addressing_service::decref"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return;
    }

    try {
        std::unique_lock<mutex_type> l(refcnt_requests_mtx_);

        // Match the decref request with entries in the incref table
        typedef refcnt_requests_type::iterator iterator;
        typedef refcnt_requests_type::value_type mapping;

        iterator matches = refcnt_requests_->find(raw);
        if (matches != refcnt_requests_->end())
        {
            matches->second -= credit;
        }
        else
        {
            std::pair<iterator, bool> p =
                refcnt_requests_->insert(mapping(raw, -credit));

            if (HPX_UNLIKELY(!p.second))
            {
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "addressing_service::decref"
                  , boost::str(boost::format("couldn't insert decref request "
                        "for %1% (%2%)") % raw % credit));
                return;
            }
        }

        send_refcnt_requests(l, ec);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::decref");
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::register_name(
    std::string const& name
  , naming::gid_type const& id
  , error_code& ec
    )
{ // {{{
    try {
        return symbol_ns_.bind(name, naming::detail::get_stripped_gid(id));
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_name");
        return false;
    }
} // }}}

static bool correct_credit_on_failure(future<bool> f, naming::id_type id,
    std::int64_t mutable_gid_credit, std::int64_t new_gid_credit)
{
    // Return the credit to the GID if the operation failed
    if ((f.has_exception() && mutable_gid_credit != 0) || !f.get())
    {
        naming::detail::add_credit_to_gid(id.get_gid(), new_gid_credit);
        return false;
    }
    return true;
}

lcos::future<bool> addressing_service::register_name_async(
    std::string const& name
  , naming::id_type const& id
    )
{ // {{{
    // We need to modify the reference count.
    naming::gid_type& mutable_gid = const_cast<naming::id_type&>(id).get_gid();
    naming::gid_type new_gid = naming::detail::split_gid_if_needed(mutable_gid).get();

    future<bool> f = symbol_ns_.bind_async(name, new_gid);

    std::int64_t new_credit = naming::detail::get_credit_from_gid(new_gid);
    if (new_credit != 0)
    {
        using util::placeholders::_1;
        return f.then(util::bind(
                util::one_shot(&correct_credit_on_failure),
                _1, id, std::int64_t(HPX_GLOBALCREDIT_INITIAL), new_credit
            ));
    }

    return f;
} // }}}

///////////////////////////////////////////////////////////////////////////////
naming::id_type addressing_service::unregister_name(
    std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        return symbol_ns_.unbind(name);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unregister_name");
        return naming::invalid_id;
    }
} // }}}

lcos::future<naming::id_type> addressing_service::unregister_name_async(
    std::string const& name
    )
{ // {{{
    return symbol_ns_.unbind_async(name);
} // }}}

///////////////////////////////////////////////////////////////////////////////
naming::id_type addressing_service::resolve_name(
    std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        return symbol_ns_.resolve(name);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_name");
        return naming::invalid_id;
    }
} // }}}

lcos::future<naming::id_type> addressing_service::resolve_name_async(
    std::string const& name
    )
{ // {{{
    return symbol_ns_.resolve_async(name);
} // }}}

namespace detail
{
    hpx::future<hpx::id_type> on_register_event(hpx::future<bool> f,
        hpx::future<hpx::id_type> result_f)
    {
        if (!f.get())
        {
            HPX_THROW_EXCEPTION(bad_request,
                "hpx::agas::detail::on_register_event",
                "request 'symbol_ns_on_event' failed");
            return hpx::future<hpx::id_type>();
        }

        return result_f;
    }
}

future<hpx::id_type> addressing_service::on_symbol_namespace_event(
    std::string const& name, bool call_for_past_events)
{
    lcos::promise<naming::id_type, naming::gid_type> p;
    auto result_f = p.get_future();

    hpx::future<bool> f = symbol_ns_.on_event(name, call_for_past_events, p.get_id());

    using util::placeholders::_1;
    return f.then(util::bind(
            util::one_shot(&detail::on_register_event), _1, std::move(result_f)
        ));
}

}}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::agas::server::symbol_namespace::on_event_action
    symbol_namespace_on_event_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(symbol_namespace_on_event_action,
    symbol_namespace_on_event_action)
HPX_REGISTER_BROADCAST_ACTION_ID(symbol_namespace_on_event_action,
    symbol_namespace_on_event_action,
    hpx::actions::broadcast_symbol_namespace_on_event_action_id)

namespace hpx { namespace agas
{
    namespace detail
    {
        std::vector<hpx::id_type> find_all_symbol_namespace_services()
        {
            std::vector<hpx::id_type> ids;
            for (hpx::id_type const& id : hpx::find_all_localities())
            {
                ids.push_back(hpx::id_type(
                    agas::symbol_namespace::get_service_instance(id),
                    id_type::unmanaged));
            }
            return ids;
        }
    }

/// Invoke the supplied hpx::function for every registered global name
bool addressing_service::iterate_ids(
    iterate_names_function_type const& f
  , error_code& ec
    )
{ // {{{
    try {
        server::symbol_namespace::iterate_action act;
        lcos::broadcast(act, detail::find_all_symbol_namespace_services(), f).get(ec);

        return !ec;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_ids");
        return false;
    }
} // }}}

// This function has to return false if the key is already in the cache (true
// means go ahead with the cache update).
bool check_for_collisions(
    addressing_service::gva_cache_key const& new_key
  , addressing_service::gva_cache_key const& old_key
    )
{
    return (new_key.get_gid() != old_key.get_gid())
        || (new_key.get_count() != old_key.get_count());
}

void addressing_service::update_cache_entry(
    naming::gid_type const& id
  , gva const& g
  , error_code& ec
    )
{ // {{{
    if (!caching_)
    {
        // If caching is disabled, we silently pretend success.
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    // don't look at cache if id is marked as non-cache-able
    if (!naming::detail::store_in_cache(id))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    naming::gid_type gid = naming::detail::get_stripped_gid(id);

    // don't look at the cache if the id is locally managed
    if (naming::get_locality_id_from_gid(gid) ==
        naming::get_locality_id_from_gid(locality_))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    if(hpx::threads::get_self_ptr() == nullptr)
    {
        // Don't update the cache while HPX is starting up ...
        if(hpx::is_starting())
        {
            return;
        }
        void (addressing_service::*update_cache_entry_ptr)(
            naming::gid_type const&
          , gva const &
          , error_code&
        ) = &addressing_service::update_cache_entry;
        threads::register_thread_nullary(
            util::deferred_call(update_cache_entry_ptr, this, id, g, std::ref(throws)),
            "addressing_service::update_cache_entry", threads::pending, true,
            threads::thread_priority_normal, std::size_t(-1),
            threads::thread_stacksize_default, ec);
    }

    try {
        // The entry in AGAS for a locality's RTS component has a count of 0,
        // so we convert it to 1 here so that the cache doesn't break.
        const std::uint64_t count = (g.count ? g.count : 1);

        LAGAS_(debug) <<
            ( boost::format(
            "addressing_service::update_cache_entry, gid(%1%), count(%2%)"
            ) % gid % count);

        const gva_cache_key key(gid, count);

        {
            std::unique_lock<mutex_type> lock(gva_cache_mtx_);
            if (!gva_cache_->update_if(key, g, check_for_collisions))
            {
                if (LAGAS_ENABLED(warning))
                {
                    // Figure out who we collided with.
                    addressing_service::gva_cache_key idbase;
                    addressing_service::gva_cache_type::entry_type e;

                    if (!gva_cache_->get_entry(key, idbase, e))
                    {
                        // This is impossible under sane conditions.
                        lock.unlock();
                        HPX_THROWS_IF(ec, invalid_data
                          , "addressing_service::update_cache_entry"
                          , "data corruption or lock error occurred in cache");
                        return;
                    }

                    LAGAS_(warning) <<
                        ( boost::format(
                            "addressing_service::update_cache_entry, "
                            "aborting update due to key collision in cache, "
                            "new_gid(%1%), new_count(%2%), old_gid(%3%), old_count(%4%)"
                        ) % gid % count % idbase.get_gid() % idbase.get_count());
                }
            }
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::update_cache_entry");
    }
} // }}}

bool addressing_service::get_cache_entry(
    naming::gid_type const& gid
  , gva& gva
  , naming::gid_type& idbase
  , error_code& ec
    )
{
    // Don't use the cache while HPX is starting up
    if(hpx::is_starting())
    {
        return false;
    }
    HPX_ASSERT(hpx::threads::get_self_ptr());
    gva_cache_key k(gid);
    gva_cache_key idbase_key;

    std::unique_lock<mutex_type> lock(gva_cache_mtx_);
    if(gva_cache_->get_entry(k, idbase_key, gva))
    {
        const std::uint64_t id_msb =
            naming::detail::strip_internal_bits_from_gid(gid.get_msb());

        if (HPX_UNLIKELY(id_msb != idbase_key.get_gid().get_msb()))
        {
            lock.unlock();
            HPX_THROWS_IF(ec, internal_server_error
              , "addressing_service::get_cache_entry"
              , "bad entry in cache, MSBs of GID base and GID do not match");
            return false;
        }
        idbase = idbase_key.get_gid();
        return true;
    }

    return false;
}


void addressing_service::clear_cache(
    error_code& ec
    )
{ // {{{
    if (!caching_)
    {
        // If caching is disabled, we silently pretend success.
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    try {
        LAGAS_(warning) << "addressing_service::clear_cache, clearing cache";

        std::lock_guard<mutex_type> lock(gva_cache_mtx_);

        gva_cache_->clear();

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::clear_cache");
    }
} // }}}

void addressing_service::remove_cache_entry(
    naming::gid_type const& id
  , error_code& ec
    )
{
    // If caching is disabled, we silently pretend success.
    if (!caching_)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    // don't look at cache if id is marked as non-cache-able
    if (!naming::detail::store_in_cache(id))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    naming::gid_type gid = naming::detail::get_stripped_gid(id);

    // don't look at the cache if the id is locally managed
    if (naming::get_locality_id_from_gid(gid) ==
        naming::get_locality_id_from_gid(locality_))
    {
        if (&ec != &throws)
            ec = make_success_code();
        return;
    }

    try {
        LAGAS_(warning) << "addressing_service::remove_cache_entry";

        std::lock_guard<mutex_type> lock(gva_cache_mtx_);

        gva_cache_->erase(
            [&gid](std::pair<gva_cache_key, gva> const& p)
            {
                return gid == p.first.get_gid();
            });

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::clear_cache");
    }
}

// Disable refcnt caching during shutdown
void addressing_service::start_shutdown(error_code& ec)
{
    // If caching is disabled, we silently pretend success.
    if (!caching_)
        return;

    std::unique_lock<mutex_type> l(refcnt_requests_mtx_);
    enable_refcnt_caching_ = false;
    send_refcnt_requests_sync(l, ec);
}

namespace detail
{
    // get action code from counter type
    namespace_action_code retrieve_action_code(
        std::string const& name
      , error_code& ec
        )
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec) return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0;
             i != num_component_namespace_services;
             ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].code_;
        }

        // locality_ns
        for (std::size_t i = 0;
             i != num_locality_namespace_services;
             ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].code_;
        }

        // primary_ns
        for (std::size_t i = 0;
             i != num_primary_namespace_services;
             ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].code_;
        }

        // symbol_ns
        for (std::size_t i = 0;
             i != num_symbol_namespace_services;
             ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }

    // get service action code from counter type
    namespace_action_code retrieve_action_service_code(
        std::string const& name
      , error_code& ec
        )
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec) return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0;
             i != num_component_namespace_services;
             ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].service_code_;
        }

        // locality_ns
        for (std::size_t i = 0;
             i != num_locality_namespace_services;
             ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].service_code_;
        }

        // primary_ns
        for (std::size_t i = 0;
             i != num_primary_namespace_services;
             ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].service_code_;
        }

        // symbol_ns
        for (std::size_t i = 0;
             i != num_symbol_namespace_services;
             ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].service_code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions to access the current cache statistics
std::uint64_t addressing_service::get_cache_entries(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->size();
}

std::uint64_t addressing_service::get_cache_hits(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().hits(reset);
}

std::uint64_t addressing_service::get_cache_misses(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().misses(reset);
}

std::uint64_t addressing_service::get_cache_evictions(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().evictions(reset);
}

std::uint64_t addressing_service::get_cache_insertions(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().insertions(reset);
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t addressing_service::get_cache_get_entry_count(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_get_entry_count(reset);
}

std::uint64_t addressing_service::get_cache_insertion_entry_count(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_insert_entry_count(reset);
}

std::uint64_t addressing_service::get_cache_update_entry_count(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_update_entry_count(reset);
}

std::uint64_t addressing_service::get_cache_erase_entry_count(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_erase_entry_count(reset);
}

std::uint64_t addressing_service::get_cache_get_entry_time(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_get_entry_time(reset);
}

std::uint64_t addressing_service::get_cache_insertion_entry_time(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_insert_entry_time(reset);
}

std::uint64_t addressing_service::get_cache_update_entry_time(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_update_entry_time(reset);
}

std::uint64_t addressing_service::get_cache_erase_entry_time(bool reset)
{
    std::lock_guard<mutex_type> lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_erase_entry_time(reset);
}

/// Install performance counter types exposing properties from the local cache.
void addressing_service::register_counter_types()
{ // {{{
    using util::placeholders::_1;
    using util::placeholders::_2;

    // install
    util::function_nonser<std::int64_t(bool)> cache_entries(
        util::bind(&addressing_service::get_cache_entries, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_hits(
        util::bind(&addressing_service::get_cache_hits, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_misses(
        util::bind(&addressing_service::get_cache_misses, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_evictions(
        util::bind(&addressing_service::get_cache_evictions, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_insertions(
        util::bind(&addressing_service::get_cache_insertions, this, _1));

    util::function_nonser<std::int64_t(bool)> cache_get_entry_count(
        util::bind(
            &addressing_service::get_cache_get_entry_count, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_insertion_count(
        util::bind(
            &addressing_service::get_cache_insertion_entry_count, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_update_entry_count(
        util::bind(
            &addressing_service::get_cache_update_entry_count, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_erase_entry_count(
        util::bind(
            &addressing_service::get_cache_erase_entry_count, this, _1));

    util::function_nonser<std::int64_t(bool)> cache_get_entry_time(
        util::bind(
            &addressing_service::get_cache_get_entry_time, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_insertion_time(
        util::bind(
            &addressing_service::get_cache_insertion_entry_time, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_update_entry_time(
        util::bind(
            &addressing_service::get_cache_update_entry_time, this, _1));
    util::function_nonser<std::int64_t(bool)> cache_erase_entry_time(
        util::bind(
            &addressing_service::get_cache_erase_entry_time, this, _1));

    performance_counters::generic_counter_type_data const counter_types[] =
    {
        { "/agas/count/cache/entries", performance_counters::counter_raw,
          "returns the number of cache entries in the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_entries, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/hits", performance_counters::counter_raw,
          "returns the number of cache hits while accessing the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_hits, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/misses", performance_counters::counter_raw,
          "returns the number of cache misses while accessing the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_misses, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/evictions", performance_counters::counter_raw,
          "returns the number of cache evictions from the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_evictions, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/insertions", performance_counters::counter_raw,
          "returns the number of cache insertions into the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insertions, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/get_entry", performance_counters::counter_raw,
          "returns the number of invocations of get_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_get_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/insert_entry", performance_counters::counter_raw,
          "returns the number of invocations of insert function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insertion_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/update_entry", performance_counters::counter_raw,
          "returns the number of invocations of update_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_update_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache/erase_entry", performance_counters::counter_raw,
          "returns the number of invocations of erase_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_erase_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/time/cache/get_entry", performance_counters::counter_raw,
          "returns the overall time spent executing of the get_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_get_entry_time, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
        },
        { "/agas/time/cache/insert_entry", performance_counters::counter_raw,
          "returns the overall time spent executing of the insert_entry API "
              "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insertion_time, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/time/cache/update_entry", performance_counters::counter_raw,
          "returns the overall time spent executing of the update_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_update_entry_time, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
        },
        { "/agas/time/cache/erase_entry", performance_counters::counter_raw,
          "returns the overall time spent executing of the erase_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          util::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_erase_entry_time, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
    };
    performance_counters::install_counter_types(
        counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

    // install counters for services
    primary_ns_.register_counter_types();
    component_ns_->register_counter_types();
    locality_ns_->register_counter_types();
    symbol_ns_.register_counter_types();

    // register root server
    std::uint32_t locality_id =
        naming::get_locality_id_from_gid(get_local_locality());
    locality_ns_->register_server_instance(locality_id);
    primary_ns_.register_server_instance(locality_id);
    component_ns_->register_server_instance(locality_id);
    symbol_ns_.register_server_instance(locality_id);
} // }}}

void addressing_service::garbage_collect_non_blocking(
    error_code& ec
    )
{
    std::unique_lock<mutex_type> l(refcnt_requests_mtx_, std::try_to_lock);
    if (!l.owns_lock()) return;     // no need to compete for garbage collection

    send_refcnt_requests_non_blocking(l, ec);
}

void addressing_service::garbage_collect(
    error_code& ec
    )
{
    std::unique_lock<mutex_type> l(refcnt_requests_mtx_, std::try_to_lock);
    if (!l.owns_lock()) return;     // no need to compete for garbage collection

    send_refcnt_requests_sync(l, ec);
}

void addressing_service::send_refcnt_requests(
    std::unique_lock<addressing_service::mutex_type>& l
  , error_code& ec
    )
{
    if (!l.owns_lock())
    {
        HPX_THROWS_IF(ec, lock_error
          , "addressing_service::send_refcnt_requests"
          , "mutex is not locked");
        return;
    }

    if (!enable_refcnt_caching_ || max_refcnt_requests_ == ++refcnt_requests_count_)
        send_refcnt_requests_non_blocking(l, ec);

    else if (&ec != &throws)
        ec = make_success_code();
}

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
    void dump_refcnt_requests(
        std::unique_lock<addressing_service::mutex_type>& l
      , addressing_service::refcnt_requests_type const& requests
      , const char* func_name
        )
    {
        HPX_ASSERT(l.owns_lock());

        std::stringstream ss;
        ss << ( boost::format(
              "%1%, dumping client-side refcnt table, requests(%2%):")
              % func_name % requests.size());

        typedef addressing_service::refcnt_requests_type::const_reference
            const_reference;

        for (const_reference e : requests)
        {
            // The [client] tag is in there to make it easier to filter
            // through the logs.
            ss << ( boost::format(
                  "\n  [client] gid(%1%), credits(%2%)")
                  % e.first
                  % e.second);
        }

        LAGAS_(debug) << ss.str();
    }
#endif

void addressing_service::send_refcnt_requests_non_blocking(
    std::unique_lock<addressing_service::mutex_type>& l
  , error_code& ec
    )
{
    HPX_ASSERT(l.owns_lock());

    try {
        if (refcnt_requests_->empty())
        {
            l.unlock();
            return;
        }

        std::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

        p.swap(refcnt_requests_);
        refcnt_requests_count_ = 0;

        l.unlock();

        LAGAS_(info) << (boost::format(
            "addressing_service::send_refcnt_requests_non_blocking, "
            "requests(%1%)")
            % p->size());

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
        if (LAGAS_ENABLED(debug))
            dump_refcnt_requests(l, *p,
                "addressing_service::send_refcnt_requests_non_blocking");
#endif

        // collect all requests for each locality
        typedef
            std::map<
                naming::id_type,
                std::vector<
                    hpx::util::tuple<std::int64_t, naming::gid_type, naming::gid_type>
                >
            >
            requests_type;
        requests_type requests;

        for (refcnt_requests_type::const_reference e : *p)
        {
            HPX_ASSERT(e.second < 0);

            naming::gid_type raw(e.first);

            naming::id_type target(
                primary_namespace::get_service_instance(raw)
              , naming::id_type::unmanaged);

            requests[target].push_back(hpx::util::make_tuple(e.second, raw, raw));
        }

        // send requests to all locality
        requests_type::iterator end = requests.end();
        for (requests_type::iterator it = requests.begin(); it != end; ++it)
        {
            server::primary_namespace::decrement_credit_action action;
            hpx::apply(action, std::move(it->first), std::move(it->second));
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        l.unlock();
        HPX_RETHROWS_IF(ec, e,
            "addressing_service::send_refcnt_requests_non_blocking");
    }
}

std::vector<hpx::future<std::vector<std::int64_t> > >
addressing_service::send_refcnt_requests_async(
    std::unique_lock<addressing_service::mutex_type>& l
    )
{
    HPX_ASSERT(l.owns_lock());

    if (refcnt_requests_->empty())
    {
        l.unlock();
        return std::vector<hpx::future<std::vector<std::int64_t> > >();
    }

    std::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

    p.swap(refcnt_requests_);
    refcnt_requests_count_ = 0;

    l.unlock();

    LAGAS_(info) << (boost::format(
        "addressing_service::send_refcnt_requests_async, "
        "requests(%1%)")
        % p->size());

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
    if (LAGAS_ENABLED(debug))
        dump_refcnt_requests(l, *p,
            "addressing_service::send_refcnt_requests_sync");
#endif

    // collect all requests for each locality
    typedef
        std::map<
            naming::id_type,
            std::vector<
                hpx::util::tuple<std::int64_t, naming::gid_type, naming::gid_type>
            >
        >
        requests_type;
    requests_type requests;

    std::vector<hpx::future<std::vector<std::int64_t> > > lazy_results;
    for (refcnt_requests_type::const_reference e : *p)
    {
        HPX_ASSERT(e.second < 0);

        naming::gid_type raw(e.first);

        naming::id_type target(
            primary_namespace::get_service_instance(raw)
          , naming::id_type::unmanaged);

        requests[target].push_back(hpx::util::make_tuple(e.second, raw, raw));
    }

    // send requests to all locality
    requests_type::const_iterator end = requests.end();
    for (requests_type::const_iterator it = requests.begin(); it != end; ++it)
    {
        server::primary_namespace::decrement_credit_action action;
        lazy_results.push_back(
            hpx::async(action, std::move(it->first), std::move(it->second)));
    }

    return lazy_results;
}

void addressing_service::send_refcnt_requests_sync(
    std::unique_lock<addressing_service::mutex_type>& l
  , error_code& ec
    )
{
    std::vector<hpx::future<std::vector<std::int64_t> > > lazy_results =
        send_refcnt_requests_async(l);

    // re throw possible errors
    when_all(lazy_results).get();

    if (&ec != &throws)
        ec = make_success_code();
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<void> addressing_service::mark_as_migrated(
    naming::gid_type const& gid_
  , util::unique_function_nonser<std::pair<bool, hpx::future<void> >()> && f //-V669
    )
{
    if (!gid_)
    {
        return hpx::make_exceptional_future<void>(
            HPX_GET_EXCEPTION(bad_parameter,
                "addressing_service::mark_as_migrated",
                "invalid reference gid"));
    }

    naming::gid_type gid(naming::detail::get_stripped_gid(gid_));

    // Always first grab the AGAS lock before invoking the user supplied
    // function. The user supplied code will grab another lock. Both locks have
    // to be acquired and always in the same sequence.
    // The AGAS lock needs to be acquired first as the migrated object might
    // not exist on this locality, in which case it should not be accessed
    // anymore. The only way to determine whether the object still exists on
    // this locality is to query the migrated objects table in AGAS.
    typedef std::unique_lock<mutex_type> lock_type;

    lock_type lock(migrated_objects_mtx_);
    util::ignore_while_checking<lock_type> ignore(&lock);

    // call the user code for the component instance to be migrated, the
    // returned future becomes ready whenever the component instance can be
    // migrated (no threads are pending/active any more)
    std::pair<bool, hpx::future<void> > result = f();

    // mark the gid as 'migrated' right away - the worst what can happen is
    // that a parcel which comes in for this object is bouncing between this
    // locality and the locality managing the address resolution for the object
    if (result.first)
    {
        migrated_objects_table_type::iterator it =
            migrated_objects_table_.find(gid);

        // insert the object into the map of migrated objects
        if (it == migrated_objects_table_.end())
            migrated_objects_table_.insert(gid);

        // avoid interactions with the locking in the cache
        lock.unlock();

        // remove entry from cache
        remove_cache_entry(gid_);
    }

    return std::move(result.second);
}

void addressing_service::unmark_as_migrated(
    naming::gid_type const& gid_
    )
{
    if (!gid_)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::unmark_as_migrated",
            "invalid reference gid");
        return;
    }

    naming::gid_type gid(naming::detail::get_stripped_gid(gid_));

    std::unique_lock<mutex_type> lock(migrated_objects_mtx_);

    migrated_objects_table_type::iterator it =
        migrated_objects_table_.find(gid);

    // insert the object into the map of migrated objects
    if (it != migrated_objects_table_.end())
    {
        migrated_objects_table_.erase(it);

        // remove entry from cache
        if (caching_ && naming::detail::store_in_cache(gid_))
        {
            // avoid interactions with the locking in the cache
            lock.unlock();

            // remove entry from cache
            remove_cache_entry(gid_);
        }
    }
}

hpx::future<std::pair<naming::id_type, naming::address> >
addressing_service::begin_migration_async(naming::id_type const& id)
{
    typedef std::pair<naming::id_type, naming::address> result_type;

    if (!id)
    {
        return hpx::make_exceptional_future<result_type>(
            HPX_GET_EXCEPTION(bad_parameter,
                "addressing_service::begin_migration_async",
                "invalid reference id"));
    }

    naming::gid_type gid(naming::detail::get_stripped_gid(id.get_gid()));

    return primary_ns_.begin_migration(gid);
}

hpx::future<bool> addressing_service::end_migration_async(
    naming::id_type const& id
    )
{
    if (!id)
    {
        return hpx::make_exceptional_future<bool>(
            HPX_GET_EXCEPTION(bad_parameter,
                "addressing_service::end_migration_async",
                "invalid reference id"));
    }

    naming::gid_type gid(naming::detail::get_stripped_gid(id.get_gid()));

    return primary_ns_.end_migration(gid);
}

bool addressing_service::was_object_migrated_locked(
    naming::gid_type const& gid_
    )
{
    naming::gid_type gid(naming::detail::get_stripped_gid(gid_));

    return
        migrated_objects_table_.find(gid) !=
        migrated_objects_table_.end();
}

std::pair<bool, components::pinned_ptr>
    addressing_service::was_object_migrated(
        naming::gid_type const& gid
      , util::unique_function_nonser<components::pinned_ptr()> && f //-V669
        )
{
    if (!gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::was_object_migrated",
            "invalid reference gid");
        return std::make_pair(false, components::pinned_ptr());
    }

    // Always first grab the AGAS lock before invoking the user supplied
    // function. The user supplied code will grab another lock. Both locks have
    // to be acquired and always in the same sequence.
    // The AGAS lock needs to be acquired first as the migrated object might
    // not exist on this locality, in which case it should not be accessed
    // anymore. The only way to determine whether the object still exists on
    // this locality is to query the migrated objects table in AGAS.
    typedef std::unique_lock<mutex_type> lock_type;

    lock_type lock(migrated_objects_mtx_);

    if (was_object_migrated_locked(gid))
        return std::make_pair(true, components::pinned_ptr());

    util::ignore_while_checking<lock_type> ignore(&lock);
    return std::make_pair(false, f());
}

}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        std::string name_from_basename(std::string const& basename,
            std::size_t idx)
        {
            HPX_ASSERT(!basename.empty());

            std::string name;

            if (basename[0] != '/')
                name += '/';

            name += basename;
            if (name[name.size()-1] != '/')
                name += '/';
            name += std::to_string(idx);

            return name;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<hpx::id_type> >
        find_all_from_basename(std::string const& basename, std::size_t num_ids)
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_all_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type> > results;
        for(std::size_t i = 0; i != num_ids; ++i)
        {
            std::string name = detail::name_from_basename(basename, i);
            results.push_back(agas::on_symbol_namespace_event(
                std::move(name), true));
        }
        return results;
    }

    std::vector<hpx::future<hpx::id_type> >
        find_from_basename(std::string const& basename,
            std::vector<std::size_t> const& ids)
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type> > results;
        for (std::size_t i : ids)
        {
            std::string name = detail::name_from_basename(basename, i); //-V106
            results.push_back(agas::on_symbol_namespace_event(
                std::move(name), true));
        }
        return results;
    }

    hpx::future<hpx::id_type> find_from_basename(std::string const& basename,
        std::size_t sequence_nr)
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_from_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::on_symbol_namespace_event(std::move(name), true);
    }

    hpx::future<bool> register_with_basename(std::string const& basename,
        hpx::id_type id, std::size_t sequence_nr)
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::register_with_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::register_name(std::move(name), id);
    }

    hpx::future<hpx::id_type> unregister_with_basename(
        std::string const& basename, std::size_t sequence_nr)
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::unregister_with_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::unregister_name(std::move(name));
    }
}
