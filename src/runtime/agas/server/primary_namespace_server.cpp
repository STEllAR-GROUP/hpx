////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 408000
#  include <memory>
#endif

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hpx { namespace agas
{

naming::gid_type bootstrap_primary_namespace_gid()
{
    return naming::gid_type(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB);
}

naming::id_type bootstrap_primary_namespace_id()
{
    return naming::id_type(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB
      , naming::id_type::unmanaged);
}

namespace server
{

// TODO: This isn't scalable, we have to update it every time we add a new
// AGAS request/response type.
response primary_namespace::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    switch (req.get_action_code())
    {
        case primary_ns_route:
            {
                HPX_ASSERT(false);
                return response();
            }
        case primary_ns_bind_gid:
            {
                detail::update_time_on_exit update(
                    counter_data_.bind_gid_.time_
                );
                counter_data_.increment_bind_gid_count();
                return bind_gid(req, ec);
            }
        case primary_ns_resolve_gid:
            {
                detail::update_time_on_exit update(
                    counter_data_.resolve_gid_.time_
                );
                counter_data_.increment_resolve_gid_count();
                return resolve_gid(req, ec);
            }
        case primary_ns_unbind_gid:
            {
                detail::update_time_on_exit update(
                    counter_data_.unbind_gid_.time_
                );
                counter_data_.increment_unbind_gid_count();
                return unbind_gid(req, ec);
            }
        case primary_ns_increment_credit:
            {
                detail::update_time_on_exit update(
                    counter_data_.increment_credit_.time_
                );
                counter_data_.increment_increment_credit_count();
                return increment_credit(req, ec);
            }
        case primary_ns_decrement_credit:
            {
                detail::update_time_on_exit update(
                    counter_data_.decrement_credit_.time_
                );
                counter_data_.increment_decrement_credit_count();
                return decrement_credit(req, ec);
            }
        case primary_ns_allocate:
            {
                detail::update_time_on_exit update(
                    counter_data_.allocate_.time_
                );
                counter_data_.increment_allocate_count();
                return allocate(req, ec);
            }
        case primary_ns_begin_migration:
            {
                detail::update_time_on_exit update(
                    counter_data_.begin_migration_.time_
                );
                counter_data_.increment_begin_migration_count();
                return begin_migration(req, ec);
            }
        case primary_ns_end_migration:
            {
                detail::update_time_on_exit update(
                    counter_data_.end_migration_.time_
                );
                counter_data_.increment_end_migration_count();
                return end_migration(req, ec);
            }
        case primary_ns_statistics_counter:
            return statistics_counter(req, ec);

        case locality_ns_allocate:
        case locality_ns_free:
        case locality_ns_localities:
        case locality_ns_num_localities:
        case locality_ns_num_threads:
        case locality_ns_resolve_locality:
        case locality_ns_resolved_localities:
        {
            LAGAS_(warning) <<
                "primary_namespace::service, redirecting request to "
                "locality_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case component_ns_bind_prefix:
        case component_ns_bind_name:
        case component_ns_resolve_id:
        case component_ns_unbind_name:
        case component_ns_iterate_types:
        case component_ns_get_component_type_name:
        case component_ns_num_localities:
        {
            LAGAS_(warning) <<
                "primary_namespace::service, redirecting request to "
                "component_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case symbol_ns_bind:
        case symbol_ns_resolve:
        case symbol_ns_unbind:
        case symbol_ns_iterate_names:
        case symbol_ns_on_event:
        {
            LAGAS_(warning) <<
                "primary_namespace::service, redirecting request to "
                "symbol_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case locality_ns_service:
        case component_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "primary_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    }
} // }}}

// register all performance counter types exposed by this component
void primary_namespace::register_counter_types(
    error_code& ec
    )
{
    using util::placeholders::_1;
    using util::placeholders::_2;
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    performance_counters::create_counter_func creator(
        util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::primary_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_primary_namespace_services;
          ++i)
    {
        // global counters are handled elsewhere
        if (detail::primary_namespace_services[i].code_ == primary_ns_statistics_counter)
            continue;

        std::string name(detail::primary_namespace_services[i].name_);
        std::string help;
        std::string::size_type p = name.find_last_of('/');
        HPX_ASSERT(p != std::string::npos);

        if (detail::primary_namespace_services[i].target_
            == detail::counter_target_count)
            help = boost::str(help_count % name.substr(p+1));
        else
            help = boost::str(help_time % name.substr(p+1));

        performance_counters::install_counter_type(
            agas::performance_counter_basename + name
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::primary_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void primary_namespace::register_global_counter_types(
    error_code& ec
    )
{
    using util::placeholders::_1;
    using util::placeholders::_2;
    performance_counters::create_counter_func creator(
        util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::primary_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_primary_namespace_services;
          ++i)
    {
        // local counters are handled elsewhere
        if (detail::primary_namespace_services[i].code_
            != primary_ns_statistics_counter)
            continue;

        std::string help;
        if (detail::primary_namespace_services[i].target_
            == detail::counter_target_count)
            help = "returns the overall number of invocations \
                     of all primary AGAS services";
        else
            help = "returns the overall execution time of all primary AGAS services";

        performance_counters::install_counter_type(
            std::string(agas::performance_counter_basename) +
                detail::primary_namespace_services[i].name_
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::primary_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void primary_namespace::register_server_instance(
    char const* servicename
  , boost::uint32_t locality_id
  , error_code& ec
    )
{
    // set locality_id for this component
    if (locality_id == naming::invalid_locality_id)
        locality_id = 0;        // if not given, we're on the root

    this->base_type::set_locality_id(locality_id);

    // now register this AGAS instance with AGAS :-P
    instance_name_ = agas::service_name;
    instance_name_ += servicename;
    instance_name_ += agas::server::primary_namespace_service_name;

    // register a gid (not the id) to avoid AGAS holding a reference to this
    // component
    agas::register_name_sync(instance_name_, get_unmanaged_id().get_gid(), ec);
}

void primary_namespace::unregister_server_instance(
    error_code& ec
    )
{
    agas::unregister_name_sync(instance_name_, ec);
    this->base_type::finalize();
}

void primary_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name_sync(instance_name_, ec);
    }
}

// Parcel routing forwards the message handler request to the routed action
parcelset::policies::message_handler* primary_namespace::get_message_handler(
    parcelset::parcelhandler* ph
  , parcelset::locality const& loc
  , parcelset::parcel const& p
    )
{
    typedef hpx::actions::transfer_action<
        server::primary_namespace::route_action
    > action_type;

    action_type * act = static_cast<action_type *>(p.get_action());

    parcelset::parcel const& routed_p = hpx::actions::get<0>(*act);
    return routed_p.get_message_handler(ph, loc);
}

serialization::binary_filter* primary_namespace::get_serialization_filter(
    parcelset::parcel const& p
    )
{
    typedef hpx::actions::transfer_action<
        server::primary_namespace::route_action
    > action_type;

    action_type * act = static_cast<action_type *>(p.get_action());

    parcelset::parcel const& routed_p = hpx::actions::get<0>(*act);
    return routed_p.get_serialization_filter();
}

// TODO: do/undo semantics (e.g. transactions)
std::vector<response> primary_namespace::bulk_service(
    std::vector<request> const& reqs
  , error_code& ec
    )
{
    std::vector<response> r;
    r.reserve(reqs.size());

    for (request const& req : reqs)
    {
        r.push_back(service(req, ec));
        if (ec)
            break;      // on error: for now stop iterating
    }

    return r;
}

// start migration of the given object
response primary_namespace::begin_migration(
    request const& req
  , error_code& ec)
{
    using hpx::util::get;

    naming::gid_type id = req.get_gid();

    std::unique_lock<mutex_type> l(mutex_);

    resolved_type r = resolve_gid_locked(l, id, ec);
    if (get<0>(r) == naming::invalid_gid)
    {
        l.unlock();

        LAGAS_(info) << (boost::format(
            "primary_namespace::begin_migration, gid(%1%), response(no_success)")
            % id);

        return response(primary_ns_begin_migration, naming::invalid_gid, gva(),
            naming::invalid_gid, no_success);
    }

    migration_table_type::iterator it = migrating_objects_.find(id);
    if (it == migrating_objects_.end())
    {
        std::pair<migration_table_type::iterator, bool> p =
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
            migrating_objects_.emplace(std::piecewise_construct,
                std::forward_as_tuple(id), std::forward_as_tuple());
#else
            migrating_objects_.insert(migration_table_type::value_type(
                id,
                hpx::util::make_tuple(
                    false, 0,
                    std::make_shared<lcos::local::condition_variable_any>()
                )
            ));
#endif
        HPX_ASSERT(p.second);
        it = p.first;
    }

    // flag this id as being migrated
    hpx::util::get<0>(it->second) = true; //-V601

    return response(primary_ns_begin_migration, get<0>(r), get<1>(r), get<2>(r));
}

// migration of the given object is complete
response primary_namespace::end_migration(
    request const& req
  , error_code& ec)
{
    naming::gid_type id = req.get_gid();

    std::lock_guard<mutex_type> l(mutex_);

    using hpx::util::get;

    migration_table_type::iterator it = migrating_objects_.find(id);
    if (it == migrating_objects_.end() || !get<0>(it->second))
        return response(primary_ns_end_migration, no_success);

#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
    get<2>(it->second).notify_all(ec);
#else
    get<2>(it->second)->notify_all(ec);
#endif

    // flag this id as not being migrated anymore
    get<0>(it->second) = false;

    return response(primary_ns_end_migration, success);
}

// wait if given object is currently being migrated
void primary_namespace::wait_for_migration_locked(
    std::unique_lock<mutex_type>& l
  , naming::gid_type id
  , error_code& ec)
{
    HPX_ASSERT_OWNS_LOCK(l);

    using hpx::util::get;

    migration_table_type::iterator it = migrating_objects_.find(id);
    if (it != migrating_objects_.end() && get<0>(it->second))
    {
        ++get<1>(it->second);

#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
        get<2>(it->second).wait(l, ec);
#else
        get<2>(it->second)->wait(l, ec);
#endif

        if (--get<1>(it->second) == 0 && !get<0>(it->second))
            migrating_objects_.erase(it);
    }
}

response primary_namespace::bind_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ bind_gid implementation
    using hpx::util::get;

    // parameters
    gva g = req.get_gva();
    naming::gid_type id = req.get_gid();
    naming::gid_type locality = req.get_locality();

    naming::detail::strip_internal_bits_from_gid(id);

    std::unique_lock<mutex_type> l(mutex_);

    gva_table_type::iterator it = gvas_.lower_bound(id)
                           , begin = gvas_.begin()
                           , end = gvas_.end();

    if (it != end)
    {
        // If we got an exact match, this is a request to update an existing
        // binding (e.g. move semantics).
        if (it->first == id)
        {
            gva& gaddr = it->second.first;
            naming::gid_type& loc = it->second.second;

            // Check for count mismatch (we can't change block sizes of
            // existing bindings).
            if (HPX_UNLIKELY(gaddr.count != g.count))
            {
                // REVIEW: Is this the right error code to use?
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , "cannot change block size of existing binding");
                return response();
            }

            if (HPX_UNLIKELY(components::component_invalid == g.type))
            {
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , boost::str(boost::format(
                        "attempt to update a GVA with an invalid type, "
                        "gid(%1%), gva(%2%), locality(%3%)")
                        % id % g % locality));
                return response();
            }

            if (HPX_UNLIKELY(!locality))
            {
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , boost::str(boost::format(
                        "attempt to update a GVA with an invalid locality id, "
                        "gid(%1%), gva(%2%), locality(%3%)")
                        % id % g % locality));
                return response();
            }

            // Store the new endpoint and offset
            gaddr.prefix = g.prefix;
            gaddr.type   = g.type;
            gaddr.lva(g.lva());
            gaddr.offset = g.offset;
            loc = locality;

            l.unlock();

            LAGAS_(info) << (boost::format(
                "primary_namespace::bind_gid, gid(%1%), gva(%2%), "
                "locality(%3%), response(repeated_request)")
                % id % g % locality);

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_bind_gid, repeated_request);
        }

        // We're about to decrement the iterator it - first, we
        // check that it's safe to do this.
        else if (it != begin)
        {
            --it;

            // Check that a previous range doesn't cover the new id.
            if (HPX_UNLIKELY((it->first + it->second.first.count) > id))
            {
                // REVIEW: Is this the right error code to use?
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , "the new GID is contained in an existing range");
                return response();
            }
        }
    }

    else if (HPX_LIKELY(!gvas_.empty()))
    {
        --it;

        // Check that a previous range doesn't cover the new id.
        if ((it->first + it->second.first.count) > id)
        {
            // REVIEW: Is this the right error code to use?
            l.unlock();

            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::bind_gid"
              , "the new GID is contained in an existing range");
            return response();
        }
    }

    naming::gid_type upper_bound(id + (g.count - 1));

    if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
    {
        l.unlock();

        HPX_THROWS_IF(ec, internal_server_error
          , "primary_namespace::bind_gid"
          , "MSBs of lower and upper range bound do not match");
        return response();
    }

    if (HPX_UNLIKELY(components::component_invalid == g.type))
    {
        l.unlock();

        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "attempt to insert a GVA with an invalid type, "
                "gid(%1%), gva(%2%), locality(%3%)")
                % id % g % locality));
        return response();
    }

    // Insert a GID -> GVA entry into the GVA table.
    if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
            std::make_pair(id, std::make_pair(g, locality))))))
    {
        l.unlock();

        HPX_THROWS_IF(ec, lock_error
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "GVA table insertion failed due to a locking error or "
                "memory corruption, gid(%1%), gva(%2%)")
                % id % g % locality));
        return response();
    }

    l.unlock();

    LAGAS_(info) << (boost::format(
        "primary_namespace::bind_gid, gid(%1%), gva(%2%), locality(%3%)")
        % id % g % locality);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_bind_gid);
} // }}}

response primary_namespace::resolve_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve_gid implementation
    using hpx::util::get;

    // parameters
    naming::gid_type id = req.get_gid();

    resolved_type r;

    {
        std::unique_lock<mutex_type> l(mutex_);

        // wait for any migration to be completed
        wait_for_migration_locked(l, id, ec);

        // now, resolve the id
        r = resolve_gid_locked(l, id, ec);
    }

    if (get<0>(r) == naming::invalid_gid)
    {
        LAGAS_(info) << (boost::format(
            "primary_namespace::resolve_gid, gid(%1%), response(no_success)")
            % id);

        return response(primary_ns_resolve_gid
                      , naming::invalid_gid
                      , gva()
                      , naming::invalid_gid
                      , no_success);
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::resolve_gid, gid(%1%), base(%2%), "
        "gva(%3%), locality_id(%4%)")
        % id % get<0>(r) % get<1>(r) % get<2>(r));

    return response(primary_ns_resolve_gid, get<0>(r), get<1>(r), get<2>(r));
} // }}}

response primary_namespace::unbind_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ unbind_gid implementation
    // parameters
    boost::uint64_t count = req.get_count();
    naming::gid_type id = req.get_gid();
    naming::detail::strip_internal_bits_from_gid(id);

    std::unique_lock<mutex_type> l(mutex_);

    gva_table_type::iterator it = gvas_.find(id)
                           , end = gvas_.end();

    if (it != end)
    {
        if (HPX_UNLIKELY(it->second.first.count != count))
        {
            l.unlock();

            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::unbind_gid"
              , "block sizes must match");
            return response();
        }

        gva_table_data_type& data = it->second;
        response r(primary_ns_unbind_gid, data.first, data.second);

        gvas_.erase(it);

        l.unlock();
        LAGAS_(info) << (boost::format(
            "primary_namespace::unbind_gid, gid(%1%), count(%2%), gva(%3%), "
            "locality_id(%4%)")
            % id % count % data.first % data.second);

        if (&ec != &throws)
            ec = make_success_code();

        return r;
    }

    l.unlock();

    LAGAS_(info) << (boost::format(
        "primary_namespace::unbind_gid, gid(%1%), count(%2%), "
        "response(no_success)")
        % id % count);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_unbind_gid
                  , gva()
                  , naming::invalid_gid
                  , no_success);
} // }}}

response primary_namespace::increment_credit(
    request const& req
  , error_code& ec
    )
{ // increment_credit implementation
    // parameters
    std::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_lower_bound();
    naming::gid_type upper = req.get_upper_bound();

    naming::detail::strip_internal_bits_from_gid(lower);
    naming::detail::strip_internal_bits_from_gid(upper);

    if (lower == upper)
        ++upper;

    // Increment.
    if (credits > 0)
    {
        increment(lower, upper, credits, ec);
        if (ec) return response();
    }
    else
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::increment_credit"
          , boost::str(boost::format("invalid credit count of %1%") % credits));
        return response();
    }

    return response(primary_ns_increment_credit, credits);
}

response primary_namespace::decrement_credit(
    request const& req
  , error_code& ec
    )
{ // decrement_credit implementation
    // parameters
    std::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_lower_bound();
    naming::gid_type upper = req.get_upper_bound();

    naming::detail::strip_internal_bits_from_gid(lower);
    naming::detail::strip_internal_bits_from_gid(upper);

    if (lower == upper)
        ++upper;

    // Decrement.
    if (credits < 0)
    {
        std::list<free_entry> free_list;
        decrement_sweep(free_list, lower, upper, -credits, ec);
        if (ec) return response();

        free_components_sync(free_list, lower, upper, ec);
        if (ec) return response();
    }

    else
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::decrement_credit"
          , boost::str(boost::format("invalid credit count of %1%") % credits));
        return response();
    }

    return response(primary_ns_decrement_credit, credits);
}

response primary_namespace::allocate(
    request const& req
  , error_code& ec
    )
{ // {{{ allocate implementation
    boost::uint64_t const count = req.get_count();
    boost::uint64_t const real_count = (count) ? (count - 1) : (0);

    // Just return the prefix
    // REVIEW: Should this be an error?
    if (0 == count)
    {
        LAGAS_(info) << (boost::format(
            "primary_namespace::allocate, count(%1%), "
            "lower(%1%), upper(%3%), prefix(%4%), response(repeated_request)")
            % count % next_id_ % next_id_
            % naming::get_locality_id_from_gid(next_id_));

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_allocate, next_id_
          , naming::get_locality_id_from_gid(next_id_), success);
    }

    // Compute the new allocation.
    naming::gid_type lower(next_id_ + 1);
    naming::gid_type upper(lower + real_count);

    // Check for overflow.
    if (upper.get_msb() != lower.get_msb())
    {
        // Check for address space exhaustion (we currently use 86 bits of
        // the gid for the actual id)
        if (HPX_UNLIKELY(
            (lower.get_msb() & naming::gid_type::virtual_memory_mask) ==
                naming::gid_type::virtual_memory_mask)
           )
        {
            HPX_THROWS_IF(ec, internal_server_error
                , "locality_namespace::allocate"
                , "primary namespace has been exhausted");
            return response();
        }

        // Otherwise, correct
        lower = naming::gid_type(upper.get_msb(), 0);
        upper = lower + real_count;
    }

    // Store the new upper bound.
    next_id_ = upper;

    // Set the initial credit count.
    naming::detail::set_credit_for_gid(lower, boost::int64_t(HPX_GLOBALCREDIT_INITIAL));
    naming::detail::set_credit_for_gid(upper, boost::int64_t(HPX_GLOBALCREDIT_INITIAL));

    LAGAS_(info) << (boost::format(
        "primary_namespace::allocate, count(%1%), "
        "lower(%2%), upper(%3%), prefix(%4%), response(repeated_request)")
        % count % lower % upper % naming::get_locality_id_from_gid(next_id_));

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_allocate, lower
      , naming::get_locality_id_from_gid(next_id_), success);
} // }}}

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
    void primary_namespace::dump_refcnt_matches(
        refcnt_table_type::iterator lower_it
      , refcnt_table_type::iterator upper_it
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , std::unique_lock<mutex_type>& l
      , const char* func_name
        )
    { // dump_refcnt_matches implementation
        HPX_ASSERT(l.owns_lock());

        if (lower_it == refcnts_.end() && upper_it == refcnts_.end())
            // We got nothing, bail - our caller is probably about to throw.
            return;

        std::stringstream ss;
        ss << (boost::format(
              "%1%, dumping server-side refcnt table matches, lower(%2%), "
              "upper(%3%):")
              % func_name % lower % upper);

        for (/**/; lower_it != upper_it; ++lower_it)
        {
            // The [server] tag is in there to make it easier to filter
            // through the logs.
            ss << (boost::format(
                   "\n  [server] lower(%1%), credits(%2%)")
                   % lower_it->first
                   % lower_it->second);
        }

        LAGAS_(debug) << ss.str();
    } // dump_refcnt_matches implementation
#endif

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::increment(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , std::int64_t& credits
  , error_code& ec
    )
{ // {{{ increment implementation
    std::unique_lock<mutex_type> l(mutex_);

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
    if (LAGAS_ENABLED(debug))
    {
        typedef refcnt_table_type::iterator iterator;

        // Find the mappings that we're about to touch.
        refcnt_table_type::iterator lower_it = refcnts_.find(lower);
        refcnt_table_type::iterator upper_it;
        if (lower != upper)
        {
            upper_it = refcnts_.find(upper);
        }
        else
        {
            upper_it = lower_it;
            ++upper_it;
        }

        dump_refcnt_matches(lower_it, upper_it, lower, upper, l,
            "primary_namespace::increment");
    }
#endif

    // TODO: Whine loudly if a reference count overflows. We reserve ~0 for
    // internal bookkeeping in the decrement algorithm, so the maximum global
    // reference count is 2^64 - 2. The maximum number of credits a single GID
    // can hold, however, is limited to 2^32 - 1.

    // The third parameter we pass here is the default data to use in case the
    // key is not mapped. We don't insert GIDs into the refcnt table when we
    // allocate/bind them, so if a GID is not in the refcnt table, we know that
    // it's global reference count is the initial global reference count.

    for (naming::gid_type raw = lower; raw != upper; ++raw)
    {
        refcnt_table_type::iterator it = refcnts_.find(raw);
        if (it == refcnts_.end())
        {
            std::int64_t count =
                std::int64_t(HPX_GLOBALCREDIT_INITIAL) + credits;

            std::pair<refcnt_table_type::iterator, bool> p =
                refcnts_.insert(refcnt_table_type::value_type(raw, count));
            if (!p.second)
            {
                l.unlock();

                HPX_THROWS_IF(ec, invalid_data
                    , "primary_namespace::increment"
                    , boost::str(boost::format(
                        "couldn't create entry in reference count table, "
                        "raw(%1%), ref-count(%3%)")
                        % raw % count));
                return;
            }

            it = p.first;
        }
        else
        {
            it->second += credits;
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::increment, raw(%1%), refcnt(%2%)")
            % lower % it->second);
    }

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::resolve_free_list(
    std::unique_lock<mutex_type>& l
  , std::list<refcnt_table_type::iterator> const& free_list
  , std::list<free_entry>& free_entry_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , error_code& ec
    )
{
    HPX_ASSERT_OWNS_LOCK(l);

    using hpx::util::get;

    typedef refcnt_table_type::iterator iterator;

    for (iterator const& it : free_list)
    {
        typedef refcnt_table_type::key_type key_type;

        // The mapping's key space.
        key_type gid = it->first;

        // wait for any migration to be completed
        wait_for_migration_locked(l, gid, ec);

        // Resolve the query GID.
        resolved_type r = resolve_gid_locked(l, gid, ec);
        if (ec) return;

        naming::gid_type& raw = get<0>(r);
        if (raw == naming::invalid_gid)
        {
            l.unlock();

            HPX_THROWS_IF(ec, internal_server_error
                , "primary_namespace::resolve_free_list"
                , boost::str(boost::format(
                    "primary_namespace::resolve_free_list, failed to resolve "
                    "gid, gid(%1%)")
                    % gid));
            return;       // couldn't resolve this one
        }

        // Make sure the GVA is valid.
        gva& g = get<1>(r);

        // REVIEW: Should we do more to make sure the GVA is valid?
        if (HPX_UNLIKELY(components::component_invalid == g.type))
        {
            l.unlock();

            HPX_THROWS_IF(ec, internal_server_error
                , "primary_namespace::resolve_free_list"
                , boost::str(boost::format(
                    "encountered a GVA with an invalid type while "
                    "performing a decrement, gid(%1%), gva(%2%)")
                    % gid % g));
            return;
        }
        else if (HPX_UNLIKELY(0 == g.count))
        {
            l.unlock();

            HPX_THROWS_IF(ec, internal_server_error
                , "primary_namespace::resolve_free_list"
                , boost::str(boost::format(
                    "encountered a GVA with a count of zero while "
                    "performing a decrement, gid(%1%), gva(%2%)")
                    % gid % g));
            return;
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::resolve_free_list, resolved match, "
            "gid(%1%), gva(%2%)")
            % gid % g);

        // Fully resolve the range.
        gva const resolved = g.resolve(gid, raw);

        // Add the information needed to destroy these components to the
        // free list.
        free_entry_list.push_back(free_entry(resolved, gid, get<2>(r)));

        // remove this entry from the refcnt table
        refcnts_.erase(it);
    }
}

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::decrement_sweep(
    std::list<free_entry>& free_entry_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , std::int64_t credits
  , error_code& ec
    )
{ // {{{ decrement_sweep implementation
    LAGAS_(info) << (boost::format(
        "primary_namespace::decrement_sweep, lower(%1%), upper(%2%), "
        "credits(%3%)")
        % lower % upper % credits);

    free_entry_list.clear();

    {
        std::unique_lock<mutex_type> l(mutex_);

#if defined(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
        if (LAGAS_ENABLED(debug))
        {
            typedef refcnt_table_type::iterator iterator;

            // Find the mappings that we just added or modified.
            refcnt_table_type::iterator lower_it = refcnts_.find(lower);
            refcnt_table_type::iterator upper_it;
            if (lower != upper)
            {
                upper_it = refcnts_.find(upper);
            }
            else
            {
                upper_it = lower_it;
                ++upper_it;
            }

            dump_refcnt_matches(lower_it, upper_it, lower, upper, l,
                "primary_namespace::decrement_sweep");
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // Apply the decrement across the entire key space (e.g. [lower, upper]).

        // The third parameter we pass here is the default data to use in case
        // the key is not mapped. We don't insert GIDs into the refcnt table
        // when we allocate/bind them, so if a GID is not in the refcnt table,
        // we know that it's global reference count is the initial global
        // reference count.

        std::list<refcnt_table_type::iterator> free_list;
        for (naming::gid_type raw = lower; raw != upper; ++raw)
        {
            refcnt_table_type::iterator it = refcnts_.find(raw);
            if (it == refcnts_.end())
            {
                if (credits > std::int64_t(HPX_GLOBALCREDIT_INITIAL))
                {
                    l.unlock();

                    HPX_THROWS_IF(ec, invalid_data
                      , "primary_namespace::decrement_sweep"
                      , boost::str(boost::format(
                            "negative entry in reference count table, raw(%1%), "
                            "refcount(%2%)")
                            % raw
                            % (std::int64_t(HPX_GLOBALCREDIT_INITIAL) - credits)));
                    return;
                }

                std::int64_t count =
                    std::int64_t(HPX_GLOBALCREDIT_INITIAL) - credits;

                std::pair<refcnt_table_type::iterator, bool> p =
                    refcnts_.insert(refcnt_table_type::value_type(raw, count));
                if (!p.second)
                {
                    l.unlock();

                    HPX_THROWS_IF(ec, invalid_data
                      , "primary_namespace::decrement_sweep"
                      , boost::str(boost::format(
                            "couldn't create entry in reference count table, "
                            "raw(%1%), ref-count(%3%)")
                            % raw % count));
                    return;
                }

                it = p.first;
            }
            else
            {
                it->second -= credits;
            }

            // Sanity check.
            if (it->second < 0)
            {
                l.unlock();

                HPX_THROWS_IF(ec, invalid_data
                  , "primary_namespace::decrement_sweep"
                  , boost::str(boost::format(
                        "negative entry in reference count table, raw(%1%), "
                        "refcount(%2%)")
                        % raw % it->second));
                return;
            }

            // this objects needs to be deleted
            if (it->second == 0)
                free_list.push_back(it);
        }

        // Resolve the objects which have to be deleted.
        resolve_free_list(l, free_list, free_entry_list, lower, upper, ec);

    } // Unlock the mutex.

    if (&ec != &throws)
        ec = make_success_code();
}

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::free_components_sync(
    std::list<free_entry>& free_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , error_code& ec
    )
{ // {{{ free_components_sync implementation
    using hpx::util::get;

    std::vector<lcos::future<void> > futures;

    ///////////////////////////////////////////////////////////////////////////
    // Delete the objects on the free list.
    components::server::runtime_support::free_component_action act;

    for (free_entry const& e : free_list)
    {
        // Bail if we're in late shutdown and non-local.
        if (HPX_UNLIKELY(!threads::threadmanager_is(state_running)) &&
            e.locality_ != locality_)
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::free_components_sync, cancelling free "
                "operation because the threadmanager is down, lower(%1%), "
                "upper(%2%), base(%3%), gva(%4%), locality(%5%)")
                % lower
                % upper
                % e.gid_ % e.gva_ % e.locality_);
            continue;
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::free_components_sync, freeing component, "
            "lower(%1%), upper(%2%), base(%3%), gva(%4%), locality(%5%)")
            % lower
            % upper
            % e.gid_ % e.gva_ % e.locality_);

        // Free the object directly, if local (this avoids creating another
        // local promise via async which would create a snowball effect of
        // free_component calls.
        if (e.locality_ == locality_)
        {
            get_runtime_support_ptr()->free_component(e.gva_, e.gid_, 1);
        }
        else
        {
            naming::id_type const target_locality(e.locality_
              , naming::id_type::unmanaged);
            futures.push_back(hpx::async(act, target_locality, e.gva_, e.gid_, 1));
        }
    }

    if (!futures.empty())
        hpx::wait_all(futures);

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

primary_namespace::resolved_type primary_namespace::resolve_gid_locked(
    std::unique_lock<mutex_type>& l
  , naming::gid_type const& gid
  , error_code& ec
    )
{ // {{{ resolve_gid_locked implementation
    HPX_ASSERT_OWNS_LOCK(l);

    // parameters
    naming::gid_type id = gid;
    naming::detail::strip_internal_bits_from_gid(id);

    gva_table_type::const_iterator it = gvas_.lower_bound(id)
                                 , begin = gvas_.begin()
                                 , end = gvas_.end();

    if (it != end)
    {
        // Check for exact match
        if (it->first == id)
        {
            if (&ec != &throws)
                ec = make_success_code();

            gva_table_data_type const& data = it->second;
            return resolved_type(it->first, data.first, data.second);
        }

        // We need to decrement the iterator, first we check that it's safe
        // to do this.
        else if (it != begin)
        {
            --it;

            // Found the GID in a range
            gva_table_data_type const& data = it->second;
            if ((it->first + data.first.count) > id)
            {
                if (HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
                {
                    l.unlock();

                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::resolve_gid_locked"
                      , "MSBs of lower and upper range bound do not match");
                    return resolved_type(naming::invalid_gid, gva(),
                        naming::invalid_gid);
                }

                if (&ec != &throws)
                    ec = make_success_code();

                return resolved_type(it->first, data.first, data.second);
            }
        }
    }

    else if (HPX_LIKELY(!gvas_.empty()))
    {
        --it;

        // Found the GID in a range
        gva_table_data_type const& data = it->second;
        if ((it->first + data.first.count) > id)
        {
            if (HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
            {
                l.unlock();

                HPX_THROWS_IF(ec, internal_server_error
                  , "primary_namespace::resolve_gid_locked"
                  , "MSBs of lower and upper range bound do not match");
                return resolved_type(naming::invalid_gid, gva(),
                    naming::invalid_gid);
            }

            if (&ec != &throws)
                ec = make_success_code();

            return resolved_type(it->first, data.first, data.second);
        }
    }

    if (&ec != &throws)
        ec = make_success_code();

    return resolved_type(naming::invalid_gid, gva(), naming::invalid_gid);
} // }}}

response primary_namespace::statistics_counter(
    request const& req
  , error_code& ec
    )
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "primary_namespace::statistics_counter";

    std::string name(req.get_statistics_counter_name());

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p, ec);
    if (ec) return response();

    if (p.objectname_ != "agas")
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "primary_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    namespace_action_code code = invalid_request;
    detail::counter_target target = detail::counter_target_invalid;
    for (std::size_t i = 0;
          i != detail::num_primary_namespace_services;
          ++i)
    {
        if (p.countername_ == detail::primary_namespace_services[i].name_)
        {
            code = detail::primary_namespace_services[i].code_;
            target = detail::primary_namespace_services[i].target_;
            break;
        }
    }

    if (code == invalid_request || target == detail::counter_target_invalid)
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "primary_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS?)");
        return response();
    }

    util::function_nonser<boost::int64_t(bool)> get_data_func =
        counter_data_.get_counter_function(target, code, ec);
    if (ec) return response();

    performance_counters::counter_info info;
    performance_counters::get_counter_type(name, info, ec);
    if (ec) return response();

    performance_counters::complement_counter_info(info, ec);
    if (ec) return response();

    using performance_counters::detail::create_raw_counter;
    naming::gid_type gid = create_raw_counter(info, get_data_func, ec);
    if (ec) return response();

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_statistics_counter, gid);
}

}}}

