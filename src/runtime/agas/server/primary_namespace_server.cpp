////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <list>

#include <boost/foreach.hpp>
#include <boost/fusion/include/at_c.hpp>

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
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.route_.time_
                );
                counter_data_.increment_route_count();
                return route(req, ec);
            }
        case primary_ns_bind_gid:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.bind_gid_.time_
                );
                counter_data_.increment_bind_gid_count();
                return bind_gid(req, ec);
            }
        case primary_ns_resolve_gid:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.resolve_gid_.time_
                );
                counter_data_.increment_resolve_gid_count();
                return resolve_gid(req, ec);
            }
        case primary_ns_unbind_gid:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.unbind_gid_.time_
                );
                counter_data_.increment_unbind_gid_count();
                return unbind_gid(req, ec);
            }
        case primary_ns_increment_credit:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.increment_credit_.time_
                );
                counter_data_.increment_increment_credit_count();
                return increment_credit(req, ec);
            }
        case primary_ns_decrement_credit:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.decrement_credit_.time_
                );
                counter_data_.increment_decrement_credit_count();
                return decrement_credit(req, ec);
            }
        case primary_ns_allocate:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.allocate_.time_
                );
                counter_data_.increment_allocate_count();
                return allocate(req, ec);
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
    };
} // }}}

// register all performance counter types exposed by this component
void primary_namespace::register_counter_types(
    error_code& ec
    )
{
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    HPX_STD_FUNCTION<performance_counters::create_counter_func> creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::primary_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_primary_namespace_services;
          ++i)
    {
        std::string name(detail::primary_namespace_services[i].name_);
        std::string help;
        std::string::size_type p = name.find_last_of('/');
        if (p != std::string::npos) {
            if (detail::primary_namespace_services[i].target_ == detail::counter_target_count)
                help = boost::str(help_count % name.substr(p+1));
            else
                help = boost::str(help_time % name.substr(p+1));
        }
        else {
            HPX_ASSERT(detail::primary_namespace_services[i].code_ ==
                primary_ns_statistics_counter);
            name = primary_namespace_service_name + name;
            if (detail::primary_namespace_services[i].target_ == detail::counter_target_count)
                help = "returns the overall number of invocations of all primary AGAS services";
            else
                help = "returns the overall execution time of all primary AGAS services";
        }

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
    agas::register_name_sync(instance_name_, get_gid().get_gid(), ec);
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
  , naming::locality const& loc
  , parcelset::connection_type t
  , parcelset::parcel const& p
    )
{
    typedef hpx::actions::transfer_action<
        server::primary_namespace::service_action
    > action_type;

    boost::shared_ptr<action_type> act =
        boost::static_pointer_cast<action_type>(p.get_action());
    agas::request const& req = hpx::actions::get<0>(*act);

    // only routing is handled in a special way
    if (req.get_action_code() != primary_ns_route)
        return 0;

    parcelset::parcel routed_p = req.get_parcel();
    return routed_p.get_message_handler(ph, loc, t);
}

util::binary_filter* primary_namespace::get_serialization_filter(
    parcelset::parcel const& p
    )
{
    typedef hpx::actions::transfer_action<
        server::primary_namespace::service_action
    > action_type;

    boost::shared_ptr<action_type> act =
        boost::static_pointer_cast<action_type>(p.get_action());
    agas::request const& req = hpx::actions::get<0>(*act);

    // only routing is handled in a special way
    if (req.get_action_code() != primary_ns_route)
        return 0;

    parcelset::parcel routed_p = req.get_parcel();
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

    BOOST_FOREACH(request const& req, reqs)
    {
        r.push_back(service(req, ec));
        if (ec)
            break;      // on error: for now stop iterating
    }

    return r;
}

response primary_namespace::bind_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ bind_gid implementation
    using boost::fusion::at_c;

    // parameters
    gva g = req.get_gva();
    naming::gid_type id = req.get_gid();
    boost::uint32_t locality_id = req.get_locality_id();

    naming::detail::strip_internal_bits_from_gid(id);

    mutex_type::scoped_lock l(mutex_);

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
            boost::uint32_t& loc_id = it->second.second;

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
                        "gid(%1%), gva(%2%), locality_id(%3%)")
                        % id % g % locality_id));
                return response();
            }

            if (HPX_UNLIKELY(locality_id == naming::invalid_locality_id))
            {
                l.unlock();

                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , boost::str(boost::format(
                        "attempt to update a GVA with an invalid locality id, "
                        "gid(%1%), gva(%2%), locality_id(%3%)")
                        % id % g % locality_id));
                return response();
            }

            // Store the new endpoint and offset
            gaddr.endpoint = g.endpoint;
            gaddr.type = g.type;
            gaddr.lva(g.lva());
            gaddr.offset = g.offset;
            loc_id = locality_id;

            LAGAS_(info) << (boost::format(
                "primary_namespace::bind_gid, gid(%1%), gva(%2%), "
                "locality_id(%3%), response(repeated_request)")
                % id % g % locality_id);

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
                "gid(%1%), gva(%2%), locality_id(%3%)")
                % id % g % locality_id));
        return response();
    }

    // Insert a GID -> GVA entry into the GVA table.
    if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
            std::make_pair(id, std::make_pair(g, locality_id))))))
    {
        l.unlock();

        HPX_THROWS_IF(ec, lock_error
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "GVA table insertion failed due to a locking error or "
                "memory corruption, gid(%1%), gva(%2%)")
                % id % g % locality_id));
        return response();
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::bind_gid, gid(%1%), gva(%2%), locality_id(%3%)")
        % id % g % locality_id);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_bind_gid);
} // }}}

response primary_namespace::resolve_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve_gid implementation
    using boost::fusion::at_c;

    // parameters
    naming::gid_type id = req.get_gid();

    boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t> r;

    {
        mutex_type::scoped_lock l(mutex_);
        r = resolve_gid_locked(id, ec);
    }

    if (at_c<0>(r) == naming::invalid_gid)
    {
        LAGAS_(info) << (boost::format(
            "primary_namespace::resolve_gid, gid(%1%), response(no_success)")
            % id);

        return response(primary_ns_resolve_gid
                      , naming::invalid_gid
                      , gva()
                      , naming::invalid_locality_id
                      , no_success);
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::resolve_gid, gid(%1%), base(%2%), gva(%3%), locality_id(%4%)")
        % id % at_c<0>(r) % at_c<1>(r) % at_c<2>(r));

    return response(primary_ns_resolve_gid
                    , at_c<0>(r)
                    , at_c<1>(r)
                    , at_c<2>(r));
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

    mutex_type::scoped_lock l(mutex_);

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
        LAGAS_(info) << (boost::format(
            "primary_namespace::unbind_gid, gid(%1%), count(%2%), gva(%3%), "
            "locality_id(%4%)")
            % id % count % data.first % data.second);

        gvas_.erase(it);

        if (&ec != &throws)
            ec = make_success_code();

        return r;
    }

    else
    {
        LAGAS_(info) << (boost::format(
            "primary_namespace::unbind_gid, gid(%1%), count(%2%), "
            "response(no_success)")
            % id % count);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_unbind_gid
                      , gva()
                      , naming::invalid_locality_id
                      , no_success);
    }
} // }}}

response primary_namespace::increment_credit(
    request const& req
  , error_code& ec
    )
{ // change_credit_non_blocking implementation
    // parameters
    boost::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_gid();

    naming::detail::strip_internal_bits_from_gid(lower);

    // Increment.
    if (credits > 0)
    {
        increment(lower, lower, credits, ec);
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
    boost::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_lower_bound();
    naming::gid_type upper = req.get_upper_bound();

    naming::detail::strip_internal_bits_from_gid(lower);
    naming::detail::strip_internal_bits_from_gid(upper);

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

        return response(primary_ns_allocate, next_id_, next_id_
          , naming::get_locality_id_from_gid(next_id_), success);
    }

    // Compute the new allocation.
    naming::gid_type lower(next_id_ + 1);
    naming::gid_type upper(lower + real_count);

    // Check for overflow.
    if (upper.get_msb() != lower.get_msb())
    {
        // Check for address space exhaustion (we currently use 80 bits of
        // the gid for the actual id)
        if (HPX_UNLIKELY((lower.get_msb() & ~0xFF) == 0xFF))
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
    naming::detail::set_credit_for_gid(lower, HPX_GLOBALCREDIT_INITIAL);
    naming::detail::set_credit_for_gid(upper, HPX_GLOBALCREDIT_INITIAL);

    LAGAS_(info) << (boost::format(
        "primary_namespace::allocate, count(%1%), "
        "lower(%2%), upper(%3%), prefix(%4%), response(repeated_request)")
        % count % lower % upper % naming::get_locality_id_from_gid(next_id_));

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_allocate, lower, upper
      , naming::get_locality_id_from_gid(next_id_), success);
} // }}}

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
    void primary_namespace::dump_refcnt_matches(
        refcnt_match match
      , naming::gid_type const& lower
      , naming::gid_type const& upper
      , mutex_type::scoped_lock& l
      , const char* func_name
        )
    { // dump_refcnt_matches implementation
        HPX_ASSERT(l.owns_lock());

        if (match.first == refcnts_.end() && match.second == refcnts_.end())
            // We got nothing, bail - our caller is probably about to throw.
            return;

        std::stringstream ss;
        ss << (boost::format(
              "%1%, dumping server-side refcnt table matches, lower(%2%), "
              "upper(%3%):")
              % func_name % lower % upper);

        for (/**/; match.first != match.second; ++match.first)
        {
            naming::gid_type const lower = boost::icl::lower(match.first->key_);
            naming::gid_type const upper = boost::icl::upper(match.first->key_);

            naming::gid_type const length_gid = (upper - lower);
            HPX_ASSERT(length_gid.get_msb() == 0);
            boost::uint64_t const length = length_gid.get_lsb() + 1;

            // The [server] tag is in there to make it easier to filter
            // through the logs.
            ss << (boost::format(
                   "\n  [server] lower(%1%), upper(%2%), length(%3%), "
                   "credits(%4%)")
                   % lower
                   % upper
                   % length
                   % match.first->data_);
        }

        LAGAS_(debug) << ss.str();
    } // dump_refcnt_matches implementation
#endif

///////////////////////////////////////////////////////////////////////////////
boost::int64_t default_global_credit(
    util::merging_map<naming::gid_type, boost::int64_t>::key_type const&)
{
    return boost::int64_t(HPX_GLOBALCREDIT_INITIAL);
}

boost::int64_t init_credit(
    util::merging_map<naming::gid_type, boost::int64_t>::key_type const&
  , boost::int64_t& added_credits
    )
{
    added_credits += HPX_GLOBALCREDIT_INITIAL;
    return boost::int64_t(HPX_GLOBALCREDIT_INITIAL);
}

void primary_namespace::increment(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t& credits
  , error_code& ec
    )
{ // {{{ increment implementation
    mutex_type::scoped_lock l(mutex_);

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
    if (LAGAS_ENABLED(debug))
    {
        typedef refcnt_table_type::iterator iterator;

        // Find the mappings that we're about to touch.
        std::pair<iterator, iterator> match = refcnts_.find(lower, upper);

        dump_refcnt_matches(match, lower, upper, l,
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
    using util::placeholders::_1;
    refcnts_.apply(
            lower, upper
          , util::incrementer<boost::int64_t>(credits)
          , util::bind(&init_credit, _1, std::ref(credits)));

    LAGAS_(info) << (boost::format(
        "primary_namespace::increment, lower(%1%), upper(%2%), credits(%3%)")
        % lower % upper % credits);

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::resolve_free_list(
    mutex_type::scoped_lock& l
  , std::list<refcnt_table_type::iterator> const& free_list
  , std::list<free_entry>& free_entry_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , error_code& ec
    )
{
    using boost::fusion::at_c;

    typedef refcnt_table_type::iterator iterator;

    BOOST_FOREACH(iterator const& it, free_list)
    {
        // Both the client- and server-side merging algorithms are unaware
        // of component type, so a merged mapping in the reference count
        // tables might contain multiple component types, or the same
        // component type on different localities (currently, the latter is
        // improbable due to the nature of address allocation and the fact
        // that moving objects is still unsupported).

        typedef refcnt_table_type::key_type key_type;

        // The mapping's key space.
        key_type super = it->key_;

        // Make sure we stay within the bounds of the decrement request.
        if (boost::icl::lower(super) < lower &&
            boost::icl::upper(super) > upper)
        {
            super = key_type(lower, upper);
        }

        else if (boost::icl::lower(super) < lower)
        {
            super = key_type(lower, boost::icl::upper(super));
        }

        else if (boost::icl::upper(super) > upper)
        {
            super = key_type(boost::icl::lower(super), upper);
        }

        // Keep resolving GIDs to GVAs until we've covered all of this
        // mapping's key space.
        while (!boost::icl::is_empty(super))
        {
            naming::gid_type query = boost::icl::lower(super);

            // Resolve the query GID.
            boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                r = resolve_gid_locked(query, ec);

            if (ec)
                return;

            if (at_c<0>(r) == naming::invalid_gid)
            {
                LAGAS_(info) << (boost::format(
                    "primary_namespace::resolve_free_list, failed to resolve gid, "
                    "lower(%1%), upper(%2%), super-object(%3%)")
                    % lower % upper % super);
                continue;       // couldn't resolve this one
            }

            // Make sure the GVA is valid.
            // REVIEW: Should we do more to make sure the GVA is valid?
            if (HPX_UNLIKELY(components::component_invalid
                                == at_c<1>(r).type))
            {
                l.unlock();

                HPX_THROWS_IF(ec, internal_server_error
                    , "primary_namespace::resolve_free_list"
                    , boost::str(boost::format(
                        "encountered a GVA with an invalid type while "
                        "performing a decrement, gid:%1%, gva:%2%")
                        % query % at_c<1>(r)));
                return;
            }

            else if (HPX_UNLIKELY(0 == at_c<1>(r).count))
            {
                l.unlock();

                HPX_THROWS_IF(ec, internal_server_error
                    , "primary_namespace::resolve_free_list"
                    , boost::str(boost::format(
                        "encountered a GVA with a count of zero while "
                        "performing a decrement, gid:%1%, gva:%2%")
                        % query % at_c<1>(r)));
                return;
            }

            // Determine how much of the mapping's key space this GVA covers.
            // Note that at_c<1>(r).count must be greater than 0 if we've
            // reached this point in the code.
            naming::gid_type sub_upper(at_c<0>(r) + (at_c<1>(r).count - 1));

            // If this GVA ends after the key space, we just set the upper
            // limit to the end of the key space.
            if (sub_upper > boost::icl::upper(super))
                sub_upper = boost::icl::upper(super);

            HPX_ASSERT(query <= sub_upper);

            // We don't use the base GID returned by resolve_gid_locked, but
            // instead we use the GID that we queried the GVA table with.
            // This ensures that GVAs which cover a range that begins before
            // our key space are handled properly.
            key_type const sub(query, sub_upper);

            LAGAS_(info) << (boost::format(
                "primary_namespace::resolve_free_list, resolved match, "
                "lower(%1%), upper(%2%), super-object(%3%), "
                "sub-object(%4%)")
                % lower % upper % super % sub);

            // Subtract the GIDs that are bound to this GVA from the
            // key space.
            super = boost::icl::left_subtract(super, sub);

            // Compute the length of sub.
            naming::gid_type const length = boost::icl::length(sub);

            // Fully resolve the range.
            gva const g = at_c<1>(r).resolve(query, at_c<0>(r));

            // Add the information needed to destroy these components to the
            // free list.
            free_entry_list.push_back(free_entry(g, query, length, at_c<2>(r)));
        }

        // If this is just a partial match, we need to split it up with a
        // remapping so that we can erase it.
        if (super != it->key_)
            // We use ~0 to prevent merges.
            refcnts_.erase(refcnts_.bind(super, boost::int64_t(~0)));
        else
            refcnts_.erase(it);
    }
}

///////////////////////////////////////////////////////////////////////////////
void primary_namespace::decrement_sweep(
    std::list<free_entry>& free_entry_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credits
  , error_code& ec
    )
{ // {{{ decrement_sweep implementation
    LAGAS_(info) << (boost::format(
        "primary_namespace::decrement_sweep, lower(%1%), upper(%2%), "
        "credits(%3%)")
        % lower % upper % credits);

    free_entry_list.clear();

    {
        mutex_type::scoped_lock l(mutex_);

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
        if (LAGAS_ENABLED(debug))
        {
            typedef refcnt_table_type::iterator iterator;

            // Find the mappings that we just added or modified.
            std::pair<iterator, iterator> match = refcnts_.find(lower, upper);

            dump_refcnt_matches(match, lower, upper, l,
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
        using util::placeholders::_1;
        refcnts_.apply(
            lower, upper
          , util::decrementer<boost::int64_t>(credits)
          , default_global_credit);

        ///////////////////////////////////////////////////////////////////////
        // Search for dead objects.

        typedef refcnt_table_type::iterator iterator;

        // Find the mappings that we just added or modified.
        std::pair<iterator, iterator> match = refcnts_.find(lower, upper);

        // This search should always succeed.
        if (match.first == refcnts_.end() && match.second == refcnts_.end())
        {
            l.unlock();

            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::decrement_sweep"
              , boost::str(boost::format(
                    "reference count table insertion failed due to a locking "
                    "error or memory corruption, lower(%1%), upper(%2%)")
                    % lower % upper));
            return;
        }

        // Ranges containing dead objects.
        std::list<iterator> free_list;

        for (/**/; match.first != match.second; ++match.first)
        {
            // Sanity check.
            if (match.first->data_ < 0)
            {
                l.unlock();

                HPX_THROWS_IF(ec, invalid_data
                  , "primary_namespace::decrement_sweep"
                  , boost::str(boost::format(
                        "negative entry in reference count table, lower(%1%), "
                        "upper(%2%), count(%3%)")
                        % boost::icl::lower(match.first->key_)
                        % boost::icl::upper(match.first->key_)
                        % match.first->data_));
                return;
            }

            // Any object with a reference count of 0 is dead.
            if (match.first->data_ == 0)
                free_list.push_back(match.first);
        }

        ///////////////////////////////////////////////////////////////////////
        // Resolve the dead objects.
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
{ // {{{ kill_sync implementation
    using boost::fusion::at_c;

    std::vector<lcos::future<void> > futures;

    ///////////////////////////////////////////////////////////////////////////
    // Delete the objects on the free list.
    components::server::runtime_support::free_component_action act;

    BOOST_FOREACH(free_entry const& e, free_list)
    {
        // Bail if we're in late shutdown and non-local.
        if (HPX_UNLIKELY(!threads::threadmanager_is(running)) &&
            at_c<3>(e) != locality_id_)
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::free_components_sync, cancelling free "
                "operation because the threadmanager is down, lower(%1%), "
                "upper(%2%), base(%3%), gva(%4%), count(%5%), locality_id(%6%)")
                % lower
                % upper
                % at_c<1>(e) % at_c<0>(e) % at_c<2>(e) % at_c<3>(e));
            continue;
        }

        naming::id_type const target_locality(
            naming::get_gid_from_locality_id(at_c<3>(e))
          , naming::id_type::unmanaged);

        LAGAS_(info) << (boost::format(
            "primary_namespace::free_components_sync, freeing component%1%, "
            "lower(%2%), upper(%3%), base(%4%), gva(%5%), count(%6%), locality_id(%7%)")
            % ((at_c<2>(e) == naming::gid_type(0, 1)) ? "" : "s")
            % lower
            % upper
            % at_c<1>(e) % at_c<0>(e) % at_c<2>(e) % at_c<3>(e));

        // Free the object directly, if local (this avoids creating another
        // local promise via async which would create a snowball effect of
        // free_component calls.
        if (at_c<3>(e) == hpx::get_locality_id())
        {
            get_runtime_support_ptr()->free_component(at_c<0>(e),
                at_c<1>(e), at_c<2>(e).get_lsb());
        }
        else
        {
            futures.push_back(hpx::async(act, target_locality, at_c<0>(e),
                at_c<1>(e), at_c<2>(e).get_lsb()));
        }
    }

    hpx::wait_all(futures);

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
primary_namespace::resolve_gid_locked(
    naming::gid_type const& gid
  , error_code& ec
    )
{ // {{{ resolve_gid implementation
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
            return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                (it->first, data.first, data.second);
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
                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::resolve_gid_locked"
                      , "MSBs of lower and upper range bound do not match");
                    return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                        (naming::invalid_gid, gva(), naming::invalid_locality_id);
                }

                if (&ec != &throws)
                    ec = make_success_code();

                return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                    (it->first, data.first, data.second);
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
                HPX_THROWS_IF(ec, internal_server_error
                  , "primary_namespace::resolve_gid_locked"
                  , "MSBs of lower and upper range bound do not match");
                return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                    (naming::invalid_gid, gva(), naming::invalid_locality_id);
            }

            if (&ec != &throws)
                ec = make_success_code();

            return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
                (it->first, data.first, data.second);
        }
    }

    if (&ec != &throws)
        ec = make_success_code();

    return boost::fusion::vector3<naming::gid_type, gva, boost::uint32_t>
        (naming::invalid_gid, gva(), naming::invalid_locality_id);
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

    typedef primary_namespace::counter_data cd;

    HPX_STD_FUNCTION<boost::int64_t(bool)> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case primary_ns_route:
            get_data_func = boost::bind(&cd::get_route_count, &counter_data_, ::_1);
            break;
        case primary_ns_bind_gid:
            get_data_func = boost::bind(&cd::get_bind_gid_count, &counter_data_, ::_1);
            break;
        case primary_ns_resolve_gid:
            get_data_func = boost::bind(&cd::get_resolve_gid_count, &counter_data_, ::_1);
            break;
        case primary_ns_unbind_gid:
            get_data_func = boost::bind(&cd::get_unbind_gid_count, &counter_data_, ::_1);
            break;
        case primary_ns_increment_credit:
            get_data_func = boost::bind(&cd::get_increment_credit_count, &counter_data_, ::_1);
            break;
        case primary_ns_decrement_credit:
            get_data_func = boost::bind(&cd::get_decrement_credit_count, &counter_data_, ::_1);
            break;
        case primary_ns_allocate:
            get_data_func = boost::bind(&cd::get_allocate_count, &counter_data_, ::_1);
            break;
        case primary_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_count, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }
    else {
        HPX_ASSERT(detail::counter_target_time == target);
        switch (code) {
        case primary_ns_route:
            get_data_func = boost::bind(&cd::get_route_time, &counter_data_, ::_1);
            break;
        case primary_ns_bind_gid:
            get_data_func = boost::bind(&cd::get_bind_gid_time, &counter_data_, ::_1);
            break;
        case primary_ns_resolve_gid:
            get_data_func = boost::bind(&cd::get_resolve_gid_time, &counter_data_, ::_1);
            break;
        case primary_ns_unbind_gid:
            get_data_func = boost::bind(&cd::get_unbind_gid_time, &counter_data_, ::_1);
            break;
        case primary_ns_increment_credit:
            get_data_func = boost::bind(&cd::get_increment_credit_time, &counter_data_, ::_1);
            break;
        case primary_ns_decrement_credit:
            get_data_func = boost::bind(&cd::get_decrement_credit_time, &counter_data_, ::_1);
            break;
        case primary_ns_allocate:
            get_data_func = boost::bind(&cd::get_allocate_time, &counter_data_, ::_1);
            break;
        case primary_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_time, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }

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

// access current counter values
boost::int64_t primary_namespace::counter_data::get_route_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(route_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_bind_gid_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(bind_gid_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_resolve_gid_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(resolve_gid_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_unbind_gid_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(unbind_gid_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_increment_credit_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(increment_credit_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_decrement_credit_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(decrement_credit_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_allocate_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(allocate_.count_, reset);
}

boost::int64_t primary_namespace::counter_data::get_overall_count(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(route_.count_, reset) +
        util::get_and_reset_value(bind_gid_.count_, reset) +
        util::get_and_reset_value(resolve_gid_.count_, reset) +
        util::get_and_reset_value(unbind_gid_.count_, reset) +
        util::get_and_reset_value(increment_credit_.count_, reset) +
        util::get_and_reset_value(decrement_credit_.count_, reset) +
        util::get_and_reset_value(allocate_.count_, reset);
}

// access execution time counters
boost::int64_t primary_namespace::counter_data::get_route_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(route_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_bind_gid_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(bind_gid_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_resolve_gid_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(resolve_gid_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_unbind_gid_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(unbind_gid_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_increment_credit_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(increment_credit_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_decrement_credit_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(decrement_credit_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_allocate_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(allocate_.time_, reset);
}

boost::int64_t primary_namespace::counter_data::get_overall_time(bool reset)
{
    mutex_type::scoped_lock l(mtx_);
    return util::get_and_reset_value(route_.time_, reset) +
        util::get_and_reset_value(bind_gid_.time_, reset) +
        util::get_and_reset_value(resolve_gid_.time_, reset) +
        util::get_and_reset_value(unbind_gid_.time_, reset) +
        util::get_and_reset_value(increment_credit_.time_, reset) +
        util::get_and_reset_value(decrement_credit_.time_, reset) +
        util::get_and_reset_value(allocate_.time_, reset);
}

// increment counter values
void primary_namespace::counter_data::increment_route_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++route_.count_;
}

void primary_namespace::counter_data::increment_bind_gid_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++bind_gid_.count_;
}

void primary_namespace::counter_data::increment_resolve_gid_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++resolve_gid_.count_;
}

void primary_namespace::counter_data::increment_unbind_gid_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++unbind_gid_.count_;
}

void primary_namespace::counter_data::increment_increment_credit_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++increment_credit_.count_;
}

void primary_namespace::counter_data::increment_decrement_credit_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++decrement_credit_.count_;
}

void primary_namespace::counter_data::increment_allocate_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++allocate_.count_;
}

}}}

