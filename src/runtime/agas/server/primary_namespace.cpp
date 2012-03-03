////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

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
    return naming::id_type( bootstrap_primary_namespace_gid()
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
        case primary_ns_allocate:
            return allocate(req, ec);
        case primary_ns_bind_gid:
            return bind_gid(req, ec);
        case primary_ns_resolve_gid:
            return resolve_gid(req, ec);
        case primary_ns_free:
            return free(req, ec);
        case primary_ns_unbind_gid:
            return unbind_gid(req, ec);
        case primary_ns_change_credit_non_blocking:
            return change_credit_non_blocking(req, ec);
        case primary_ns_change_credit_sync:
            return change_credit_sync(req, ec);
        case primary_ns_localities:
            return localities(req, ec);

        case component_ns_bind_prefix:
        case component_ns_bind_name:
        case component_ns_resolve_id:
        case component_ns_unbind:
        case component_ns_iterate_types:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "component_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case symbol_ns_bind:
        case symbol_ns_resolve:
        case symbol_ns_unbind:
        case symbol_ns_iterate_names:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "symbol_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case component_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "component_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    };
} // }}}

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
        error_code ign;
        r.push_back(service(req, ign));
    }

    return r;
}

response primary_namespace::allocate(
    request const& req
  , error_code& ec
    )
{ // {{{ allocate implementation
    using boost::fusion::at_c;

    // parameters
    naming::locality ep = req.get_locality();
    boost::uint64_t const count = req.get_count();
    boost::uint64_t const real_count = (count) ? (count - 1) : (0);

    mutex_type::scoped_lock l(mutex_);

    partition_table_type::iterator it = partitions_.find(ep)
                                 , end = partitions_.end();

    // If the endpoint is in the table, then this is a resize.
    if (it != end)
    {
        // Just return the prefix
        // REVIEW: Should this be an error?
        if (0 == count)
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::allocate, ep(%1%), count(%2%), "
                "lower(%3%), upper(%4%), prefix(%5%), "
                "response(repeated_request)")
                % ep
                % count
                % at_c<1>(it->second)
                % at_c<1>(it->second)
                % at_c<0>(it->second));

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_allocate
                          , at_c<1>(it->second)
                          , at_c<1>(it->second)
                          , at_c<0>(it->second)
                          , repeated_request);
        }

        // Compute the new allocation.
        naming::gid_type lower(at_c<1>(it->second) + 1),
                         upper(lower + real_count);

        // Check for overflow.
        if (upper.get_msb() != lower.get_msb())
        {
            // Check for address space exhaustion
            if (HPX_UNLIKELY((lower.get_msb() & ~0xFFFFFFFF) == 0xFFFFFFF))
            {
                HPX_THROWS_IF(ec, internal_server_error
                  , "primary_namespace::allocate"
                  , "primary namespace has been exhausted");
                return response();
            }

            // Otherwise, correct
            lower = naming::gid_type(upper.get_msb(), 0);
            upper = lower + real_count;
        }

        // Store the new upper bound.
        at_c<1>(it->second) = upper;

        // Set the initial credit count.
        naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
        naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT);

        LAGAS_(info) << (boost::format(
            "primary_namespace::allocate, ep(%1%), count(%2%), "
            "lower(%3%), upper(%4%), prefix(%5%), response(repeated_request)")
            % ep % count % lower % upper % at_c<0>(it->second));

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_allocate
                      , lower
                      , upper
                      , at_c<0>(it->second)
                      , repeated_request);
    }

    // If the endpoint isn't in the table, then we're registering it.
    else
    {
        // Check for address space exhaustion.
        if (HPX_UNLIKELY(0xFFFFFFFE < partitions_.size()))
        {
            HPX_THROWS_IF(ec, internal_server_error
              , "primary_namespace::allocate"
              , "primary namespace has been exhausted");
            return response();
        }

        // Compute the locality's prefix.
        boost::uint32_t prefix = prefix_counter_++;
        naming::gid_type id(naming::get_gid_from_locality_id(prefix));

        // Check if this prefix has already been assigned.
        while (gvas_.count(id))
        {
            prefix = prefix_counter_++;
            id = naming::get_gid_from_locality_id(prefix);
        }

        // Start assigning ids with the second block of 64bit numbers only.
        // The first block is reserved for components with LVA-encoded GIDs.
        naming::gid_type lower_id(id.get_msb() + 1, 0);

        // We need to create an entry in the partition table for this
        // locality.
        partition_table_type::iterator pit;

        if (HPX_UNLIKELY(!util::insert_checked(partitions_.insert(
                std::make_pair(ep,
                    partition_type(prefix, lower_id))), pit)))
        {
            // If this branch is taken, then the partition table was updated
            // at some point after we first checked it, which would indicate
            // memory corruption or a locking failure.
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::allocate"
              , boost::str(boost::format(
                    "partition table insertion failed due to a locking "
                    "error or memory corruption, endpoint(%1%), "
                    "prefix(%2%), lower_id(%3%)")
                    % ep % prefix % lower_id));
            return response();
        }

        const gva g(ep, components::component_runtime_support, count);

        // Now that we've inserted the locality into the partition table
        // successfully, we need to put the locality's GID into the GVA
        // table so that parcels can be sent to the memory of a locality.
        if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
                std::make_pair(id, g)))))
        {
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::allocate"
              , boost::str(boost::format(
                    "GVA table insertion failed due to a locking error "
                    "or memory corruption, gid(%1%), gva(%2%)")
                    % id % g));
            return response();
        }

        // Generate the requested GID range
        naming::gid_type lower = lower_id + 1;
        naming::gid_type upper = lower + real_count;

        at_c<1>((*pit).second) = upper;

        // Set the initial credit count.
        naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
        naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT);

        LAGAS_(info) << (boost::format(
            "primary_namespace::allocate, ep(%1%), count(%2%), "
            "lower(%3%), upper(%4%), prefix(%5%)")
            % ep % count % lower % upper % prefix);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_allocate
                      , lower
                      , upper
                      , prefix);
    }
} // }}}

response primary_namespace::bind_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ bind_gid implementation
    using boost::fusion::at_c;

    // parameters
    gva g = req.get_gva();
    naming::gid_type id = req.get_gid();
    naming::strip_credit_from_gid(id);

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
            // Check for count mismatch (we can't change block sizes of
            // existing bindings).
            if (HPX_UNLIKELY(it->second.count != g.count))
            {
                // REVIEW: Is this the right error code to use?
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , "cannot change block size of existing binding");
                return response();
            }

            if (HPX_UNLIKELY(components::component_invalid == g.type))
            {
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid"
                  , boost::str(boost::format(
                        "attempt to update a GVA with an invalid type, "
                        "gid(%1%), gva(%2%)")
                        % id % g));
                return response();
            }

            // Store the new endpoint and offset
            it->second.endpoint = g.endpoint;
            it->second.type = g.type;
            it->second.lva(g.lva());
            it->second.offset = g.offset;

            LAGAS_(info) << (boost::format(
                "primary_namespace::bind_gid, gid(%1%), gva(%2%), "
                "response(repeated_request)")
                % id % g);

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
            if (HPX_UNLIKELY((it->first + it->second.count) > id))
            {
                // REVIEW: Is this the right error code to use?
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
        if ((it->first + it->second.count) > id)
        {
            // REVIEW: Is this the right error code to use?
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::bind_gid"
              , "the new GID is contained in an existing range");
            return response();
        }
    }

    naming::gid_type upper_bound(id + (g.count - 1));

    if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
    {
        HPX_THROWS_IF(ec, internal_server_error
          , "primary_namespace::bind_gid"
          , "MSBs of lower and upper range bound do not match");
        return response();
    }

    if (HPX_UNLIKELY(components::component_invalid == g.type))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "attempt to insert a GVA with an invalid type, "
                "gid(%1%), gva(%2%)")
                % id % g));
        return response();
    }

    // Insert a GID -> GVA entry into the GVA table.
    if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
            std::make_pair(id, g)))))
    {
        HPX_THROWS_IF(ec, lock_error
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "GVA table insertion failed due to a locking error or "
                "memory corruption, gid(%1%), gva(%2%)")
                % id % g));
        return response();
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::bind_gid, gid(%1%), gva(%2%)")
        % id % g);

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

    boost::fusion::vector2<naming::gid_type, gva> r;

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
                      , no_success);
    }

    else
    {
        LAGAS_(info) << (boost::format(
            "primary_namespace::resolve_gid, gid(%1%), base(%2%), gva(%3%)")
            % id % at_c<0>(r) % at_c<1>(r));

        return response(primary_ns_resolve_gid
                      , at_c<0>(r)
                      , at_c<1>(r));
    }
} // }}}

response primary_namespace::resolve_locality(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve_locality implementation
    using boost::fusion::at_c;

    // parameters
    naming::locality ep = req.get_locality();

    mutex_type::scoped_lock l(mutex_);

    partition_table_type::iterator pit = partitions_.find(ep)
                                 , pend = partitions_.end();

    if (pit != pend)
    {
        boost::uint32_t const prefix = at_c<0>(pit->second);

        LAGAS_(info) << (boost::format(
            "primary_namespace::resolve_locality, ep(%1%), prefix(%2%)")
            % ep % prefix);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_resolve_locality, prefix);
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::resolve_locality, ep(%1%), response(no_success)")
        % ep);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_resolve_locality, 0, no_success);
} // }}}

response primary_namespace::free(
    request const& req
  , error_code& ec
    )
{ // {{{ free implementation
    using boost::fusion::at_c;

    // parameters
    naming::locality ep = req.get_locality();

    mutex_type::scoped_lock l(mutex_);

    partition_table_type::iterator pit = partitions_.find(ep)
                                 , pend = partitions_.end();

    if (pit != pend)
    {
        gva_table_type::iterator git = gvas_.find
            (naming::get_gid_from_locality_id(at_c<0>(pit->second)));
        gva_table_type::iterator gend = gvas_.end();

        if (HPX_UNLIKELY(git == gend))
        {
            HPX_THROWS_IF(ec, internal_server_error
              , "primary_namespace::free"
              , boost::str(boost::format(
                    "partition table entry has no corresponding GVA table "
                    "entry, endpoint(%1%)")
                    % ep));
            return response();
        }

        // Wipe the locality from the tables.
        partitions_.erase(pit);
        gvas_.erase(git);

        LAGAS_(info) << (boost::format(
            "primary_namespace::free, ep(%1%)")
            % ep);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_free);
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::free, ep(%1%), "
        "response(no_success)")
        % ep);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_free
                       , no_success);
} // }}}

response primary_namespace::unbind_gid(
    request const& req
  , error_code& ec
    )
{ // {{{ unbind_gid implementation
    // parameters
    boost::uint64_t count = req.get_count();
    naming::gid_type id = req.get_gid();
    naming::strip_credit_from_gid(id);

    mutex_type::scoped_lock l(mutex_);

    gva_table_type::iterator it = gvas_.find(id)
                           , end = gvas_.end();

    if (it != end)
    {
        if (HPX_UNLIKELY(it->second.count != count))
        {
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::unbind_gid"
              , "block sizes must match");
            return response();
        }

        response r(primary_ns_unbind_gid, it->second);
        LAGAS_(info) << (boost::format(
            "primary_namespace::unbind_gid, gid(%1%), count(%2%), gva(%3%)")
            % id % count % it->second);

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
                      , no_success);
    }
} // }}}

response primary_namespace::change_credit_non_blocking(
    request const& req
  , error_code& ec
    )
{ // change_credit_non_blocking implementation
    // parameters
    boost::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_lower_bound();
    naming::gid_type upper = req.get_upper_bound();

    naming::strip_credit_from_gid(lower);
    naming::strip_credit_from_gid(upper);

    // Increment.
    if (0 < credits)
    {
        increment(lower, upper, credits, ec);

        if (ec)
            return response();
    }

    // Decrement.
    else if (0 > credits)
    {
        std::list<free_entry> free_list;
        decrement_sweep(free_list, lower, upper, -credits, ec);

        if (ec)
            return response();

        kill_non_blocking(free_list, lower, upper, ec);

        if (ec)
            return response();
    }

    else
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::change_credit_non_blocking"
          , boost::str(boost::format("invalid credit count of %1%") % credits));
        return response();
    }

    return response(primary_ns_change_credit_non_blocking);
}

response primary_namespace::change_credit_sync(
    request const& req
  , error_code& ec
    )
{ // change_credit_sync implementation
    // parameters
    boost::int64_t credits = req.get_credit();
    naming::gid_type lower = req.get_lower_bound();
    naming::gid_type upper = req.get_upper_bound();

    naming::strip_credit_from_gid(lower);
    naming::strip_credit_from_gid(upper);

    // Increment.
    if (0 < credits)
    {
        increment(lower, upper, credits, ec);

        if (ec)
            return response();
    }

    // Decrement.
    else if (0 > credits)
    {
        std::list<free_entry> free_list;
        decrement_sweep(free_list, lower, upper, -credits, ec);

        if (ec)
            return response();

        kill_sync(free_list, lower, upper, ec);

        if (ec)
            return response();
    }

    else
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::change_credit_sync"
          , boost::str(boost::format("invalid credit count of %1%") % credits));
        return response();
    }

    return response(primary_ns_change_credit_sync);
}

void primary_namespace::increment(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credits
  , error_code& ec
    )
{ // {{{ increment implementation
    mutex_type::scoped_lock l(mutex_);

    // TODO: Whine loudly if a reference count overflows. We reserve ~0 for
    // internal bookkeeping in the decrement algorithm, so the maximum global
    // reference count is 2^64 - 2. The maximum number of credits a single GID
    // can hold, however, is limited to 2^32 - 1.

    // The third parameter we pass here is the default data to use in case the
    // key is not mapped. We don't insert GIDs into the refcnt table when we
    // allocate/bind them, so if a GID is not in the refcnt table, we know that
    // it's global reference count is the initial global reference count.
    refcnts_.apply(lower, upper
                 , util::incrementer<boost::int64_t>(credits)
                 , boost::int64_t(HPX_INITIAL_GLOBALCREDIT));

    LAGAS_(info) << (boost::format(
        "primary_namespace::increment, lower(%1%), upper(%2%), credits(%3%)")
        % lower % upper % credits);

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

void primary_namespace::decrement_sweep(
    std::list<free_entry>& free_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credits
  , error_code& ec
    )
{ // {{{ decrement_sweep implementation
    using boost::fusion::at_c;

    LAGAS_(info) << (boost::format(
        "primary_namespace::decrement_sweep, lower(%1%), upper(%2%), "
        "credits(%3%)")
        % lower % upper % credits);

    free_list.clear();

    {
        mutex_type::scoped_lock l(mutex_);

        ///////////////////////////////////////////////////////////////////////
        // Apply the decrement across the entire keyspace (e.g. [lower, upper]).

        // The third parameter we pass here is the default data to use in case
        // the key is not mapped. We don't insert GIDs into the refcnt table
        // when we allocate/bind them, so if a GID is not in the refcnt table,
        // we know that it's global reference count is the initial global
        // reference count.
        refcnts_.apply(lower, upper
                     , util::decrementer<boost::int64_t>(credits)
                     , boost::int64_t(HPX_INITIAL_GLOBALCREDIT));

        ///////////////////////////////////////////////////////////////////////
        // Search for dead objects.

        typedef refcnt_table_type::iterator iterator;

        // Find the mappings that we just added or modified.
        std::pair<iterator, iterator> matches = refcnts_.find(lower, upper);

        // This search should always succeed.
        if (matches.first == refcnts_.end() && matches.second == refcnts_.end())
        {
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::decrement_sweep"
              , boost::str(boost::format(
                    "reference count table insertion failed due to a locking "
                    "error or memory corruption, lower(%1%), upper(%2%)")
                    % lower % upper));
            return;
        }

        // Ranges containing dead objects.
        std::list<iterator> dead_list;

        for (; matches.first != matches.second; ++matches.first)
        {
            // Sanity check.
            if (matches.first->data_ < 0)
            {
                HPX_THROWS_IF(ec, invalid_data 
                  , "primary_namespace::decrement_sweep"
                  , boost::str(boost::format(
                        "negative entry in reference count table, lower(%1%) "
                        "upper(%2%), count(%3%)")
                        % boost::icl::lower(matches.first->key_)
                        % boost::icl::upper(matches.first->key_)
                        % matches.first->data_));
                return;
            }

            // Any object with a reference count of 0 is dead.
            if (matches.first->data_ == 0)
                dead_list.push_back(matches.first);
        }

        ///////////////////////////////////////////////////////////////////////
        // Resolve the dead objects.

        BOOST_FOREACH(iterator const& it, dead_list)
        {
            // Both the client- and server-side merging algorithms are unaware
            // of component type, so a merged mapping in the reference count
            // tables might contain multiple component types, or the same
            // component type on different localities (currently, the latter is
            // improbable due to the nature of address allocation and the fact
            // that moving objects is still unsupported).

            typedef refcnt_table_type::key_type key_type;

            // The mapping's keyspace.
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
            // mapping's keyspace.
            while (!boost::icl::is_empty(super))
            {
                naming::gid_type query = boost::icl::lower(super);

                // Resolve the query GID.
                boost::fusion::vector2<naming::gid_type, gva>
                    r = resolve_gid_locked(query, ec);

                if (ec)
                    return;

                // Make sure the GVA is valid.
                // REVIEW: Should we do more to make sure the GVA is valid?
                if (HPX_UNLIKELY(components::component_invalid
                                 == at_c<1>(r).type))
                {
                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::decrement_sweep"
                      , boost::str(boost::format(
                            "encountered a GVA with an invalid type while"
                            "performing a decrement, gid(%1%), gva(%2%)")
                            % query % at_c<1>(r)));
                    return;
                }

                else if (HPX_UNLIKELY(0 == at_c<1>(r).count))
                {
                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::decrement_sweep"
                      , boost::str(boost::format(
                            "encountered a GVA with a count of zero while"
                            "performing a decrement, gid(%1%), gva(%2%)")
                            % query % at_c<1>(r)));
                    return;
                }

                // Determine how much of the mapping's keyspace this GVA covers.
                // Note that at_c<1>(r).countmust be greater than 0 if we've
                // reached this point in the code.
                naming::gid_type sub_upper(at_c<0>(r) + (at_c<1>(r).count - 1));

                // If this GVA ends after the keyspace, we just set the upper
                // limit to the end of the keyspace.
                if (sub_upper > boost::icl::upper(super))
                    sub_upper = boost::icl::upper(super);

                BOOST_ASSERT(query <= sub_upper);

                // We don't use the base GID returned by resolve_gid_locked, but
                // instead we use the GID that we queried the GVA table with.
                // This ensures that GVAs which cover a range that begins before
                // our keyspace are handled properly.
                key_type const sub(query, sub_upper);

                LAGAS_(info) << (boost::format(
                    "primary_namespace::decrement_sweep, resolved match, "
                    "lower(%1%), upper(%2%), super-object(%3%), "
                    "sub-object(%4%)")
                    % lower % upper % super % sub);

                // Subtract the GIDs that are bound to this GVA from the
                // keyspace.
                super = boost::icl::left_subtract(super, sub);

                // Compute the length of sub.
                naming::gid_type const length = boost::icl::length(sub);

                // Fully resolve the range.
                gva const g = at_c<1>(r).resolve(query, at_c<0>(r));

                // Add the information needed to destroy these components to the
                // free list.
                free_list.push_back(free_entry(g, query, length));
            }

            // If this is just a partial match, we need to split it up with a
            // remapping so that we can erase it.
            if (super != it->key_)
                // We use ~0 to prevent merges.
                refcnts_.erase(refcnts_.bind(super, boost::int64_t(~0)));
            else
                refcnts_.erase(it);
        }

    } // Unlock the mutex.

    if (&ec != &throws)
        ec = make_success_code();
}

void primary_namespace::kill_non_blocking(
    std::list<free_entry>& free_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , error_code& ec
    )
{ // {{{ kill_non_blocking implementation
    using boost::fusion::at_c;

    naming::gid_type const agas_prefix_
        = naming::get_gid_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX);

    ///////////////////////////////////////////////////////////////////////////
    // Kill the dead objects.

    BOOST_FOREACH(free_entry const& e, free_list)
    {
        // Bail if we're in late shutdown.
        if (HPX_UNLIKELY(!threads::threadmanager_is(running)))
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::kill_non_blocking, cancelling free "
                "operation because the threadmanager is down, lower(%1%), "
                "upper(%2%), base(%3%), gva(%4%), count(%5%)")
                % lower
                % upper
                % at_c<1>(e) % at_c<0>(e) % at_c<2>(e));
            continue;
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::kill_non_blocking, freeing component%1%, "
            "lower(%2%), upper(%3%), base(%4%), gva(%5%), count(%6%)")
            % ((at_c<2>(e) == naming::gid_type(0, 1)) ? "" : "s")
            % lower
            % upper
            % at_c<1>(e) % at_c<0>(e) % at_c<2>(e));

        typedef components::server::runtime_support::free_component_action
            action_type;

        components::component_type const type_ =
            components::component_type(at_c<0>(e).type);

        // FIXME: Resolve the locality instead of deducing it from the
        // target GID, otherwise this will break once we start moving
        // objects.
        if (agas_prefix_ == naming::get_locality_from_gid(at_c<1>(e)))
        {
            naming::address rts_addr(at_c<0>(e).endpoint,
                components::component_runtime_support,
                get_runtime_support_ptr());

            // FIXME: Priority?
            applier::apply_l<action_type>
                (rts_addr, type_, at_c<1>(e), at_c<2>(e));
        }

        else
        {
            // get_lva<> will resolve the LVA to the runtime support pointer
            // on the target locality.
            naming::address rts_addr(at_c<0>(e).endpoint,
                components::component_runtime_support);

            naming::id_type const prefix_(
                naming::get_locality_from_gid(at_c<1>(e)),
                naming::id_type::unmanaged);

            // FIXME: Priority?
            applier::apply_r<action_type>
                (rts_addr, prefix_, type_, at_c<1>(e), at_c<2>(e));
        }
    }

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

void primary_namespace::kill_sync(
    std::list<free_entry>& free_list
  , naming::gid_type const& lower
  , naming::gid_type const& upper
  , error_code& ec
    )
{ // {{{ kill_sync implementation
    using boost::fusion::at_c;

    naming::gid_type const agas_prefix_
        = naming::get_gid_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX);

    std::list<lcos::promise<void> > futures;

    ///////////////////////////////////////////////////////////////////////////
    // Kill the dead objects.

    BOOST_FOREACH(free_entry const& e, free_list)
    {
        // Bail if we're in late shutdown.
        if (HPX_UNLIKELY(!threads::threadmanager_is(running)))
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::kill_sync, cancelling free "
                "operation because the threadmanager is down, lower(%1%), "
                "upper(%2%), base(%3%), gva(%4%), count(%5%)")
                % lower
                % upper
                % at_c<1>(e) % at_c<0>(e) % at_c<2>(e));
            continue;
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::kill_sync, freeing component%1%, "
            "lower(%2%), upper(%3%), base(%4%), gva(%5%), count(%6%)")
            % ((at_c<2>(e) == naming::gid_type(0, 1)) ? "" : "s")
            % lower
            % upper
            % at_c<1>(e) % at_c<0>(e) % at_c<2>(e));

        typedef components::server::runtime_support::free_component_action
            action_type;

        components::component_type const type_ =
            components::component_type(at_c<0>(e).type);

        // FIXME: Resolve the locality instead of deducing it from the
        // target GID, otherwise this will break once we start moving
        // objects.
        naming::id_type const prefix_(
            naming::get_locality_from_gid(at_c<1>(e))
          , naming::id_type::unmanaged);

        futures.push_back(lcos::promise<void>());

        // FIXME: Priority?
        applier::apply_c<action_type>
            (futures.back().get_gid(), prefix_, type_, at_c<1>(e), at_c<2>(e));
    }

    BOOST_FOREACH(lcos::promise<void> const& f, futures)
    {
        f.get();
    }

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

response primary_namespace::localities(
    request const& req
  , error_code& ec
    )
{ // {{{ localities implementation
    using boost::fusion::at_c;

    mutex_type::scoped_lock l(mutex_);

    std::vector<boost::uint32_t> p;

    partition_table_type::const_iterator it = partitions_.begin()
                                       , end = partitions_.end();

    for (; it != end; ++it)
        p.push_back(at_c<0>(it->second));

    LAGAS_(info) << (boost::format(
        "primary_namespace::localities, localities(%1%)")
        % p.size());

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_localities, p);
} // }}}

boost::fusion::vector2<naming::gid_type, gva>
primary_namespace::resolve_gid_locked(
    naming::gid_type const& gid
  , error_code& ec
    )
{ // {{{ resolve_gid implementation
    // parameters
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id);

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

            return boost::fusion::vector2<naming::gid_type, gva>
                (it->first, it->second);
        }

        // We need to decrement the iterator, first we check that it's safe
        // to do this.
        else if (it != begin)
        {
            --it;

            // Found the GID in a range
            if ((it->first + it->second.count) > id)
            {
                if (HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
                {
                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::resolve_gid_locked"
                      , "MSBs of lower and upper range bound do not match");
                    return boost::fusion::vector2<naming::gid_type, gva>
                        (naming::invalid_gid, gva());
                }

                if (&ec != &throws)
                    ec = make_success_code();

                return boost::fusion::vector2<naming::gid_type, gva>
                    (it->first, it->second);
            }
        }
    }

    else if (HPX_LIKELY(!gvas_.empty()))
    {
        --it;

        // Found the GID in a range
        if ((it->first + it->second.count) > id)
        {
            if (HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
            {
                HPX_THROWS_IF(ec, internal_server_error
                  , "primary_namespace::resolve_gid_locked"
                  , "MSBs of lower and upper range bound do not match");
                return boost::fusion::vector2<naming::gid_type, gva>
                    (naming::invalid_gid, gva());
            }

            if (&ec != &throws)
                ec = make_success_code();

            return boost::fusion::vector2<naming::gid_type, gva>
                (it->first, it->second);
        }
    }

    if (&ec != &throws)
        ec = make_success_code();

    return boost::fusion::vector2<naming::gid_type, gva>
        (naming::invalid_gid, gva());
} // }}}

}}}

