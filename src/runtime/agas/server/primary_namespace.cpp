////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>

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

response primary_namespace::service(
    request const& req
  , error_code& ec
    )
{
    // IMPLEMENT
    return response();
}

response primary_namespace::bind_locality(
    endpoint_type const& ep
  , count_type count
  , error_code& ec
    )
{ // {{{ bind_locality implementation
    using boost::fusion::at_c;

    database_mutex_type::scoped_lock l(mutex_);

    partition_table_type::iterator it = partitions_.find(ep)
                                 , end = partitions_.end(); 

    count_type const real_count = (count) ? (count - 1) : (0);

    // If the endpoint is in the table, then this is a resize.
    if (it != end)
    {
        // Just return the prefix
        if (0 == count)
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::bind_locality, ep(%1%), count(%2%), "
                "lower(%3%), upper(%4%), prefix(%5%)")
                % ep
                % count
                % at_c<1>(it->second)
                % at_c<1>(it->second)
                % at_c<0>(it->second));

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_bind_locality
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
                  , "primary_namespace::bind_locality" 
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
            "primary_namespace::bind_locality, ep(%1%), count(%2%), "
            "lower(%3%), upper(%4%), prefix(%5%)")
            % ep % count % lower % upper % at_c<0>(it->second));

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_bind_locality
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
              , "primary_namespace::bind_locality" 
              , "primary namespace has been exhausted");
            return response();
        }

        // Compute the locality's prefix.
        boost::uint32_t prefix = ++prefix_counter_;

        // Don't allow 0 to be used as a prefix.
        if (0 == prefix)
            prefix = ++prefix_counter_;

        naming::gid_type id(naming::get_gid_from_prefix(prefix));

        // Check if this prefix has already been assigned.
        while (gvas_.count(id))
        {
            prefix = ++prefix_counter_;
            id = naming::get_gid_from_prefix(prefix); 
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
              , "primary_namespace::bind_locality" 
              , boost::str(boost::format(
                    "partition table insertion failed due to a locking "
                    "error or memory corruption, endpoint(%1%), "
                    "prefix(%2%), lower_id(%3%)")
                    % ep % prefix % lower_id));
            return response();
        }

        const gva_type gva
            (ep, components::component_runtime_support, count);

        // Now that we've inserted the locality into the partition table
        // successfully, we need to put the locality's GID into the GVA
        // table so that parcels can be sent to the memory of a locality.
        if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
                std::make_pair(id, gva)))))
        {
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::bind_locality"  
              , boost::str(boost::format(
                    "GVA table insertion failed due to a locking error "
                    "or memory corruption, gid(%1%), gva(%2%)")
                    % id % gva));
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
            "primary_namespace::bind_locality, ep(%1%), count(%2%), "
            "lower(%3%), upper(%4%), prefix(%5%)")
            % ep % count % lower % upper % prefix);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_bind_locality
            , lower, upper, prefix);
    }
} // }}}

response primary_namespace::bind_gid(
    naming::gid_type const& gid
  , gva_type const& gva
  , error_code& ec
    )
{ // {{{ bind_gid implementation
    using boost::fusion::at_c;

    // TODO: Implement and use a non-mutating version of
    // strip_credit_from_gid()
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id); 

    database_mutex_type::scoped_lock l(mutex_);

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
            if (HPX_UNLIKELY(it->second.count != gva.count))
            {
                // REVIEW: Is this the right error code to use?
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid" 
                  , "cannot change block size of existing binding");
                return response();
            }

            if (HPX_UNLIKELY(gva.type == components::component_invalid))
            {
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::bind_gid" 
                  , boost::str(boost::format(
                        "attempt to update a GVA with an invalid type, "
                        "gid(%1%), gva(%2%)")
                        % id % gva));
                return response();
            }

            // Store the new endpoint and offset
            it->second.endpoint = gva.endpoint;
            it->second.type = gva.type;
            it->second.lva(gva.lva());
            it->second.offset = gva.offset;

            LAGAS_(info) << (boost::format(
                "primary_namespace::bind_gid, gid(%1%), gva(%2%), "
                "response(repeated_request)")
                % gid % gva);

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

    naming::gid_type upper_bound(id + (gva.count - 1));

    if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
    {
        HPX_THROWS_IF(ec, internal_server_error
          , "primary_namespace::bind_gid" 
          , "MSBs of lower and upper range bound do not match");
        return response();
    }

    if (HPX_UNLIKELY(gva.type == components::component_invalid))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::bind_gid" 
          , boost::str(boost::format(
                "attempt to insert a GVA with an invalid type, "
                "gid(%1%), gva(%2%)")
                % id % gva));
        return response();
    }
    
    // Insert a GID -> GVA entry into the GVA table. 
    if (HPX_UNLIKELY(!util::insert_checked(gvas_.insert(
            std::make_pair(id, gva)))))
    {
        HPX_THROWS_IF(ec, lock_error 
          , "primary_namespace::bind_gid"
          , boost::str(boost::format(
                "GVA table insertion failed due to a locking error or "
                "memory corruption, gid(%1%), gva(%2%)")
                % id % gva));
        return response();
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::bind_gid, gid(%1%), gva(%2%)")
        % gid % gva);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_bind_gid);
} // }}}

response primary_namespace::page_fault(
    naming::gid_type const& gid
  , error_code& ec
    )
{ // {{{ page_fault implementation 
    // TODO: Implement and use a non-mutating version of
    // strip_credit_from_gid()
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id); 
        
    database_mutex_type::scoped_lock l(mutex_);

    gva_table_type::const_iterator it = gvas_.lower_bound(id)
                                 , begin = gvas_.begin()
                                 , end = gvas_.end();

    if (it != end)
    {
        // Check for exact match
        if (it->first == id)
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::page_fault, soft page fault, faulting "
                "address %1%, gva(%2%)")
                % gid % it->second);

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_page_fault
                               , it->first
                               , it->second);
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
                    HPX_THROWS_IF(ec, invalid_page_fault
                      , "primary_namespace::page_fault" 
                      , "MSBs of lower and upper range bound do not match");
                    return response();
                }

                // Calculation of the lva address occurs in gva<>::resolve()
                LAGAS_(info) << (boost::format(
                    "primary_namespace::page_fault, soft page fault, "
                    "faulting address %1%, gva(%2%)")
                    % gid % it->second);

                if (&ec != &throws)
                    ec = make_success_code();

                return response(primary_ns_page_fault
                                   , it->first
                                   , it->second);
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
                HPX_THROWS_IF(ec, invalid_page_fault
                  , "primary_namespace::page_fault" 
                  , "MSBs of lower and upper range bound do not match");
                return response();
            }

            LAGAS_(info) << (boost::format(
                "primary_namespace::page_fault, soft page fault, faulting "
                "address %1%, gva(%2%)")
                % gid % it->second);

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_page_fault
                               , it->first
                               , it->second);
        }
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::page_fault, invalid page fault, faulting "
        "address %1%")
        % gid);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_page_fault
                       , naming::invalid_gid 
                       , gva_type()
                       , invalid_page_fault);
} // }}}

response primary_namespace::unbind_locality(
    endpoint_type const& ep
  , error_code& ec
    )
{ // {{{ unbind_locality implementation
    using boost::fusion::at_c;

    database_mutex_type::scoped_lock l(mutex_);

    partition_table_type::iterator pit = partitions_.find(ep)
                                 , pend = partitions_.end(); 

    if (pit != pend)
    {
        gva_table_type::iterator git = gvas_.find
            (naming::get_gid_from_prefix(at_c<0>(pit->second)));
        gva_table_type::iterator gend = gvas_.end();

        if (HPX_UNLIKELY(git == gend))
        {
            HPX_THROWS_IF(ec, internal_server_error
              , "primary_namespace::unbind_locality" 
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
            "primary_namespace::unbind_locality, ep(%1%)")
            % ep);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_unbind_locality);
    }

    LAGAS_(info) << (boost::format(
        "primary_namespace::unbind_locality, ep(%1%), "
        "response(no_success)")
        % ep);

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_unbind_locality
                       , no_success);
} // }}}

response primary_namespace::unbind_gid(
    naming::gid_type const& gid
  , count_type count
  , error_code& ec
    )
{ // {{{ unbind_gid implementation
    // TODO: Implement and use a non-mutating version of
    // strip_credit_from_gid()
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id); 
    
    database_mutex_type::scoped_lock l(mutex_);

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
            % gid % count % it->second);

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
            % gid % count);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_unbind_gid
                           , gva_type()
                           , no_success);
    }
} // }}}

response primary_namespace::increment(
    naming::gid_type const& gid
  , count_type credits
  , error_code& ec
    )
{ // {{{ increment implementation
    // TODO: Implement and use a non-mutating version of
    // strip_credit_from_gid()
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id); 
    
    database_mutex_type::scoped_lock l(mutex_);

    refcnt_table_type::iterator it = refcnts_.find(id)
                              , end = refcnts_.end();

    // If this is the first increment request for this GID, we need to
    // register the GID in the reference counting table
    if (it == end)
    {
        if (HPX_UNLIKELY(!util::insert_checked(refcnts_.insert(
                std::make_pair(id,
                    count_type(HPX_INITIAL_GLOBALCREDIT))), it)))
        {
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::increment" 
              , boost::str(boost::format(
                    "refcnt table insertion failed due to a locking error "
                    "or memory corruption, gid(%1%)")
                    % id));
            return response();
        }
    }
   
    // Add the requested amount and return the new total 
    LAGAS_(info) << (boost::format(
        "primary_namespace::increment, gid(%1%), credits(%2%), "
        "new_count(%3%)")
        % gid % credits % (it->second + credits));

    if (&ec != &throws)
        ec = make_success_code();

    return response(primary_ns_increment, it->second += credits);
} // }}}
   
response primary_namespace::decrement(
    naming::gid_type const& gid
  , count_type credits
  , error_code& ec
    )
{ // {{{ decrement implementation
    // TODO: Implement and use a non-mutating version of
    // strip_credit_from_gid()
    naming::gid_type id = gid;
    naming::strip_credit_from_gid(id); 

    if (HPX_UNLIKELY(credits > HPX_INITIAL_GLOBALCREDIT))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::decrement" 
          , "cannot decrement more than "
            BOOST_PP_STRINGIZE(HPX_INITIAL_GLOBALCREDIT)
            " credits");
        return response();
    } 
    
    database_mutex_type::scoped_lock l(mutex_);
    
    refcnt_table_type::iterator it = refcnts_.find(id)
                              , end = refcnts_.end();

    if (it != end)
    {
        if (HPX_UNLIKELY(it->second < credits))
        {
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::decrement" 
              , "bogus credit encountered while decrement global reference "
                "count");
            return response();
        }

        count_type cnt = (it->second -= credits);
        
        if (0 == cnt)
        {
            refcnts_.erase(it);

            gva_table_type::iterator git = gvas_.lower_bound(id)
                                   , gbegin = gvas_.begin()
                                   , gend = gvas_.end();

            if (git != gend)
            {
                // Did we get an exact match?
                if (git->first == id)
                {
                    LAGAS_(info) << (boost::format(
                        "primary_namespace::decrement, gid(%1%), "
                        "credits(%2%), new_total(0), type(%3%)")
                        % gid % credits % git->second.type);

                    if (&ec != &throws)
                        ec = make_success_code();

                    // TODO: Check that git->second.type isn't invalid?
                    return response(primary_ns_decrement
                                       , cnt
                                       , git->second.type);
                }

                // Check if we can safely decrement the iterator.
                else if (git != gbegin)
                {
                    --git;

                    // See if the previous range covers this GID
                    if ((git->first + git->second.count) > id)
                    {
                        // Make sure that the MSBs match.
                        // TODO: Shouldn't this be an error if the MSBs
                        // don't match?
                        if (id.get_msb() == git->first.get_msb())
                        {
                            LAGAS_(info) << (boost::format(
                                "primary_namespace::decrement, gid(%1%), "
                                "credits(%2%), new_total(0), type(%3%)")
                                % gid % credits % git->second.type);

                            if (&ec != &throws)
                                ec = make_success_code();

                            // TODO: Check that git->second.type isn't
                            // invalid?
                            return response(primary_ns_decrement
                                               , cnt
                                               , git->second.type);
                        }
                    } 
                }
            }
            
            else if (HPX_LIKELY(!gvas_.empty()))
            {
                --git;

                // See if the previous range covers this GID
                if ((git->first + git->second.count) > id)
                {
                    // Make sure that the MSBs match
                    // TODO: Shouldn't this be an error if the MSBs
                    // don't match?
                    if (id.get_msb() == git->first.get_msb())
                    {
                        LAGAS_(info) << (boost::format(
                            "primary_namespace::decrement, gid(%1%), "
                            "credits(%2%), new_total(0), type(%3%)")
                            % gid % credits % git->second.type);

                        if (&ec != &throws)
                            ec = make_success_code();

                        // TODO: Check that git->second.type isn't invalid?
                        return response(primary_ns_decrement
                                           , cnt
                                           , git->second.type);
                    }
                } 
            }

            // If we didn't find anything, we've got a problem
            // TODO: Use a better error code + throw message.
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::decrement"
              , "unregistered GID encountered while decrementing global "
                "reference count");
            return response();
        }

        else
        {
            LAGAS_(info) << (boost::format(
                "primary_namespace::decrement, gid(%1%), credits(%2%), "
                "new_total(%3%)")
                % gid % credits % cnt);

            if (&ec != &throws)
                ec = make_success_code();

            return response(primary_ns_decrement
                               , cnt
                               , components::component_invalid);
        }
    }
    
    // If the id isn't in the refcnt table and the credit count is
    // HPX_INITIAL_GLOBALCREDIT, then it needs to be destroyed. 
    else if (HPX_INITIAL_GLOBALCREDIT == credits)
    {
        gva_table_type::iterator git = gvas_.lower_bound(id)
                               , gbegin = gvas_.begin()
                               , gend = gvas_.end();

        if (git != gend)
        {
            // Did we get an exact match?
            if (git->first == id)
            {
                LAGAS_(info) << (boost::format(
                    "primary_namespace::decrement, gid(%1%), credits(%2%), "
                    "new_total(0), type(%3%)")
                    % gid % credits % git->second.type);

                if (&ec != &throws)
                    ec = make_success_code();

                // TODO: Check that git->second.type isn't invalid?
                return response(primary_ns_decrement
                                   , 0
                                   , git->second.type);
            }

            // Check if we can safely decrement the iterator.
            else if (git != gbegin)
            {
                --git;

                // See if the previous range covers this GID
                if ((git->first + git->second.count) > id)
                {
                    // Make sure that the MSBs match
                    // TODO: Shouldn't this be an error if the MSBs
                    // don't match?
                    if (id.get_msb() == git->first.get_msb())
                    {
                        LAGAS_(info) << (boost::format(
                            "primary_namespace::decrement, gid(%1%), "
                            "credits(%2%), new_total(0), type(%3%)")
                            % gid % credits % git->second.type);

                        if (&ec != &throws)
                            ec = make_success_code();

                        // TODO: Check that git->second.type isn't invalid?
                        return response(primary_ns_decrement
                                           , 0
                                           , git->second.type);
                    }
                } 
            }
        }
        
        else if (HPX_LIKELY(!gvas_.empty()))
        {
            --git;

            // See if the previous range covers this GID
            if ((git->first + git->second.count) > id)
            {
                // Make sure that the MSBs match
                // TODO: Shouldn't this be an error if the MSBs
                // don't match?
                if (id.get_msb() == git->first.get_msb())
                {
                    LAGAS_(info) << (boost::format(
                        "primary_namespace::decrement, gid(%1%), "
                        "credits(%2%), new_total(0), type(%3%)")
                        % gid % credits % git->second.type);

                    if (&ec != &throws)
                        ec = make_success_code();

                    // TODO: Check that git->second.type isn't invalid?
                    return response(primary_ns_decrement
                                       , 0
                                       , git->second.type);
                }
            } 
        }

        // If we didn't find anything, we've got a problem
        HPX_THROWS_IF(ec, bad_parameter
          , "primary_namespace::decrement" 
          , "unknown component type encountered while decrement global "
            "reference count");
        return response();
    }

    // We need to insert a new reference count entry. We assume that
    // binding has already created a first reference + credits.
    else 
    {
        if (HPX_UNLIKELY(!util::insert_checked(refcnts_.insert(
                std::make_pair(id,
                    count_type(HPX_INITIAL_GLOBALCREDIT))), it)))
        {
            HPX_THROWS_IF(ec, lock_error
              , "primary_namespace::decrement" 
              , boost::str(boost::format(
                    "refcnt table insertion failed due to a locking error "
                    "or memory corruption, gid(%1%)")
                    % id));
            return response();
        }
        
        if (HPX_UNLIKELY(it->second < credits))
        {
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::decrement" 
              , "bogus credit encountered while decrement global reference "
                "count");
            return response();
        }
        
        count_type cnt = (it->second -= credits);

        LAGAS_(info) << (boost::format(
            "primary_namespace::decrement, gid(%1%), credits(%2%), "
            "new_total(%3%)")
            % gid % credits % cnt);

        if (&ec != &throws)
            ec = make_success_code();

        return response(primary_ns_decrement
                           , cnt
                           , components::component_invalid);
    }
} // }}}

response primary_namespace::localities(
    error_code& ec
    )
{ // {{{ localities implementation
    using boost::fusion::at_c;

    database_mutex_type::scoped_lock l(mutex_);

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

}}}

