////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/response.hpp>

namespace hpx { namespace agas { namespace server
{

/// \brief AGAS's primary namespace maps 128-bit global identifiers (GIDs) to
/// resolved addresses.
///
/// \note The layout of the address space is implementation defined, and
/// subject to change. Never write application code that relies on the internal
/// layout of GIDs. AGAS only guarantees that all assigned GIDs will be unique.
/// 
/// The following is the canonical description of the partitioning of AGAS's
/// primary namespace.
///
///     |-----MSB------||------LSB-----|
///     BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
///     |prefix||RC||----identifier----|
///     
///     MSB        - Most significant bits (bit 64 to bit 127)
///     LSB        - Least significant bits (bit 0 to bit 63)
///     prefix     - Highest 32 bits (bit 96 to bit 127) of the MSB. Each
///                  locality is assigned a prefix. This creates a 96-bit
///                  address space for each locality.
///     RC         - Bit 80 to bit 95 of the MSB. This is the number of
///                  reference counting credits on the GID.
///     identifier - Bit 64 to bit 80 of the MSB, and the entire LSB. The
///                  content of these bits depends on the component type of
///                  the underlying object. For all user-defined components,
///                  these bits contain a unique 80-bit number which is
///                  assigned sequentially for each locality. For
///                  \a hpx#components#component_runtime_support and
///                  \a hpx#components#component_memory, the high 16 bits are
///                  zeroed and the low 64 bits hold the LVA of the component.
///
/// The following address ranges are reserved. Some are either explicitly or 
/// implicitly protected by AGAS. The letter x represents a single-byte
/// wildcard.
///
///     00000000xxxxxxxxxxxxxxxxxxxxxxxx
///         Historically unused address space reserved for future use.
///     xxxxxxxxxxxx0000xxxxxxxxxxxxxxxx
///         Address space for LVA-encoded GIDs.
///     00000001xxxxxxxxxxxxxxxxxxxxxxxx
///         Prefix of the bootstrap AGAS locality.
///     00000001000000010000000000000001
///         Address of the primary_namespace component on the bootstrap AGAS
///         locality.
///     00000001000000010000000000000002 
///         Address of the component_namespace component on the bootstrap AGAS
///         locality.
///     00000001000000010000000000000003
///         Address of the symbol_namespace component on the bootstrap AGAS
///         locality.
///
struct primary_namespace : 
  components::fixed_component_base<
    HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
    primary_namespace
  >
{
    // {{{ nested types
    typedef util::spinlock database_mutex_type;

    typedef naming::locality endpoint_type;

    typedef gva gva_type;
    typedef gva_type::count_type count_type;
    typedef gva_type::offset_type offset_type;
    typedef boost::int32_t component_type;
    typedef boost::uint32_t prefix_type;

    typedef boost::fusion::vector2<prefix_type, naming::gid_type>
        partition_type;

    typedef response response_type;

    typedef std::map<naming::gid_type, gva_type>
        gva_table_type; 

    typedef std::map<endpoint_type, partition_type>
        partition_table_type;
    
    typedef std::map<naming::gid_type, count_type>
        refcnt_table_type;
    // }}}
 
  private:
    database_mutex_type mutex_;
    gva_table_type gvas_;
    partition_table_type partitions_;
    refcnt_table_type refcnts_;
    boost::uint32_t prefix_counter_; 

  public:
    primary_namespace()
      : mutex_()
      , gvas_()
      , partitions_()
      , refcnts_()
      , prefix_counter_(0)
    {}

    response_type bind_locality(
        endpoint_type const& ep
      , count_type count
        )
    { 
        return bind_locality(ep, count, throws);
    } 

    response_type bind_locality(
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

                return response_type(primary_ns_bind_locality
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
                    return response_type();
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

            return response_type(primary_ns_bind_locality
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
                return response_type();
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
                return response_type();
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
                return response_type();
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

            return response_type(primary_ns_bind_locality
                , lower, upper, prefix);
        }
    } // }}}

    response_type bind_gid(
        naming::gid_type const& gid
      , gva_type const& gva
        )
    {
        return bind_gid(gid, gva, throws);
    }

    response_type bind_gid(
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
                    return response_type();
                }

                if (HPX_UNLIKELY(gva.type == components::component_invalid))
                {
                    HPX_THROWS_IF(ec, bad_parameter
                      , "primary_namespace::bind_gid" 
                      , boost::str(boost::format(
                            "attempt to update a GVA with an invalid type, "
                            "gid(%1%), gva(%2%)")
                            % id % gva));
                    return response_type();
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

                return response_type(primary_ns_bind_gid, repeated_request);
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
                    return response_type();
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
                return response_type();
            }
        }

        naming::gid_type upper_bound(id + (gva.count - 1));

        if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
        {
            HPX_THROWS_IF(ec, internal_server_error
              , "primary_namespace::bind_gid" 
              , "MSBs of lower and upper range bound do not match");
            return response_type();
        }

        if (HPX_UNLIKELY(gva.type == components::component_invalid))
        {
            HPX_THROWS_IF(ec, bad_parameter
              , "primary_namespace::bind_gid" 
              , boost::str(boost::format(
                    "attempt to insert a GVA with an invalid type, "
                    "gid(%1%), gva(%2%)")
                    % id % gva));
            return response_type();
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
            return response_type();
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::bind_gid, gid(%1%), gva(%2%)")
            % gid % gva);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(primary_ns_bind_gid);
    } // }}}

    response_type page_fault(
        naming::gid_type const& gid
        )
    {
        return page_fault(gid, throws);
    }

    response_type page_fault(
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

                return response_type(primary_ns_page_fault
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
                        return response_type();
                    }
 
                    // Calculation of the lva address occurs in gva<>::resolve()
                    LAGAS_(info) << (boost::format(
                        "primary_namespace::page_fault, soft page fault, "
                        "faulting address %1%, gva(%2%)")
                        % gid % it->second);

                    if (&ec != &throws)
                        ec = make_success_code();

                    return response_type(primary_ns_page_fault
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
                    return response_type();
                }
 
                LAGAS_(info) << (boost::format(
                    "primary_namespace::page_fault, soft page fault, faulting "
                    "address %1%, gva(%2%)")
                    % gid % it->second);

                if (&ec != &throws)
                    ec = make_success_code();

                return response_type(primary_ns_page_fault
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

        return response_type(primary_ns_page_fault
                           , naming::invalid_gid 
                           , gva_type()
                           , invalid_page_fault);
    } // }}}

    response_type unbind_locality(
        endpoint_type const& ep
        )
    {
        return unbind_locality(ep, throws);
    }

    response_type unbind_locality(
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
                return response_type();
            }

            // Wipe the locality from the tables.
            partitions_.erase(pit);
            gvas_.erase(git);

            LAGAS_(info) << (boost::format(
                "primary_namespace::unbind_locality, ep(%1%)")
                % ep);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(primary_ns_unbind_locality);
        }

        LAGAS_(info) << (boost::format(
            "primary_namespace::unbind_locality, ep(%1%), "
            "response(no_success)")
            % ep);

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(primary_ns_unbind_locality
                           , no_success);
    } // }}}

    response_type unbind_gid(
        naming::gid_type const& gid
      , count_type count
        )
    {
        return unbind_gid(gid, count, throws);
    }

    response_type unbind_gid(
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
                return response_type();
            }

            response_type r(primary_ns_unbind_gid, it->second);
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

            return response_type(primary_ns_unbind_gid
                               , gva_type()
                               , no_success);
        }
    } // }}}

    response_type increment(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return increment(gid, credits, throws);
    }

    response_type increment(
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
                return response_type();
            }
        }
       
        // Add the requested amount and return the new total 
        LAGAS_(info) << (boost::format(
            "primary_namespace::increment, gid(%1%), credits(%2%), "
            "new_count(%3%)")
            % gid % credits % (it->second + credits));

        if (&ec != &throws)
            ec = make_success_code();

        return response_type(primary_ns_increment, it->second += credits);
    } // }}}
    
    response_type decrement(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return decrement(gid, credits, throws); 
    }
       
    response_type decrement(
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
            return response_type();
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
                return response_type();
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
                        return response_type(primary_ns_decrement
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
                                return response_type(primary_ns_decrement
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
                            return response_type(primary_ns_decrement
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
                return response_type();
            }

            else
            {
                LAGAS_(info) << (boost::format(
                    "primary_namespace::decrement, gid(%1%), credits(%2%), "
                    "new_total(%3%)")
                    % gid % credits % cnt);

                if (&ec != &throws)
                    ec = make_success_code();

                return response_type(primary_ns_decrement
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
                    return response_type(primary_ns_decrement
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
                            return response_type(primary_ns_decrement
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
                        return response_type(primary_ns_decrement
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
            return response_type();
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
                return response_type();
            }
            
            if (HPX_UNLIKELY(it->second < credits))
            {
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::decrement" 
                  , "bogus credit encountered while decrement global reference "
                    "count");
                return response_type();
            }
            
            count_type cnt = (it->second -= credits);

            LAGAS_(info) << (boost::format(
                "primary_namespace::decrement, gid(%1%), credits(%2%), "
                "new_total(%3%)")
                % gid % credits % cnt);

            if (&ec != &throws)
                ec = make_success_code();

            return response_type(primary_ns_decrement
                               , cnt
                               , components::component_invalid);
        }
    } // }}}
 
    response_type localities()
    {
        return localities(throws);
    }

    response_type localities(
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

        return response_type(primary_ns_localities, p);
    } // }}}

    enum actions 
    { // {{{ action enum
        namespace_bind_locality    = BOOST_BINARY_U(1000000),
        namespace_bind_gid         = BOOST_BINARY_U(1000001),
        namespace_page_fault      = BOOST_BINARY_U(1000010),
        namespace_unbind_locality  = BOOST_BINARY_U(1000011),
        namespace_unbind_gid       = BOOST_BINARY_U(1000100),
        namespace_increment        = BOOST_BINARY_U(1000101),
        namespace_decrement        = BOOST_BINARY_U(1000110),
        namespace_localities       = BOOST_BINARY_U(1000111),
    }; // }}}
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_bind_locality,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace::bind_locality
      , threads::thread_priority_critical
    > bind_locality_action; 
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_bind_gid,
        /* arguments */   naming::gid_type const&, gva_type const&,
        &primary_namespace::bind_gid
      , threads::thread_priority_critical
    > bind_gid_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_page_fault,
        /* arguments */   naming::gid_type const&,
        &primary_namespace::page_fault
      , threads::thread_priority_critical
    > page_fault_action;

    typedef hpx::actions::result_action1<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_unbind_locality,
        /* arguments */   endpoint_type const&,
        &primary_namespace::unbind_locality
      , threads::thread_priority_critical
    > unbind_locality_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_unbind_gid,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::unbind_gid
      , threads::thread_priority_critical
    > unbind_gid_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response_type,  
        /* enum value */  namespace_increment,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::increment
      , threads::thread_priority_critical
    > increment_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace::decrement
      , threads::thread_priority_critical
    > decrement_action;
    
    typedef hpx::actions::result_action0<
        primary_namespace,
        /* return type */ response_type,
        /* enum value */  namespace_localities,
        &primary_namespace::localities
      , threads::thread_priority_critical
    > localities_action;
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

