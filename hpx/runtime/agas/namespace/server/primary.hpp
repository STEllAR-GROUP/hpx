////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
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
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct primary_namespace : 
  components::fixed_component_base<
    HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
    primary_namespace<Database, Protocol>
  >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef naming::locality endpoint_type;

    typedef gva<Protocol> gva_type;
    typedef typename gva_type::count_type count_type;
    typedef typename gva_type::offset_type offset_type;
    typedef boost::int32_t component_type;
    typedef boost::uint32_t prefix_type;

    typedef boost::fusion::vector2<prefix_type, naming::gid_type>
        partition_type;

    typedef response<Protocol> response_type;

    typedef table<Database, naming::gid_type, gva_type>
        gva_table_type; 

    typedef table<Database, endpoint_type, partition_type>
        partition_table_type;
    
    typedef table<Database, naming::gid_type, count_type>
        refcnt_table_type;
    // }}}
 
  private:
    database_mutex_type mutex_;
    gva_table_type gvas_;
    partition_table_type partitions_;
    refcnt_table_type refcnts_;
    boost::uint32_t prefix_counter_; 

  public:
    primary_namespace(
        std::string const& name = "root_primary_namespace"
        )
      : mutex_(),
        gvas_(std::string("hpx.agas.") + name + ".gva"),
        partitions_(std::string("hpx.agas.") + name + ".partition"),
        refcnts_(std::string("hpx.agas.") + name +".refcnt"),
        prefix_counter_(0)
    {
        traits::initialize_mutex(mutex_);
    }

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

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the partition table 
        typename partition_table_type::map_type&
            partition_table = partitions_.get();

        typename partition_table_type::map_type::iterator
            it = partition_table.find(ep),
            end = partition_table.end(); 

        count_type const real_count = (count) ? (count - 1) : (0);

        // If the endpoint is in the table, then this is a resize.
        if (it != end)
        {
            // Just return the prefix
            if (count == 0)
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
            if (HPX_UNLIKELY(partition_table.size() > 0xFFFFFFFE))
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

            // Load the GVA table, so that we can check if this prefix has
            // already been assigned.
            typename gva_table_type::map_type& gva_table = gvas_.get();

            while (gva_table.count(id))
            {
                prefix = ++prefix_counter_;
                id = naming::get_gid_from_prefix(prefix); 
            }

            // Start assigning ids with the second block of 64bit numbers only.
            naming::gid_type lower_id(id.get_msb() + 1, 0);

            // Create an entry in the partition table for this endpoint
            typename partition_table_type::map_type::iterator pit;

            if (HPX_UNLIKELY(!util::insert_checked(partition_table.insert(
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
            if (HPX_UNLIKELY(!util::insert_checked(gva_table.insert(
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

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the GVA table 
        typename gva_table_type::map_type& gva_table = gvas_.get();

        typename gva_table_type::map_type::iterator
            it = gva_table.lower_bound(id),
            begin = gva_table.begin(),
            end = gva_table.end();

        if (it != end)
        {
            // If we got an exact match, this is a request to update an existing
            // locality binding (e.g. move semantics).
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

            // We're about to decrement it, so make sure it's safe to do so.
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

        else if (HPX_LIKELY(!gva_table.empty()))
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
        if (HPX_UNLIKELY(!util::insert_checked(gva_table.insert(
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
            
        typename database_mutex_type::scoped_lock l(mutex_);
        
        // Load the GVA table 
        typename gva_table_type::map_type const& gva_table = gvas_.get();

        typename gva_table_type::map_type::const_iterator
            it = gva_table.lower_bound(id),
            begin = gva_table.begin(),
            end = gva_table.end();

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

            // We need to decrement the iterator, check that it's safe to do
            // so.
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

        else if (HPX_LIKELY(!gva_table.empty()))
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
 
                // Calculation of the local address occurs in gva<>::resolve()
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

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the partition table 
        typename partition_table_type::map_type&
            partition_table = partitions_.get();

        typename partition_table_type::map_type::iterator
            pit = partition_table.find(ep),
            pend = partition_table.end(); 

        if (pit != pend)
        {
            // Load the GVA table 
            typename gva_table_type::map_type& gva_table = gvas_.get();

            typename gva_table_type::map_type::iterator
                git = gva_table.find(naming::get_gid_from_prefix
                    (at_c<0>(pit->second))),
                gend = gva_table.end();

            if (git == gend)
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
            partition_table.erase(pit);
            gva_table.erase(git);

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
        
        typename database_mutex_type::scoped_lock l(mutex_);
        
        // Load the GVA table 
        typename gva_table_type::map_type& gva_table = gvas_.get();

        typename gva_table_type::map_type::iterator
            it = gva_table.find(id),
            end = gva_table.end();

        if (it != end)
        {
            if (it->second.count != count)
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

            gva_table.erase(it);

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
        
        typename database_mutex_type::scoped_lock l(mutex_);

        typename refcnt_table_type::map_type& refcnt_table = refcnts_.get();

        typename refcnt_table_type::map_type::iterator
            it = refcnt_table.find(id),
            end = refcnt_table.end();

        // If this is the first increment request for this GID, we need to
        // register the GID in the reference counting table
        if (it == end)
        {
            if (HPX_UNLIKELY(!util::insert_checked(refcnt_table.insert(
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
        } 
        
        typename database_mutex_type::scoped_lock l(mutex_);
        
        // Load the reference count table    
        typename refcnt_table_type::map_type& refcnt_table = refcnts_.get();

        typename refcnt_table_type::map_type::iterator
            it = refcnt_table.find(id),
            end = refcnt_table.end();

        if (it != end)
        {
            if (HPX_UNLIKELY(it->second < credits))
            {
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace::decrement" 
                  , "bogus credit encountered while decrement global reference "
                    "count");
            }

            count_type cnt = (it->second -= credits);
            
            if (0 == cnt)
            {
                refcnt_table.erase(it);

                // Load the GVA table 
                typename gva_table_type::map_type& gva_table = gvas_.get();

                typename gva_table_type::map_type::iterator
                    git = gva_table.lower_bound(id),
                    gbegin = gva_table.begin(),
                    gend = gva_table.end();

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
                
                else if (HPX_LIKELY(!gva_table.empty()))
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
            // Load the GVA table 
            typename gva_table_type::map_type& gva_table = gvas_.get();

            typename gva_table_type::map_type::iterator
                git = gva_table.lower_bound(id),
                gbegin = gva_table.begin(),
                gend = gva_table.end();

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
            
            else if (HPX_LIKELY(!gva_table.empty()))
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
            if (HPX_UNLIKELY(!util::insert_checked(refcnt_table.insert(
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

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the partition table 
        typename partition_table_type::map_type const&
            partition_table = partitions_.get();

        std::vector<boost::uint32_t> p;

        typename partition_table_type::map_type::const_iterator
            it = partition_table.begin(),
            end = partition_table.end(); 

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
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind_locality,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace<Database, Protocol>::bind_locality
      , threads::thread_priority_critical
    > bind_locality_action; 
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_bind_gid,
        /* arguments */   naming::gid_type const&, gva_type const&,
        &primary_namespace<Database, Protocol>::bind_gid
      , threads::thread_priority_critical
    > bind_gid_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_page_fault,
        /* arguments */   naming::gid_type const&,
        &primary_namespace<Database, Protocol>::page_fault
      , threads::thread_priority_critical
    > page_fault_action;

    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_unbind_locality,
        /* arguments */   endpoint_type const&,
        &primary_namespace<Database, Protocol>::unbind_locality
      , threads::thread_priority_critical
    > unbind_locality_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_unbind_gid,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::unbind_gid
      , threads::thread_priority_critical
    > unbind_gid_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,  
        /* enum value */  namespace_increment,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::increment
      , threads::thread_priority_critical
    > increment_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::decrement
      , threads::thread_priority_critical
    > decrement_action;
    
    typedef hpx::actions::result_action0<
        primary_namespace<Database, Protocol>,
        /* return type */ response_type,
        /* enum value */  namespace_localities,
        &primary_namespace<Database, Protocol>::localities
      , threads::thread_priority_critical
    > localities_action;
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

