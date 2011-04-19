////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <boost/preprocessor/stringize.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/database/table.hpp>
#include <hpx/runtime/agas/network/gva.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct HPX_COMPONENT_EXPORT primary_namespace
  : components::simple_component_base<primary_namespace<Database, Protocol> >
{
    // {{{ nested types
    typedef typename traits::database::mutex_type<Database>::type
        database_mutex_type;

    typedef typename traits::network::endpoint_type<Protocol>::type
        endpoint_type;

    typedef gva<Protocol> gva_type;
    typedef full_gva<Protocol> full_gva_type;
    typedef typename full_gva_type::count_type count_type;
    typedef typename full_gva_type::offset_type offset_type;
    typedef components::component_type component_type;

    typedef boost::fusion::vector2<count_type, component_type>
        decrement_result_type;

    typedef boost::fusion::vector2<boost::uint32_t, naming::gid_type>
        partition_type;

    typedef boost::fusion::vector2<naming::gid_type, naming::gid_type>
        range_type;

    typedef table<Database, naming::gid_type, full_gva_type>
        gva_table_type; 

    typedef table<Database, endpoint_type, partition_type>
        partition_table_type;
    
    typedef table<Database, naming::gid_type, count_type>
        refcnt_table_type;
    // }}}
 
  private:
    mutable database_mutex_type mutex_;
    gva_table_type gvas_;
    partition_table_type partitions_;
    refcnt_table_type refcnts_;
  
  public:
    primary_namespace()
      : mutex_(),
        gvas_("hpx.agas.primary_namespace.gva"),
        partitions_("hpx.agas.primary_namespace.partition"),
        refcnts_("hpx.agas.primary_namespace.refcnt")
    { traits::initialize_mutex(mutex_); }

    range_type bind_locality(endpoint_type const& ep, count_type count)
    { // {{{ bind_locality implementation
        using boost::fusion::at_c;

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the GVA table 
        typename partition_table_type::map_type&
            partition_table = partitions_.get();

        typename partition_table_type::map_type::iterator
            it = partition_table.find(ep),
            end = partition_table.end(); 

        if (it != end)
        {
            // Compute the new allocation.
            naming::gid_type lower(at_c<1>(it->second) + 1),
                             upper(lower + (count - 1));

            // Check for overflow.
            if (upper.get_msb() != lower.get_msb())
            {
                // Check for address space exhaustion 
                if (HPX_UNLIKELY((lower.get_msb() & ~0xFFFFFFFF) == 0xFFFFFFF))
                {
                    HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                        "primary namespace has been exhausted");
                }

                // Otherwise, correct
                lower = naming::gid_type(upper.get_msb(), 0);
                upper = lower + (count - 1); 
            }
            
            // Store the new upper bound.
            at_c<1>(it->second) = upper;

            // Set the initial credit count.
            naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
            naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT); 

            return range_type(lower, upper);
        }

        else
        {
            // Check for address space exhaustion.
            if (HPX_UNLIKELY(partition_table.size() > 0xFFFFFFFE))
            {
                HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                    "primary namespace has been exhausted");
            }

            // Compute the locality's prefix
            boost::uint32_t prefix = reinterpret_cast<boost::uint32_t>
                (partition_table.size() + 1);
            naming::gid_type id(naming::get_gid_from_prefix(prefix));

            // Start assigning ids with the seconds block of 64bit numbers only
            naming::gid_type lower_id(id.get_msb() + 1, 0);

            std::pair<typename partition_table_type::map_type::iterator, bool>
                pit = partition_table.insert(ep,
                    partition_type(prefix, lower_id));

            // REVIEW: Should this be an assertion?
            // Check for an insertion failure. If this branch is triggered, then
            // the partition table was updated at some point after we first
            // checked it, which would indicate memory corruption or a locking
            // failure.
            if (HPX_UNLIKELY(!pit.second))
            {
                HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                    "insertion failed due to memory corruption or a locking "
                    "error");
            }

            // Load the GVA table. Now that we've inserted the locality into
            // the partition table successfully, we need to put the locality's
            // GID into the GVA table so that parcels can be sent to the memory
            // of a locality.
            typename gva_table_type::map_type& gva_table = gvas_.get();

            std::pair<typename gva_table_type::map_type::iterator, bool>
                git = gva_table.insert(id, full_gva_type
                    (ep, components::component_runtime_support, 1, 0));

            // REVIEW: Should this be an assertion?
            // Check for insertion failure.
            if (HPX_UNLIKELY(!git.second))
            {
                // REVIEW: Is this the right error code to use?
                HPX_THROW_IN_CURRENT_FUNC(no_success, 
                    "insertion failed due to memory corruption or a locking "
                    "error");
            }

            // Generate the requested GID range
            naming::gid_type lower(lower_id + 1), upper(lower + (count - 1));

            at_c<1>((*pit.first).second) = upper;

            // Set the initial credit count.
            naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
            naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT); 

            return range_type(lower, upper);
        }
    } // }}}

    bool bind_gid(naming::gid_type const& gid, gva_type const& gva,
                  count_type count, offset_type offset)
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
                if (it->second.count != count)
                {
                    // REVIEW: Is this the right error code to use?
                    HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                        "cannot change block size of existing binding");
                }

                // Store the new endpoint and offset
                it->second.endpoint = gva.endpoint;
                it->second.type = gva.type;
                it->second.lva = gva.lva;
                it->second.offset = offset;
                return true;
            }

            // We're about to decrement it, so make sure it's safe to do so.
            else if (it != begin)
            {
                --it;

                // Check that a previous range doesn't cover the new id.
                if ((it->first + it->second.count) > id)
                    return false;
            }
        }            

        else if (HPX_LIKELY(!gva_table.empty()))
        {
            --it;
            // Check that a previous range doesn't cover the new id.
            if ((it->first + it->second.count) > id)
                return false;
        }

        naming::gid_type upper_bound(id + (count - 1));

        if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
        {
            HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                "MSBs of lower and upper range bound do not match");
        }

        std::pair<typename gva_table_type::map_type::iterator, bool>
            p = gva_table.insert(id, full_gva_type
                (gva.endpoint, gva.type, gva.lva, count, offset));
        
        // REVIEW: Should this be an assertion?
        // Check for an insertion failure. 
        if (HPX_UNLIKELY(!p.second))
        {
            HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                "insertion failed due to memory corruption or a locking "
                "error");
        }

        return true;
    } // }}}

    // REVIEW: Right return type, yes/no?
    range_type resolve_locality(endpoint_type const& ep) const
    { // {{{ resolve_endpoint implementation
        using boost::fusion::at_c;

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the GVA table 
        typename partition_table_type::map_type&
            partition_table = partitions_.get();

        typename partition_table_type::map_type::iterator
            it = partition_table.find(ep),
            end = partition_table.end(); 

        if (it != end)
        {
            return range_type(
                naming::get_gid_from_prefix(at_c<0>(it->second)),
                at_c<1>(it->second));
        }

        return range_type();
    } // }}}

    gva_type resolve_gid(naming::gid_type const& gid) const
    { // {{{ resolve_gid implementation 
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
            // Check for exact match
            if (it->first == id)
            {
                return gva_type(it->second.endpoint,
                                it->second.type,
                                it->second.lva);
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
                        HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                            "MSBs of lower and upper range bound do not match");
                    }
 
                    // Calculation of the local address occurs in the gva ctor
                    return gva_type(it->second, id, it->first);
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
                    HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                        "MSBs of lower and upper range bound do not match");
                }
 
                // Calculation of the local address occurs in the gva ctor
                return gva_type(it->second, id, it->first);
            }
        }

        return gva_type();
    } // }}}

    void unbind(naming::gid_type const& gid, count_type count)
    { // {{{ unbind implementation
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
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "block sizes must match");
            }

            gva_table.erase(it);
        }       
    } // }}}

    count_type increment(naming::gid_type const& gid, count_type credits)
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

        // See if this is the first increment request for this GID
        if (it == end)
        {
            std::pair<typename refcnt_table_type::map_type::iterator, bool>
                p = refcnt_table.insert(id, HPX_INITIAL_GLOBALCREDIT);

            if (HPX_UNLIKELY(!p.second))
            {
                HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                    "insertion failed due to memory corruption or a locking "
                    "error");
            }

            it = p.first;
        }
       
        // Add the requested amount and return the new total 
        return (it->second += credits);
    } // }}}
    
    decrement_result_type 
    decrement(naming::gid_type const& gid, count_type credits)
    { // {{{ decrement implementation
        // TODO: Implement and use a non-mutating version of
        // strip_credit_from_gid()
        naming::gid_type id = gid;
        naming::strip_credit_from_gid(id); 

        if (HPX_UNLIKELY(credits <= HPX_INITIAL_GLOBALCREDIT))
        {
            HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                "cannot decrement more than "
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
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "bogus credit encountered while decrement global reference "
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
                        return decrement_result_type(cnt,
                            reinterpret_cast<components::component_type>
                                (git->second.type));

                    // Check if we can safely decrement the iterator.
                    else if (git != gbegin)
                    {
                        --git;

                        // See if the previous range covers this GID
                        if ((git->first + git->second.count) > id)
                        {
                            // Make sure that the msbs match
                            if (id.get_msb() == git->first.get_msb())
                                return decrement_result_type(cnt,
                                    reinterpret_cast<components::component_type>
                                        (git->second.type));
                        } 
                    }
                }
                
                else if (HPX_LIKELY(!gva_table.empty()))
                {
                    --git;

                    // See if the previous range covers this GID
                    if ((git->first + git->second.count) > id)
                    {
                        // Make sure that the msbs match
                        if (id.get_msb() == git->first.get_msb())
                            return decrement_result_type(cnt,
                                reinterpret_cast<components::component_type>
                                    (git->second.type));
                    } 
                }

                // If we didn't find anything, we've got a problem
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "unknown component type encountered while decrement global "
                    "reference count");
            }

            else
                return decrement_result_type(cnt, components::component_invalid);
        }
        
        // We need to insert a new reference count entry. We assume that
        // binding has already created a first reference + credits.
        else 
        {
            BOOST_ASSERT(credits < HPX_INITIAL_GLOBALCREDIT);

            std::pair<typename refcnt_table_type::map_type::iterator, bool>
                p = refcnt_table.insert(id, HPX_INITIAL_GLOBALCREDIT);

            if (HPX_UNLIKELY(!p.second))
            {
                HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                    "insertion failed due to memory corruption or a locking "
                    "error");
            }
            
            it = p.first;
            
            if (HPX_UNLIKELY(it->second < credits))
            {
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "bogus credit encountered while decrement global reference "
                    "count");
            }
            
            count_type cnt = (it->second -= credits);
           
            BOOST_ASSERT(0 != cnt);
 
            return decrement_result_type(cnt, components::component_invalid);
        }   
    } // }}}
 
    // {{{ action types
    enum actions 
    {
        namespace_bind_locality,
        namespace_bind_gid,
        namespace_resolve_locality,
        namespace_resolve_gid,
        namespace_unbind,
        namespace_increment,
        namespace_decrement
    };

    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ range_type,
        /* enum value */  namespace_bind_locality,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace<Database, Protocol>::bind_locality
    > bind_locality_action; 
   
    typedef hpx::actions::result_action4<
        primary_namespace<Database, Protocol>,
        /* return type */ bool,
        /* enum value */  namespace_bind_gid,
        /* arguments */   naming::gid_type const&, gva_type const&, count_type,
                          offset_type,
        &primary_namespace<Database, Protocol>::bind_gid
    > bind_gid_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ range_type,
        /* enum value */  namespace_resolve_locality,
        /* arguments */   endpoint_type const&,
        &primary_namespace<Database, Protocol>::resolve_locality
    > resolve_locality_action;

    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ gva_type,
        /* enum value */  namespace_resolve_gid,
        /* arguments */   naming::gid_type const&,
        &primary_namespace<Database, Protocol>::resolve_gid
    > resolve_gid_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ void, 
        /* enum value */  namespace_unbind,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::unbind
    > unbind_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ count_type,  
        /* enum value */  namespace_increment,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::increment
    > increment_action;
    
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ decrement_result_type,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::decrement
    > decrement_action;
    // }}}
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

