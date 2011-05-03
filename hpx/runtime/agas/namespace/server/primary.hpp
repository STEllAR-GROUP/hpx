////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21)
#define HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

#include <boost/optional.hpp>
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
    typedef typename gva_type::count_type count_type;
    typedef typename gva_type::offset_type offset_type;
    typedef boost::uint32_t component_type;
    typedef boost::uint32_t prefix_type;
    typedef std::vector<prefix_type> prefixes_type;

    typedef boost::fusion::vector2<count_type, component_type>
        decrement_type;

    typedef boost::fusion::vector2<prefix_type, naming::gid_type>
        partition_type;

    typedef boost::fusion::vector4<
        naming::gid_type, naming::gid_type, naming::gid_type, bool
    > binding_type;

    typedef boost::optional<gva_type> unbinding_type;

    typedef boost::fusion::vector2<naming::gid_type, gva_type>
        locality_type;

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
  
  public:
    primary_namespace()
      : mutex_(),
        gvas_("hpx.agas.primary_namespace.gva"),
        partitions_("hpx.agas.primary_namespace.partition"),
        refcnts_("hpx.agas.primary_namespace.refcnt")
    { traits::initialize_mutex(mutex_); }

    binding_type bind_locality(endpoint_type const& ep, count_type count)
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
              return binding_type(at_c<1>(it->second), at_c<1>(it->second),
                  naming::get_gid_from_prefix(at_c<0>(it->second)), false);

            // Compute the new allocation.
            naming::gid_type lower(at_c<1>(it->second) + 1),
                             upper(lower + real_count);

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
                upper = lower + real_count; 
            }
            
            // Store the new upper bound.
            at_c<1>(it->second) = upper;

            // Set the initial credit count.
            naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
            naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT); 

            return binding_type(lower, upper,
                naming::get_gid_from_prefix(at_c<0>(it->second)), false);
        }

        // If the endpoint isn't in the table, then we're registering it.
        else
        {
            // Check for address space exhaustion.
            if (HPX_UNLIKELY(partition_table.size() > 0xFFFFFFFE))
            {
                HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                    "primary namespace has been exhausted");
            }

            // Compute the locality's prefix
            boost::uint32_t prefix = static_cast<boost::uint32_t>
                (partition_table.size() + 1);
            naming::gid_type id(naming::get_gid_from_prefix(prefix));

            // Start assigning ids with the second block of 64bit numbers only
            naming::gid_type lower_id(id.get_msb() + 1, 0);

            std::pair<typename partition_table_type::map_type::iterator, bool>
                pit = partition_table.insert(typename
                    partition_table_type::map_type::value_type
                        (ep, partition_type(prefix, lower_id)));

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
                git = gva_table.insert(typename
                    gva_table_type::map_type::value_type
                        (id, gva_type
                            (ep, components::component_runtime_support, count)));

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
            naming::gid_type lower = lower_id + 1;
            naming::gid_type upper = lower + real_count;

            at_c<1>((*pit.first).second) = upper;

            // Set the initial credit count.
            naming::set_credit_for_gid(lower, HPX_INITIAL_GLOBALCREDIT);
            naming::set_credit_for_gid(upper, HPX_INITIAL_GLOBALCREDIT); 

            return binding_type(lower, upper, id, true);
        }
    } // }}}

    bool bind_gid(naming::gid_type const& gid, gva_type const& gva)
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
                if (it->second.count != gva.count)
                {
                    // REVIEW: Is this the right error code to use?
                    HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                        "cannot change block size of existing binding");
                }

                // Store the new endpoint and offset
                it->second.endpoint = gva.endpoint;
                it->second.type = gva.type;
                it->second.lva(gva.lva());
                it->second.offset = gva.offset;
                return false;
            }

            // We're about to decrement it, so make sure it's safe to do so.
            else if (it != begin)
            {
                --it;

                // Check that a previous range doesn't cover the new id.
                if ((it->first + it->second.count) > id)
                {
                    // REVIEW: Is this the right error code to use?
                    HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                        "the new GID is contained in an existing range");
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
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "the new GID is contained in an existing range");
            }
        }

        naming::gid_type upper_bound(id + (gva.count - 1));

        if (HPX_UNLIKELY(id.get_msb() != upper_bound.get_msb()))
        {
            HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                "MSBs of lower and upper range bound do not match");
        }

        std::pair<typename gva_table_type::map_type::iterator, bool>
            p = gva_table.insert(typename gva_table_type::map_type::value_type
                (id, gva));
        
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

    locality_type resolve_locality(endpoint_type const& ep) 
    { // {{{ resolve_endpoint implementation
        using boost::fusion::at_c;

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the partition table 
        typename partition_table_type::map_type const&
            partition_table = partitions_.get();

        typename partition_table_type::map_type::const_iterator
            pit = partition_table.find(ep),
            pend = partition_table.end(); 

        if (pit != pend)
        {
            naming::gid_type id
                = naming::get_gid_from_prefix(at_c<0>(pit->second));
                
            // Load the GVA table 
            typename gva_table_type::map_type const& gva_table = gvas_.get();
    
            typename gva_table_type::map_type::const_iterator
                git = gva_table.lower_bound(id),
                gbegin = gva_table.begin(),
                gend = gva_table.end();
    
            if (git != gend)
            {
                // Check for exact match
                if (git->first == id)
                    return locality_type(git->first, git->second);
    
                // We need to decrement the iterator, check that it's safe to do
                // so.
                else if (git != gbegin)
                {
                    --git;
    
                    // Found the GID in a range
                    if ((git->first + git->second.count) > id)
                    {
                        if (HPX_UNLIKELY(id.get_msb() != git->first.get_msb()))
                        {
                            HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                                "MSBs of lower and upper range bound do not match");
                        }
     
                        // Calculation of the lva address occurs in gva<>::resolve()
                        return locality_type
                            (git->first, git->second.resolve(id, git->first));
                    }
                }
            }
    
            else if (HPX_LIKELY(!gva_table.empty()))
            {
                --git;
    
                // Found the GID in a range
                if ((git->first + git->second.count) > id)
                {
                    if (HPX_UNLIKELY(id.get_msb() != git->first.get_msb()))
                    {
                        HPX_THROW_IN_CURRENT_FUNC(internal_server_error, 
                            "MSBs of lower and upper range bound do not match");
                    }
     
                    // Calculation of the local address occurs in gva<>::resolve()
                    return locality_type
                        (git->first, git->second.resolve(id, git->first));
                }
            }
        }

        return locality_type();
    } // }}}

    gva_type resolve_gid(naming::gid_type const& gid) 
    { // {{{ resolve_gid implementation 
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
                return it->second;

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
 
                    // Calculation of the lva address occurs in gva<>::resolve()
                    return it->second.resolve(id, it->first);
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
 
                // Calculation of the local address occurs in gva<>::resolve()
                return it->second.resolve(id, it->first);
            }
        }

        return gva_type();
    } // }}}

    unbinding_type unbind(naming::gid_type const& gid, count_type count)
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

            unbinding_type ep(it->second);
            gva_table.erase(it);
            return ep;
        }

        else
            return unbinding_type();       
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
                p = refcnt_table.insert(typename
                    refcnt_table_type::map_type::value_type
                        (id, HPX_INITIAL_GLOBALCREDIT));

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
    
    decrement_type 
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
                        return decrement_type(cnt, git->second.type);

                    // Check if we can safely decrement the iterator.
                    else if (git != gbegin)
                    {
                        --git;

                        // See if the previous range covers this GID
                        if ((git->first + git->second.count) > id)
                        {
                            // Make sure that the msbs match
                            if (id.get_msb() == git->first.get_msb())
                                return decrement_type
                                    (cnt, git->second.type);
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
                            return decrement_type(cnt, git->second.type);
                    } 
                }

                // If we didn't find anything, we've got a problem
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                    "unknown component type encountered while decrement global "
                    "reference count");
            }

            else
                return decrement_type(cnt, components::component_invalid);
        }
        
        // We need to insert a new reference count entry. We assume that
        // binding has already created a first reference + credits.
        BOOST_ASSERT(credits < HPX_INITIAL_GLOBALCREDIT);

        std::pair<typename refcnt_table_type::map_type::iterator, bool>
            p = refcnt_table.insert(typename
                refcnt_table_type::map_type::value_type
                    (id, HPX_INITIAL_GLOBALCREDIT));

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
 
        return decrement_type(cnt, components::component_invalid);
    } // }}}
 
    prefixes_type localities()
    { // {{{ localities implementation
        using boost::fusion::at_c;

        prefixes_type prefixes;

        typename database_mutex_type::scoped_lock l(mutex_);

        // Load the partition table 
        typename partition_table_type::map_type const&
            partition_table = partitions_.get();

        prefixes.reserve(partition_table.size());

        typename partition_table_type::map_type::const_iterator
            it = partition_table.begin(),
            end = partition_table.end(); 

        for (; it != end; ++it)
            prefixes.push_back(at_c<0>(it->second));

        return prefixes;
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
        namespace_decrement,
        namespace_localities
    };

    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ binding_type,
        /* enum value */  namespace_bind_locality,
        /* arguments */   endpoint_type const&, count_type,
        &primary_namespace<Database, Protocol>::bind_locality
    > bind_locality_action; 
   
    typedef hpx::actions::result_action2<
        primary_namespace<Database, Protocol>,
        /* return type */ bool,
        /* enum value */  namespace_bind_gid,
        /* arguments */   naming::gid_type const&, gva_type const&,
        &primary_namespace<Database, Protocol>::bind_gid
    > bind_gid_action;
    
    typedef hpx::actions::result_action1<
        primary_namespace<Database, Protocol>,
        /* return type */ locality_type,
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
        /* return type */ unbinding_type,
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
        /* return type */ decrement_type,
        /* enum value */  namespace_decrement,
        /* arguments */   naming::gid_type const&, count_type,
        &primary_namespace<Database, Protocol>::decrement
    > decrement_action;
    
    typedef hpx::actions::result_action0<
        primary_namespace<Database, Protocol>,
        /* return type */ prefixes_type,
        /* enum value */  namespace_localities,
        &primary_namespace<Database, Protocol>::localities
    > localities_action;
    // }}}
};

}}}

#endif // HPX_BDD56092_8F07_4D37_9987_37D20A1FEA21

