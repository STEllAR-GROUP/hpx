////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_15D904C7_CD18_46E1_A54A_65059966A34F)
#define HPX_15D904C7_CD18_46E1_A54A_65059966A34F

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>

namespace hpx { namespace agas 
{

// TODO: pass error codes once they're implemented in AGAS.
template <typename Database>
struct legacy_agent
{
    typedef primary_namespace<Database, tag::network::tcpip>
        primary_namespace_type;

    typedef component_namespace<Database> component_namespace_type;
    typedef symbol_namespace<Database> symbol_namespace_type;

    typedef typename component_namespace_type::component_id_type
        component_id_type;
  private:
    primary_namespace_type primary_ns_;
    component_namespace_type component_ns_;
    symbol_namespace_type symbol_ns_;

  public:
     explicit legacy_agent(naming::id_type const& primary_ns,
                           naming::id_type const& component_ns,
                           naming::id_type const& symbol_ns) :
        primary_ns_(primary_ns),
        component_ns_(component_ns),
        symbol_ns_(symbol_ns) {} 

    bool get_prefix(naming::locality const& l, naming::gid_type& prefix,
                    bool self = true, error_code& ec = throws) 
    {
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(l.get_address());

        typename primary_namespace_type::endpoint_type ep(addr, l.get_port()); 

        if (self)
        {
            typename primary_namespace_type::binding_type r
                = primary_ns_.bind(ep, 0);
            prefix = at_c<2>(r);
            return at_c<3>(r);
        }
        
        else 
        {
            prefix = at_c<0>(primary_ns_.resolve(ep)); 
            return false;
        }
    } 

    bool get_console_prefix(naming::gid_type& prefix,
                            error_code& ec = throws) 
    {
        prefix = symbol_ns_.resolve("/console");
        return prefix;
    } 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      component_id_type type, error_code& ec = throws) 
    {
        if (type != components::component_invalid)
        {
            typedef typename component_namespace_type::prefixes_type::
                const_iterator iterator;
    
            typename component_namespace_type::prefixes_type raw_prefixes
                = component_ns_.resolve(type);
    
            if (raw_prefixes.empty())
                return false;
    
            iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
            for (; it != end; ++it) 
                prefixes.push_back(naming::get_gid_from_prefix(*it));
    
            return true; 
        }

        typedef typename primary_namespace_type::prefixes_type::
            const_iterator iterator;
    
        typename primary_namespace_type::prefixes_type raw_prefixes
            = primary_ns_.localities();
    
        if (raw_prefixes.empty())
            return false;
    
        iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
        for (; it != end; ++it) 
            prefixes.push_back(naming::get_gid_from_prefix(*it));
    
        return true; 
    } 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      error_code& ec = throws) 
    { return get_prefixes(prefixes, components::component_invalid, ec); }

    typename component_namespace_type::component_id_type
    get_component_id(std::string const& name, error_code& ec = throws) 
    { return component_ns_.bind(name); } 

    typename component_namespace_type::component_id_type
    register_factory(naming::gid_type const& prefix, std::string const& name, 
                     error_code& ec = throws) 
    { return component_ns_.bind(name, naming::get_prefix_from_gid(prefix)); } 

    bool get_id_range(naming::locality const& l, boost::uint32_t count, 
                      naming::gid_type& lower_bound,
                      naming::gid_type& upper_bound, 
                      error_code& ec = throws) 
    {
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(l.get_address());

        typename primary_namespace_type::endpoint_type ep(addr, l.get_port()); 
         
        typename primary_namespace_type::binding_type range =
            primary_ns_.bind(ep, count);

        lower_bound = at_c<0>(range);
        upper_bound = at_c<1>(range);

        return lower_bound && upper_bound;
    } 

    bool bind(naming::gid_type const& id, naming::address const& addr,
              error_code& ec = throws) 
    { return bind_range(id, 1, addr, 0, ec); }

    bool bind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                    naming::address const& baseaddr, std::ptrdiff_t offset, 
                    error_code& ec = throws) 
    {
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(baseaddr.locality_.get_address());

        typename primary_namespace_type::endpoint_type ep
            (addr, baseaddr.locality_.get_port()); 
       
        // Create a global virtual address from the legacy calling convention
        // parameters.
        typename primary_namespace_type::gva_type gva
            (ep, baseaddr.type_, count, baseaddr.address_, offset);

        return primary_ns_.bind(lower_id, gva); 
    } 

    typename primary_namespace_type::count_type
    incref(naming::gid_type const& id, boost::uint32_t credits = 1, 
           error_code& ec = throws) 
    { return primary_ns_.increment(id, credits); } 

    typename primary_namespace_type::count_type
    decref(naming::gid_type const& id, component_id_type& t,
           boost::uint32_t credits = 1, error_code& ec = throws) 
    {
        using boost::fusion::at_c;

        typename primary_namespace_type::decrement_type r
            = primary_ns_.increment(id, credits);

        if (at_c<0>(r) == 0)
            t = at_c<1>(r);

        return at_c<0>(r);
    }

    bool unbind(naming::gid_type const& id, error_code& ec = throws) 
    {  return unbind_range(id, 1, ec); } 
        
    bool unbind(naming::gid_type const& id, naming::address& addr,
                error_code& ec = throws) 
    { return unbind_range(id, 1, addr, ec); }

    bool unbind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                      error_code& ec = throws) 
    {
        naming::address addr; 
        return unbind_range(lower_id, count, addr, ec);
    } 

    bool unbind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                      naming::address& addr, error_code& ec = throws) 
    {
        typename primary_namespace_type::unbinding_type r
            = primary_ns_.unbind(lower_id, count);

        if (r)
        {
            addr.locality_ = r->endpoint;
            addr.type_ = r->type;
            addr.address_ = r->lva();
        }

        return r; 
    }

    bool resolve(naming::gid_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    {
        typename primary_namespace_type::gva_type gva = primary_ns_.resolve(id);

        addr.locality_ = gva.endpoint;
        addr.type_ = gva.type;
        addr.address_ = gva.lva();

        typedef typename primary_namespace_type::endpoint_type endpoint_type;
        return (gva.endpoint != endpoint_type())
            && (gva.type != components::component_invalid)
            && (gva.lva() != 0);
    }

    bool resolve(naming::id_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { return resolve(id.get_gid(), addr, try_cache, ec); }

    bool resolve_cached(naming::gid_type const& id, naming::address& addr, 
                        error_code& ec = throws) 
    { return resolve(id, addr, true, ec); /* IMPLEMENT */ }

    // {{{ registerid specification
    /// \brief Register a global name with a global address (id)
    /// 
    /// This function registers an association between a global name 
    /// (string) and a global address (id) usable with one of the functions 
    /// above (bind, unbind, and resolve).
    ///
    /// \param name       [in] The global name (string) to be associated
    ///                   with the global address.
    /// \param id         [in] The global address (id) to be associated 
    ///                   with the global address.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The function returns \a true if the global name 
    ///                   got an association with a global address for the 
    ///                   first time, and it returns \a false if this 
    ///                   function call replaced a previously registered 
    ///                   global address with the global address (id) 
    ///                   given as the parameter. Any error results in an 
    ///                   exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool registerid(std::string const& name, naming::gid_type const& id,
                    error_code& ec = throws) 
    {
        naming::gid_type r = symbol_ns_.rebind(name, id);
        return r == id;
    }

    // {{{ unregisterid specification
    /// \brief Unregister a global name (release any existing association)
    ///
    /// This function releases any existing association of the given global 
    /// name with a global address (id). 
    /// 
    /// \param name       [in] The global name (string) for which any 
    ///                   association with a global address (id) has to be 
    ///                   released.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The function returns \a true if an association of 
    ///                   this global name has been released, and it returns 
    ///                   \a false, if no association existed. Any error 
    ///                   results in an exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool unregisterid(std::string const& name, error_code& ec = throws) 
    { return symbol_ns_.unbind(name); }

    // {{{ queryid specification
    /// Query for the global address associated with a given global name.
    ///
    /// This function returns the global address associated with the given 
    /// global name.
    ///
    /// string name:      [in] The global name (string) for which the 
    ///                   currently associated global address has to be 
    ///                   retrieved.
    /// id_type& id:      [out] The id currently associated with the given 
    ///                   global name (valid only if the return value is 
    ///                   true).
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// This function returns true if it returned global address (id), 
    /// which is currently associated with the given global name, and it 
    /// returns false, if currently there is no association for this global 
    /// name. Any error results in an exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool queryid(std::string const& ns_name, naming::gid_type& id,
                 error_code& ec = throws) 
    {
        id = symbol_ns_.resolve(ns_name);
        return id;         
    }
};

}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

