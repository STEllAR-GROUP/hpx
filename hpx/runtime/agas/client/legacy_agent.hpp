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
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct legacy_agent
{
    typedef primary_namespace<Database, tag::network::tcpip>
        primary_namespace_type;

    typedef component_namespace<Database> component_namespace_type;
    typedef symbol_namespace<Database> symbol_namespace_type;

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

    // {{{ get_prefix specification
    /// \brief Get unique prefix usable as locality id (locality prefix)
    ///
    /// Every locality needs to have an unique locality id, which may be 
    /// used to issue unique global ids without having to consult the AGAS
    /// server for every id to generate.
    /// 
    /// \param l          [in] The locality the locality id needs to be 
    ///                   generated for. Repeating calls using the same 
    ///                   locality results in identical prefix values.
    /// \param prefix     [out] The generated prefix value uniquely 
    ///                   identifying the given locality. This is valid 
    ///                   only, if the return value of this function is 
    ///                   true.
    /// \param self       This parameter is \a true if the request is issued
    ///                   to assign a prefix to this site, and it is \a false
    ///                   if the command should return the prefix
    ///                   for the given location.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a new prefix has 
    ///                   been generated (it has been called for the first 
    ///                   time for the given locality) and returns \a false 
    ///                   if this locality already got a prefix assigned in 
    ///                   an earlier call. Any error results in an exception 
    ///                   thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_prefix(naming::locality const& l, naming::gid_type& prefix,
                    bool self = true, error_code& ec = throws) const
    {
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr;
        addr.from_string(l.get_address());

        primary_namespace_type::endpoint_type ep(addr, l.get_port()); 

        if (self)
        {
            prefix = naming::get_gid_from_prefix
                (naming::get_prefix_from_gid(at_c<0>(symbol_ns_.bind(ep))));
            return prefix;
        }
        
        else 
        {
            prefix = at_c<0>(symbol_ns_.resolve(ep)); 
            return prefix;
        }
    } 

    // {{{ get_console_prefix
    /// \brief Get locality prefix of the console locality
    ///
    /// \param prefix     [out] The prefix value uniquely identifying the
    ///                   console locality. This is valid only, if the 
    ///                   return value of this function is true.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a console prefix 
    ///                   exists and returns \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_console_prefix(naming::gid_type& prefix,
                            error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ get_prefixes specification
    /// \brief Query for the prefixes of all known localities.
    ///
    /// This function returns the prefixes of all localities known to the 
    /// AGAS server or all localities having a registered factory for a 
    /// given component type.
    /// 
    /// \param prefixes   [out] The vector will contain the prefixes of all
    ///                   localities registered with the AGAS server. The
    ///                   returned vector holds the prefixes representing 
    ///                   the runtime_support components of these 
    ///                   localities.
    /// \param type       [in] The component type will be used to determine
    ///                   the set of prefixes having a registered factory
    ///                   for this component. The default value for this 
    ///                   parameter is \a components#component_invalid, 
    ///                   which will return prefixes of all localities.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      components::component_type type,
                      error_code& ec = throws) const 
    { /* IMPLEMENT */ } 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      error_code& ec = throws) const
    { return get_prefixes(prefixes, components::component_invalid, ec); }

    // {{{ get_component_id specification
    /// \brief Return a unique id usable as a component type.
    /// 
    /// This function returns the component type id associated with the 
    /// given component name. If this is the first request for this 
    /// component name a new unique id will be created
    ///
    /// \param name       [in] The component name (string) to get the 
    ///                   component type for.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The function returns the currently associated 
    ///                   component type. Any error results in an 
    ///                   exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    components::component_type get_component_id(std::string const& name, 
                                                error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ register_factory specification
    /// \brief Register a factory for a specific component type
    ///
    /// This function allows to register a component factory for a given
    /// locality and component type.
    ///
    /// \param prefix     [in] The prefix value uniquely identifying the 
    ///                   given locality the factory needs to be registered 
    ///                   for. 
    /// \param name       [in] The component name (string) to register a
    ///                   factory for the given component type for.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The function returns the currently associated 
    ///                   component type. Any error results in an 
    ///                   exception thrown from this function. The returned
    ///                   component type is the same as if the function
    ///                   \a get_component_id was called using the same 
    ///                   component name.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    components::component_type register_factory(naming::gid_type const& prefix,
                                                std::string const& name, 
                                                error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ get_id_range specification
    /// \brief Get unique range of freely assignable global ids 
    ///
    /// Every locality needs to be able to assign global ids to different
    /// components without having to consult the AGAS server for every id 
    /// to generate. This function can be called to preallocate a range of
    /// ids usable for this purpose.
    /// 
    /// \param l          [in] The locality the locality id needs to be 
    ///                   generated for. Repeating calls using the same 
    ///                   locality results in identical prefix values.
    /// \param count      [in] The number of global ids to be generated.
    /// \param lower_bound 
    ///                   [out] The lower bound of the assigned id range.
    ///                   The returned value can be used as the first id
    ///                   to assign. This is valid only, if the return 
    ///                   value of this function is true.
    /// \param upper_bound
    ///                   [out] The upper bound of the assigned id range.
    ///                   The returned value can be used as the last id
    ///                   to assign. This is valid only, if the return 
    ///                   value of this function is true.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a new range has 
    ///                   been generated (it has been called for the first 
    ///                   time for the given locality) and returns \a false 
    ///                   if this locality already got a range assigned in 
    ///                   an earlier call. Any error results in an exception 
    ///                   thrown from this function.
    ///
    /// \note             This function assigns a range of global ids usable
    ///                   by the given locality for newly created components.
    ///                   Any of the returned global ids still has to be 
    ///                   bound to a local address, either by calling 
    ///                   \a bind or \a bind_range.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_id_range(naming::locality const& l, boost::uint32_t count, 
                      naming::gid_type& lower_bound,
                      naming::gid_type& upper_bound, 
                      error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ bind specification
    /// \brief Bind a global address to a local address.
    ///
    /// Every element in the ParalleX namespace has a unique global address
    /// (global id). This global id is generated by the function 
    /// px_core::get_next_component_id(). This global address has to be 
    /// associated with a concrete local address to be able to address an
    /// instance of a component using it's global address.
    ///
    /// \param id         [in] The global address which has to be bound to 
    ///                   the local address.
    /// \param addr       [in] The local address to be bound to the global 
    ///                   address.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          This function returns \a true, if this global id 
    ///                   got associated with an local address for the 
    ///                   first time. It returns \a false, if the global id 
    ///                   was associated with another local address earlier 
    ///                   and the given local address replaced the 
    ///                   previously associated local address. Any error 
    ///                   results in an exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             Binding a gid to a local address sets its global
    ///                   reference count to one.
    // }}}
    bool bind(naming::gid_type const& id, naming::address const& addr,
              error_code& ec = throws) const
    { return bind_range(id, 1, addr, 0, ec); }

    // {{{ bind_range specification
    /// \brief Bind unique range of global ids to given base address
    ///
    /// Every locality needs to be able to bind global ids to different
    /// components without having to consult the AGAS server for every id 
    /// to bind. This function can be called to bind a range of consecutive 
    /// global ids to a range of consecutive local addresses (separated by 
    /// a given \a offset).
    /// 
    /// \param lower_id   [in] The lower bound of the assigned id range.
    ///                   The value can be used as the first id to assign. 
    /// \param count      [in] The number of consecutive global ids to bind
    ///                   starting at \a lower_id.
    /// \param baseaddr   [in] The local address to bind to the global id
    ///                   given by \a lower_id. This is the base address 
    ///                   for all additional local addresses to bind to the
    ///                   remaining global ids.
    /// \param offset     [in] The offset to use to calculate the local
    ///                   addresses to be bound to the range of global ids.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if the given range 
    ///                   has been successfully bound and returns \a false 
    ///                   otherwise. Any error results in an exception 
    ///                   thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             Binding a gid to a local address sets its global
    ///                   reference count to one.
    // }}}
    bool bind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                    naming::address const& baseaddr, std::ptrdiff_t offset, 
                    error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ incref specification
    /// \brief Increment the global reference count for the given id
    ///
    /// \param id         [in] The global address (id) for which the 
    ///                   global reference count has to be incremented.
    /// \param credits    [in] The number of reference counts to add for
    ///                   the given id.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The global reference count after the increment. 
    // }}}
    boost::uint32_t incref(naming::gid_type const& id,
                           boost::uint32_t credits = 1, 
                           error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ decref specification
    /// \brief Decrement the global reference count for the given id
    ///
    /// \param id         [in] The global address (id) for which the 
    ///                   global reference count has to be decremented.
    /// \param t          [out] If this was the last outstanding global 
    ///                   reference for the given gid (the return value of 
    ///                   this function is zero), t will be set to the
    ///                   component type of the corresponding element.
    ///                   Otherwise t will not be modified.
    /// \param credits    [in] The number of reference counts to add for
    ///                   the given id.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// 
    /// \returns          The global reference count after the decrement. 
    // }}}
    boost::uint32_t decref(naming::gid_type const& id, 
                           components::component_type& t,
                           boost::uint32_t credits = 1,
                           error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ unbind specification
    /// \brief Unbind a global address
    ///
    /// Remove the association of the given global address with any local 
    /// address, which was bound to this global address. Additionally it 
    /// returns the local address which was bound at the time of this call.
    /// 
    /// \param id         [in] The global address (id) for which the 
    ///                   association has to be removed.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          The function returns \a true if the association 
    ///                   has been removed, and it returns \a false if no 
    ///                   association existed. Any error results in an 
    ///                   exception thrown from this function.
    ///
    /// \note             You can unbind only global ids bound using the 
    ///                   function \a bind. Do not use this function to 
    ///                   unbind any of the global ids bound using 
    ///                   \a bind_range.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             This function will raise an error if the global 
    ///                   reference count of the given gid is not zero!
    // }}}
    bool unbind(naming::gid_type const& id, error_code& ec = throws) const
    { // {{{ unbind implementation
        naming::address addr; // ignore the return value
        return unbind_range(id, 1, addr, ec);
    } // }}}

    // {{{ unbind specification
    /// \brief Unbind a global address
    ///
    /// Remove the association of the given global address with any local 
    /// address, which was bound to this global address. Additionally it 
    /// returns the local address which was bound at the time of this call.
    /// 
    /// \param id         [in] The global address (id) for which the 
    ///                   association has to be removed.
    /// \param addr       [out] The local address which was associated with 
    ///                   the given global address (id).
    ///                   This is valid only if the return value of this 
    ///                   function is true.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          The function returns \a true if the association 
    ///                   has been removed, and it returns \a false if no 
    ///                   association existed. Any error results in an 
    ///                   exception thrown from this function.
    ///
    /// \note             You can unbind only global ids bound using the 
    ///                   function \a bind. Do not use this function to 
    ///                   unbind any of the global ids bound using 
    ///                   \a bind_range.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             This function will raise an error if the global 
    ///                   reference count of the given gid is not zero!
    // }}}
    bool unbind(naming::gid_type const& id, naming::address& addr, 
                error_code& ec = throws) const
    { return unbind_range(id, 1, addr, ec); }

    // {{{ unbind_range specification
    /// \brief Unbind the given range of global ids
    ///
    /// \param lower_id   [in] The lower bound of the assigned id range.
    ///                   The value must the first id of the range as 
    ///                   specified to the corresponding call to 
    ///                   \a bind_range. 
    /// \param count      [in] The number of consecutive global ids to unbind
    ///                   starting at \a lower_id. This number must be 
    ///                   identical to the number of global ids bound by 
    ///                   the corresponding call to \a bind_range
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a new range has 
    ///                   been generated (it has been called for the first 
    ///                   time for the given locality) and returns \a false 
    ///                   if this locality already got a range assigned in 
    ///                   an earlier call. Any error results in an exception 
    ///                   thrown from this function.
    ///
    /// \note             You can unbind only global ids bound using the 
    ///                   function \a bind_range. Do not use this function 
    ///                   to unbind any of the global ids bound using 
    ///                   \a bind.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             This function will raise an error if the global 
    ///                   reference count of the given gid is not zero!
    // }}}
    bool unbind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                      error_code& ec = throws) const
    { // {{{ unbind_range implementation
        naming::address addr; // ignore the return value
        return unbind_range(lower_id, count, addr, ec);
    } // }}}

    // {{{ unbind_range specification
    /// \brief Unbind the given range of global ids
    ///
    /// \param lower_id   [in] The lower bound of the assigned id range.
    ///                   The value must the first id of the range as 
    ///                   specified to the corresponding call to 
    ///                   \a bind_range. 
    /// \param count      [in] The number of consecutive global ids to unbind
    ///                   starting at \a lower_id. This number must be 
    ///                   identical to the number of global ids bound by 
    ///                   the corresponding call to \a bind_range
    /// \param addr       [out] The local address which was associated with 
    ///                   the given global address (id).
    ///                   This is valid only if the return value of this 
    ///                   function is true.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a new range has 
    ///                   been generated (it has been called for the first 
    ///                   time for the given locality) and returns \a false 
    ///                   if this locality already got a range assigned in 
    ///                   an earlier call. 
    ///
    /// \note             You can unbind only global ids bound using the 
    ///                   function \a bind_range. Do not use this function 
    ///                   to unbind any of the global ids bound using 
    ///                   \a bind.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    /// 
    /// \note             This function will raise an error if the global 
    ///                   reference count of the given gid is not zero!
    // }}}
    bool unbind_range(naming::gid_type const& lower_id, boost::uint32_t count, 
                      naming::address& addr, error_code& ec = throws) const
    { /* IMPLEMENT */ } 

    // {{{ resolve specification
    /// \brief Resolve a given global address (id) to its associated local 
    ///        address
    ///
    /// This function returns the local address which is currently 
    /// associated with the given global address (id).
    ///
    /// \param id         [in] The global address (id) for which the 
    ///                   associated local address should be returned.
    /// \param addr       [out] The local address which currently is 
    ///                   associated with the given global address (id), 
    ///                   this is valid only if the return value of this 
    ///                   function is true.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if the global 
    ///                   address has been resolved successfully (there 
    ///                   exists an association to a local address) and the 
    ///                   associated local address has been returned. The 
    ///                   function returns \a false if no association exists 
    ///                   for the given global address. Any error results 
    ///                   in an exception thrown from this function.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool resolve(naming::gid_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) const
    { /* IMPLEMENT */ }

    bool resolve(naming::id_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) const
    { return resolve(id.get_gid(), addr, try_cache, ec); }

    bool resolve_cached(naming::gid_type const& id, naming::address& addr, 
                        error_code& ec = throws) const
    { /* IMPLEMENT */ }

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
                    error_code& ec = throws) const
    { /* IMPLEMENT */ }

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
    bool unregisterid(std::string const& name, error_code& ec = throws) const
    { /* IMPLEMENT */ }

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
                 error_code& ec = throws) const
    { /* IMPLEMENT */ }

    // {{{ get_statistics_count specification
    /// \brief Query for the gathered statistics of this AGAS instance 
    ///        (server execution count)
    ///
    /// This function returns the execution counts for each of the 
    /// commands
    /// 
    /// \param counts     [out] The vector will contain the server 
    ///                   execution counts, one entry for each of the 
    ///                   possible resolver_client commands (i.e. will be 
    ///                   of the size 'server::command_lastcommand').
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_statistics_count(std::vector<std::size_t>& counts,
                              error_code& ec = throws) const
    { /* IMPLEMENT */ }

    // {{{ get_statistics_mean specification
    /// \brief Query for the gathered statistics of this AGAS instance 
    ///        (average server execution time)
    ///
    /// This function returns the average timings for each of the commands
    /// 
    /// \param timings    [out] The vector will contain the average server 
    ///                   execution times, one entry for each of the 
    ///                   possible resolver_client commands (i.e. will be 
    ///                   of the size 'server::command_lastcommand').
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_statistics_mean(std::vector<double>& timings,
                             error_code& ec = throws) const
    { /* IMPLEMENT */ }

    // {{{ get_statistics_moment2 specification
    /// \brief Query for the gathered statistics of this AGAS instance 
    ///        (statistical 2nd moment of server execution time)
    ///
    /// This function returns the 2nd moment for the timings for each of 
    /// the commands
    /// 
    /// \param moments    [out] The vector will contain the 2nd moment of 
    ///                   the server execution times, one entry for each of 
    ///                   the possible resolver_client commands (i.e. will 
    ///                   be of the size 'server::command_lastcommand').
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to 
    ///                   \a hpx#throws this function doesn't 
    ///                   throw but returns the result code using the 
    ///                   parameter \a ec. Otherwise it throws and instance
    ///                   of hpx#exception.
    // }}}
    bool get_statistics_moment2(std::vector<double>& moments,
                                error_code& ec = throws) const
    { /* IMPLEMENT */ }

    /// \brief Return the locality of the resolver server this resolver 
    ///        client instance is using to serve requests
    naming::locality const& there() const { /* IMPLEMENT */ }

    /// \brief Returns whether this resolver_client represents the console 
    ///        locality
    bool is_console() const { /* IMPLEMENT */ }

    /// \brief Returns whether this resolver_client runs in SMP mode
    bool is_smp_mode() const { /* IMPLEMENT */ }
};

}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

