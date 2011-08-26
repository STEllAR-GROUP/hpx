//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_BULK_CLIENT_RESOLVER_JAN_21_2009_1005AM)
#define HPX_NAMING_BULK_CLIENT_RESOLVER_JAN_21_2009_1005AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/runtime/naming/resolver_client.hpp>

#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    /// \class bulk_resolver_client resolver_client.hpp hpx/runtime/naming/bulk_resolver_client.hpp
    ///
    /// The top-level class of the AGAS client. This class exposes the bulk 
    /// related AGAS server functionality on the client side.
    class HPX_EXPORT bulk_resolver_client
    {
    public:
        /// Construct the resolver client to work with the server given by
        /// a locality
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 outgoing requests
        /// \param l        [in] This is the locality the AGAS server is 
        ///                 running on.
        /// \param isconsole [in] This parameter is true if the locality 
        ///                 represents the application console.
        bulk_resolver_client(resolver_client& resolver);

        /// \brief Cache a request for getting unique prefix usable as 
        ///        locality id (locality prefix)
        ///
        /// Every locality needs to have an unique locality id, which may be 
        /// used to issue unique global ids without having to consult the AGAS
        /// server for every id to generate.
        /// 
        /// \param l          [in] The locality the locality id needs to be 
        ///                   generated for. Repeating calls using the same 
        ///                   locality results in identical prefix values.
        /// \param self       This parameter is \a true if the request is issued
        ///                   to assign a prefix to this site, and it is \a false
        ///                   if the command should return the prefix
        ///                   for the given location.
        ///
        /// \returns          This function returns an index into the array of 
        ///                   accumulated requests allowing to reference to the
        ///                   corresponding result later.
        ///
        /// \note             As long as \a ec is not pre-initialized to 
        ///                   \a hpx#throws this function doesn't 
        ///                   throw but returns the result code using the 
        ///                   parameter \a ec. Otherwise it throws and instance
        ///                   of hpx#exception.
        int get_prefix(locality const& l, bool self = true);

        /// \brief Get unique prefix usable as locality id (locality prefix)
        ///
        /// \param index      [in] The index of the operation as returned by
        ///                   the corresponding call to \a get_prefix.
        /// \param prefix     [out] The generated prefix value uniquely 
        ///                   identifying the given locality. This is valid 
        ///                   only, if the return value of this function is 
        ///                   true.
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        /// \returns          This function returns \a true if a new prefix has 
        ///                   been generated (it has been called for the first 
        ///                   time for the given locality) and returns \a false 
        ///                   if this locality already got a prefix assigned in 
        ///                   an earlier call. Any error results in an exception 
        ///                   thrown from this function.
        ///
        /// \note             This function may be called only after the cached
        ///                   requests have been executed.
        bool get_prefix(std::size_t index, gid_type& prefix, error_code& ec = throws) const;

        /// \brief Cache a request for incrementing the global reference count 
        ///        for the given id
        ///
        /// \param id         [in] The global address (id) for which the 
        ///                   global reference count has to be incremented.
        /// \param credits    [in] The number of reference counts to add for
        ///                   the given id.
        /// 
        /// \returns          This function returns an index into the array of 
        ///                   accumulated requests allowing to reference to the
        ///                   corresponding result later.
        int incref(gid_type const& id, boost::uint32_t credits = 1);

        /// \brief Increment the global reference count for the given id
        ///
        /// \param index      [in] The index of the operation as returned by
        ///                   the corresponding call to \a get_prefix.
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        /// 
        /// \returns          The global reference count after the increment. 
        int incref(std::size_t index, error_code& ec = throws) const;

        /// \brief Cache request to resolve a given global address (id) to its 
        ///        associated local address
        ///
        /// This function returns the local address which is currently 
        /// associated with the given global address (id).
        ///
        /// \param id         [in] The global address (id) for which the 
        ///                   associated local address should be returned.
        ///
        /// \returns          This function returns an index into the array of 
        ///                   accumulated requests allowing to reference to the
        ///                   corresponding result later.
        int resolve(gid_type const& id);
        int resolve(id_type const& id)
        {
            return resolve(id.get_gid());
        }

        /// \brief Resolve a given global address (id) to its associated local 
        ///        address
        ///
        /// This function returns the local address which is currently 
        /// associated with the given global address (id).
        ///
        /// \param index      [in] The index of the operation as returned by
        ///                   the corresponding call to \a get_prefix.
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
        bool resolve(std::size_t index, address& addr, error_code& ec = throws) const;

        /// \brief Execute the accumulated requests
        bool execute(error_code& ec);

    private:
        resolver_client& resolver_;

        std::vector<server::request> requests_;
        std::vector<server::reply> responses_;
    };

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
