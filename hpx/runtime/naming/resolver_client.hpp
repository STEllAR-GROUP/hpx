//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_CLIENT_RESOLVER_MAR_24_2008_0952AM)
#define HPX_NAMING_CLIENT_RESOLVER_MAR_24_2008_0952AM

#include <boost/asio.hpp>

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/util/future.hpp>
#include <hpx/util/io_service_pool.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    namespace server
    {
        class reply;
        class request;
    }
        
    /// The top-level class of the DGAS client. This class exposes the DGAS 
    /// server functionality on the client side.
    class resolver_client
    {
    public:
        /// Construct the resolver client to work with the server given by
        /// a locality
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 outgoing requests
        /// \param l        [in] This is the locality the DGAS server is 
        ///                 running on.
        /// \param start_asynchronously 
        ///                 [in] This parameter allows to start 
        ///                 the resolver client instance immediately.
        resolver_client(util::io_service_pool& io_service_pool, locality l,
            bool start_asynchronously = true);
        
        /// Construct the resolver client to work with the server given by
        /// its address and port number.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 outgoing requests
        /// \param address  [in] This is the address (IP address or 
        ///                 host name) of the locality the DGAS server is 
        ///                 running on.
        /// \param port     [in] This is the port number the DGAS server
        ///                 is listening on.
        /// \param start_asynchronously 
        ///                 [in] This parameter allows to start 
        ///                 the resolver client instance immediately.
        resolver_client(util::io_service_pool& io_service_pool, 
            std::string const& address, unsigned short port,
            bool start_asynchronously = true);

        /// \brief Get unique prefix usable as locality id (locality prefix)
        ///
        /// Every locality needs to have an unique locality id, which may be 
        /// used to issue unique global ids without having to consult the PGAS
        /// server for every id to generate.
        /// 
        /// \param l          [in] The locality the locality id needs to be 
        ///                   generated for. Repeating calls using the same 
        ///                   locality results in identical prefix values.
        /// \param prefix     [out] The generated prefix value uniquely 
        ///                   identifying the given locality. This is valid 
        ///                   only, if the return value of this function is 
        ///                   true.
        ///
        /// \returns          This function returns \a true if a new prefix has 
        ///                   been generated (it has been called for the first 
        ///                   time for the given locality) and returns \a false 
        ///                   if this locality already got a prefix assigned in 
        ///                   an earlier call. Any error results in an exception 
        ///                   thrown from this function.
        bool get_prefix(locality const& l, id_type& prefix) const;

        /// \brief Get unique range of freely assignable global ids 
        ///
        /// Every locality needs to be able to assign global ids to different
        /// components without having to consult the DGAS server for every id 
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
        bool get_id_range(locality const& l, std::size_t count, 
            id_type& lower_bound, id_type& upper_bound) const;
        
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
        /// 
        /// \returns          This function returns \a true, if this global id 
        ///                   got associated with an local address for the 
        ///                   first time. It returns \a false, if the global id 
        ///                   was associated with another local address earlier 
        ///                   and the given local address replaced the 
        ///                   previously associated local address. Any error 
        ///                   results in an exception thrown from this function.
        bool bind(id_type id, address const& addr) const
        {
            return bind_range(id, 1, addr, 0);
        }
        
        /// \brief Bind unique range of global ids to given base address
        ///
        /// Every locality needs to be able to bind global ids to different
        /// components without having to consult the DGAS server for every id 
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
        ///
        /// \returns          This function returns \a true if the given range 
        ///                   has been successfully bound and returns \a false 
        ///                   otherwise. Any error results in an exception 
        ///                   thrown from this function.
        bool bind_range(id_type lower_id, std::size_t count, 
            address const& baseaddr, std::ptrdiff_t offset) const;

        /// \brief Asynchronously bind unique range of global ids to given base 
        ///        address
        ///
        /// Every locality needs to be able to bind global ids to different
        /// components without having to consult the DGAS server for every id 
        /// to bind. This function can be called to asynchronously bind a range 
        /// of consecutive global ids to a range of consecutive local addresses 
        /// (separated by a given \a offset).
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
        ///
        /// \returns          This function returns a future object allowing to
        ///                   defer the evaluation of the outcome of the 
        ///                   function. The actual return value can be retrieved
        ///                   by calling f.get(), where f is the returned future 
        ///                   instance, and f.get() returns \a true the given
        ///                   range has been successfully bound and returns 
        ///                   \a false otherwise. Any error results in an 
        ///                   exception thrown from f.get().
        ///
        /// \note             The difference to \a bind_range is that the 
        ///                   function call returns immediately without 
        ///                   blocking. The operation is guaranteed to be 
        ///                   fully executed only after f.get() has been called.
        util::unique_future<bool> 
            bind_range_async(id_type lower_id, std::size_t count, 
                address const& baseaddr, std::ptrdiff_t offset);
            
        /// \brief Unbind a global address
        ///
        /// Remove the association of the given global address with any local 
        /// address, which was bound to this global address. Additionally it 
        /// returns the local address which was bound at the time of this call.
        /// 
        /// \param id         [in] The global address (id) for which the 
        ///                   association has to be removed.
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
        bool unbind(id_type id) const
        {
            address addr;   // ignore the return value
            return unbind_range(id, 1, addr);
        }

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
        bool unbind(id_type id, address& addr) const
        {
            return unbind_range(id, 1, addr);
        }
        
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
        bool unbind_range(id_type lower_id, std::size_t count) const
        {
            address addr;   // ignore the return value
            return unbind_range(lower_id, 1, addr);
        }

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
        bool unbind_range(id_type lower_id, std::size_t count, address& addr) const;
        
        /// \brief Asynchronously unbind the given range of global ids
        ///
        /// \param lower_id   [in] The lower bound of the assigned id range.
        ///                   The value must the first id of the range as 
        ///                   specified to the corresponding call to 
        ///                   \a bind_range. 
        /// \param count      [in] The number of consecutive global ids to unbind
        ///                   starting at \a lower_id. This number must be 
        ///                   identical to the number of global ids bound by 
        ///                   the corresponding call to \a bind_range
        ///
        /// \returns          This function returns a future object allowing to
        ///                   defer the evaluation of the outcome of the 
        ///                   function. The actual return value can be retrieved
        ///                   by calling f.get(), where f is the returned future 
        ///                   instance, and f.get() returns \a true the given
        ///                   range has been successfully bound and returns 
        ///                   \a false otherwise. Any error results in an 
        ///                   exception thrown from f.get().
        ///
        /// \note             You can unbind only global ids bound using the 
        ///                   function \a bind_range. Do not use this function 
        ///                   to unbind any of the global ids bound using 
        ///                   \a bind.
        ///
        /// \note             The difference to \a unbind_range is that the 
        ///                   function call returns immediately without 
        ///                   blocking. The operation is guaranteed to be 
        ///                   fully executed only after f.get() has been called.
        util::unique_future<bool> 
            unbind_range_async(id_type lower_id, std::size_t count);

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
        ///
        /// \returns          This function returns \a true if the global 
        ///                   address has been resolved successfully (there 
        ///                   exists an association to a local address) and the 
        ///                   associated local address has been returned. The 
        ///                   function returns \a false if no association exists 
        ///                   for the given global address. Any error results 
        ///                   in an exception thrown from this function.
        bool resolve(id_type id, address& addr) const;

        /// \brief Asynchronously resolve a given global address (id) to its 
        ///        associated local address
        ///
        /// This function returns the local address which is currently 
        /// associated with the given global address (id).
        ///
        /// \param id         [in] The global address (id) for which the 
        ///                   associated local address should be returned.
        ///
        /// \returns          This function returns future object allowing to
        ///                   defer the evaluation of the outcome of the 
        ///                   function. The actual return value can be retrieved
        ///                   by calling f.get(), where f is the returned future 
        ///                   instance, and f.get() returns a pair containing a 
        ///                   bool and the resolved address, where the bool is
        ///                   \a true if the global address has been resolved 
        ///                   successfully (there exists an association to a 
        ///                   local address) and the associated local address 
        ///                   has been returned as the second member of this 
        ///                   pair. The bool is \a false if no association exists 
        ///                   for the given global address. Any error results 
        ///                   in an exception thrown from the f.get() function.
        ///
        /// \note             The difference to \a resolve is that the 
        ///                   function call returns immediately without 
        ///                   blocking. The operation is guaranteed to be 
        ///                   fully executed only after f.get() has been called.
        util::unique_future<std::pair<bool, address> >
            resolve_async(id_type id);

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
        /// 
        /// \returns          The function returns \a true if the global name 
        ///                   got an association with a global address for the 
        ///                   first time, and it returns \a false if this 
        ///                   function call replaced a previously registered 
        ///                   global address with the global address (id) 
        ///                   given as the parameter. Any error results in an 
        ///                   exception thrown from this function.
        bool registerid(std::string const& name, id_type id) const;

        /// \brief Unregister a global name (release any existing association)
        ///
        /// This function releases any existing association of the given global 
        /// name with a global address (id). 
        /// 
        /// \param name       [in] The global name (string) for which any 
        ///                   association with a global address (id) has to be 
        ///                   released.
        /// 
        /// \returns          The function returns \a true if an association of 
        ///                   this global name has been released, and it returns 
        ///                   \a false, if no association existed. Any error 
        ///                   results in an exception thrown from this function.
        bool unregisterid(std::string const& name) const;

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
        /// 
        /// This function returns true if it returned global address (id), 
        /// which is currently associated with the given global name, and it 
        /// returns false, if currently there is no association for this global 
        /// name. Any error results in an exception thrown from this function.
        bool queryid(std::string const& ns_name, id_type& id) const;

        /// \brief Query for the gathered statistics of this DGAS instance 
        ///        (server execution count)
        ///
        /// This function returns the execution counts for each of the 
        /// commands
        /// 
        /// \param counts     [out] The vector will contain the server 
        ///                   execution counts, one entry for each of the 
        ///                   possible resolver_client commands (i.e. will be 
        ///                   of the size 'server::command_lastcommand').
        bool get_statistics_count(std::vector<std::size_t>& counts) const;
        
        /// \brief Query for the gathered statistics of this DGAS instance 
        ///        (average server execution time)
        ///
        /// This function returns the average timings for each of the commands
        /// 
        /// \param timings    [out] The vector will contain the average server 
        ///                   execution times, one entry for each of the 
        ///                   possible resolver_client commands (i.e. will be 
        ///                   of the size 'server::command_lastcommand').
        bool get_statistics_mean(std::vector<double>& timings) const;
        
        /// \brief Query for the gathered statistics of this DGAS instance 
        ///        (statistical 2nd moment of server execution time)
        ///
        /// This function returns the 2nd moment for the timings for each of 
        /// the commands
        /// 
        /// \param moments    [out] The vector will contain the 2nd moment of 
        ///                   the server execution times, one entry for each of 
        ///                   the possible resolver_client commands (i.e. will 
        ///                   be of the size 'server::command_lastcommand').
        bool get_statistics_moment2(std::vector<double>& moments) const;
        
        /// \brief Return the locality of the resolver server this resolver 
        ///        client instance is using to serve requests
        locality const& there() const { return there_; }
        
    protected:
        static bool read_completed(boost::system::error_code const& err, 
            std::size_t bytes_transferred, boost::uint32_t size);
        void execute(server::request const& req, server::reply& rep) const;

    private:
        locality there_;
        util::io_service_pool& io_service_pool_;
        mutable boost::asio::ip::tcp::socket socket_;
    };

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming

#endif
