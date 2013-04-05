////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_15D904C7_CD18_46E1_A54A_65059966A34F)
#define HPX_15D904C7_CD18_46E1_A54A_65059966A34F

#include <hpx/config.hpp>

#include <vector>

#include <boost/make_shared.hpp>
#include <boost/cache/entries/lfu_entry.hpp>
#include <boost/cache/local_cache.hpp>
#include <boost/cache/statistics/local_statistics.hpp>
#include <boost/cstdint.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/noncopyable.hpp>
#include <boost/dynamic_bitset.hpp>

#include <hpx/exception.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/lockfree/fifo.hpp>

// TODO: split into a base class and two implementations (one for bootstrap,
// one for hosted).
// TODO: Use \copydoc.

namespace hpx { namespace agas
{

struct HPX_EXPORT addressing_service : boost::noncopyable
{
    // {{{ types
    typedef component_namespace::component_id_type component_id_type;

    typedef symbol_namespace::iterate_names_function_type
        iterate_names_function_type;

    typedef component_namespace::iterate_types_function_type
        iterate_types_function_type;

    typedef hpx::lcos::local::spinlock cache_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;
    // }}}

    // {{{ gva cache
    struct gva_cache_key
    { // {{{ gva_cache_key implementation
      private:
        typedef boost::icl::closed_interval<naming::gid_type, std::less>
            key_type;

        key_type key_;

      public:
        gva_cache_key()
          : key_()
        {}

        explicit gva_cache_key(
            naming::gid_type const& id_
          , boost::uint64_t count_ = 1
            )
          : key_(naming::detail::get_stripped_gid(id_)
               , naming::detail::get_stripped_gid(id_) + (count_ - 1))
        {
            BOOST_ASSERT(count_);
        }

        naming::gid_type get_gid() const
        {
            return boost::icl::lower(key_);
        }

        boost::uint64_t get_count() const
        {
            naming::gid_type const size = boost::icl::length(key_);
            BOOST_ASSERT(size.get_msb() == 0);
            return size.get_lsb();
        }

        friend bool operator<(
            gva_cache_key const& lhs
          , gva_cache_key const& rhs
            )
        {
            return boost::icl::exclusive_less(lhs.key_, rhs.key_);
        }

        friend bool operator==(
            gva_cache_key const& lhs
          , gva_cache_key const& rhs
            )
        {
            // Is lhs in rhs?
            if (1 == lhs.get_count() && 1 != rhs.get_count())
                return boost::icl::contains(rhs.key_, lhs.key_);

            // Is rhs in lhs?
            else if (1 != lhs.get_count() && 1 == rhs.get_count())
                return boost::icl::contains(lhs.key_, lhs.key_);

            // Direct hit
            return lhs.key_ == rhs.key_;
        }
    }; // }}}

    struct gva_erase_policy
    { // {{{ gva_erase_policy implementation
        gva_erase_policy(
            naming::gid_type const& id
          , boost::uint64_t count
            )
          : entry(id, count)
        {}

        typedef std::pair<
            gva_cache_key, boost::cache::entries::lfu_entry<gva>
        > entry_type;

        bool operator()(
            entry_type const& p
            ) const
        {
            return p.first == entry;
        }

        gva_cache_key entry;
    }; // }}}

    typedef boost::cache::entries::lfu_entry<gva> gva_entry_type;

    typedef boost::cache::local_cache<
        gva_cache_key, gva_entry_type,
        std::less<gva_entry_type>,
        boost::cache::policies::always<gva_entry_type>,
        std::map<gva_cache_key, gva_entry_type>,
        boost::cache::statistics::local_statistics
    > gva_cache_type;
    // }}}

    typedef boost::lockfree::fifo<
        lcos::packaged_action<server::locality_namespace::service_action>*
    > locality_promise_pool_type;

    typedef boost::lockfree::fifo<
        lcos::packaged_action<server::primary_namespace::service_action>*
    > primary_promise_pool_type;

    typedef util::merging_map<naming::gid_type, boost::int64_t>
        refcnt_requests_type;

    struct bootstrap_data_type
    { // {{{
        bootstrap_data_type()
          : primary_ns_server_()
          , locality_ns_server_(&primary_ns_server_)
          , component_ns_server_()
          , symbol_ns_server_()
        {}

        void register_counter_types()
        {
            server::locality_namespace::register_counter_types();
            server::primary_namespace::register_counter_types();
            server::component_namespace::register_counter_types();
            server::symbol_namespace::register_counter_types();
        }

        void register_server_instance(char const* servicename)
        {
            locality_ns_server_.register_server_instance(servicename);
            primary_ns_server_.register_server_instance(servicename);
            component_ns_server_.register_server_instance(servicename);
            symbol_ns_server_.register_server_instance(servicename);
        }

        void unregister_server_instance(error_code& ec)
        {
            locality_ns_server_.unregister_server_instance(ec);
            primary_ns_server_.unregister_server_instance(ec);
            component_ns_server_.unregister_server_instance(ec);
            symbol_ns_server_.unregister_server_instance(ec);
        }

        server::primary_namespace primary_ns_server_;
        server::locality_namespace locality_ns_server_;
        server::component_namespace component_ns_server_;
        server::symbol_namespace symbol_ns_server_;
    }; // }}}

    struct hosted_data_type
    { // {{{
        hosted_data_type()
          : locality_promise_pool_(16)
          , primary_promise_pool_(16)
        {}

        void register_counter_types()
        {
            server::locality_namespace::register_counter_types();
            server::primary_namespace::register_counter_types();
            server::component_namespace::register_counter_types();
            server::symbol_namespace::register_counter_types();
        }

        void register_server_instance(char const* servicename
          , boost::uint32_t locality_id)
        {
            primary_ns_server_.register_server_instance(servicename, locality_id);
        }

        void unregister_server_instance(error_code& ec)
        {
            primary_ns_server_.unregister_server_instance(ec);
        }

        locality_namespace locality_ns_;
        primary_namespace primary_ns_;
        component_namespace component_ns_;
        symbol_namespace symbol_ns_;

        server::primary_namespace primary_ns_server_;

        hpx::lcos::local::counting_semaphore promise_pool_semaphore_;
        locality_promise_pool_type locality_promise_pool_;
        primary_promise_pool_type primary_promise_pool_;
    }; // }}}

    mutable cache_mutex_type gva_cache_mtx_;
    gva_cache_type gva_cache_;

    mutable mutex_type console_cache_mtx_;
    boost::uint32_t console_cache_;

    std::size_t const max_refcnt_requests_;

    mutex_type refcnt_requests_mtx_;
    std::size_t refcnt_requests_count_;
    boost::shared_ptr<refcnt_requests_type> refcnt_requests_;

    service_mode const service_type;
    runtime_mode const runtime_type;

    bool const caching_;
    bool const range_caching_;
    threads::thread_priority const action_priority_;

    mutable naming::locality here_;
    boost::uint64_t rts_lva_;

    boost::shared_ptr<bootstrap_data_type> bootstrap;
    boost::shared_ptr<hosted_data_type> hosted;

    boost::atomic<hpx::state> state_;
    naming::gid_type locality_;

    naming::address locality_ns_addr_;
    naming::address primary_ns_addr_;
    naming::address component_ns_addr_;
    naming::address symbol_ns_addr_;

    addressing_service(
        parcelset::parcelport& pp
      , util::runtime_configuration const& ini_
      , runtime_mode runtime_type_
        );

    ~addressing_service()
    {
        // TODO: Free the future pools?
        destroy_big_boot_barrier();
    }

    void launch_bootstrap(
        util::runtime_configuration const& ini_
        );

    void launch_hosted();

    void adjust_local_cache_size();

    state status() const
    {
        if (!hosted && !bootstrap)
            return stopping;
        return state_.load();
    }

    void status(state new_state)
    {
        state_.store(new_state);
    }

    naming::gid_type const& get_local_locality(error_code& ec = throws) const
    {
        if (locality_ == naming::invalid_gid) {
            HPX_THROWS_IF(ec, invalid_status,
                "addressing_service::get_local_locality",
                "local locality has not been initialized (yet)");
        }
        return locality_;
    }

    void set_local_locality(naming::gid_type const& g)
    {
        locality_ = g;
    }

    bool is_bootstrap() const
    {
        return service_type == service_mode_bootstrap;
    }

    /// \brief Returns whether this addressing_service represents the console
    ///        locality.
    bool is_console() const
    {
        return runtime_type == runtime_mode_console;
    }

    /// \brief Returns whether this addressing_service is connecting to a
    ///        running application
    bool is_connecting() const
    {
        return runtime_type == runtime_mode_connect;
    }

    bool resolve_locally_known_addresses(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec
      );

    /// \brief Register performance counter types exposing properties from the
    ///        local cache.
    void register_counter_types();

    // FIXME: document (add comments)
    void garbage_collect_non_blocking(
        error_code& ec = throws
        );

    // FIXME: document (add comments)
    void garbage_collect(
        error_code& ec = throws
        );

    naming::locality const& get_here() const;

private:
    /// Assumes that \a refcnt_requests_mtx_ is locked.
    void increment_refcnt_requests(
        mutex_type::scoped_lock& l
      , error_code& ec
        );

    /// Assumes that \a refcnt_requests_mtx_ is locked.
    void send_refcnt_requests_non_blocking(
        mutex_type::scoped_lock& l
      , error_code& ec
        );

    /// Assumes that \a refcnt_requests_mtx_ is locked.
    void send_refcnt_requests_sync(
        mutex_type::scoped_lock& l
      , error_code& ec
        );

    // Helper functions to access the current cache statistics
    std::size_t get_cache_hits(bool);
    std::size_t get_cache_misses(bool);
    std::size_t get_cache_evictions(bool);
    std::size_t get_cache_insertions(bool);

public:
    response service(
        request const& req
      , error_code& ec = throws
        );

    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , error_code& ec = throws
        );

    /// \brief Add a locality to the runtime.
    bool register_locality(
        naming::locality const& l
      , naming::gid_type& prefix
      , boost::uint32_t num_threads
      , error_code& ec = throws
        );

    /// \brief Resolve a locality to it's prefix.
    ///
    /// \returns Returns 0 if the locality is not registered.
    boost::uint32_t resolve_locality(
        naming::locality const& l
      , error_code& ec = throws
        );

    /// \brief Remove a locality from the runtime.
    bool unregister_locality(
        naming::locality const& l
      , error_code& ec = throws
        );

    /// \brief Get locality locality_id of the console locality.
    ///
    /// \param locality_id     [out] The locality_id value uniquely identifying the
    ///                   console locality. This is valid only, if the
    ///                   return value of this function is true.
    /// \param try_cache  [in] If this is set to true the console is first
    ///                   tried to be found in the local cache. Otherwise
    ///                   this function will always query AGAS, even if the
    ///                   console locality_id is already known locally.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if a console locality_id
    ///                   exists and returns \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool get_console_locality(
        naming::gid_type& locality_id
      , error_code& ec = throws
        );

    /// \brief Query for the locality_ids of all known localities.
    ///
    /// This function returns the locality_ids of all localities known to the
    /// AGAS server or all localities having a registered factory for a
    /// given component type.
    ///
    /// \param locality_ids [out] The vector will contain the prefixes of all
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool get_localities(
        std::vector<naming::gid_type>& locality_ids
      , components::component_type type
      , error_code& ec = throws
        );

    bool get_localities(
        std::vector<naming::gid_type>& locality_ids
      , error_code& ec = throws
        )
    {
        return get_localities(locality_ids, components::component_invalid, ec);
    }

    /// \brief Query the resolved addresses for all know localities
    ///
    /// This function returns the resolved addresses for all localities known
    /// to the AGAS server.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    lcos::future<std::vector<naming::locality> > get_resolved_localities_async();

    std::vector<naming::locality> get_resolved_localities(
        error_code& ec = throws
        )
    {
        return get_resolved_localities_async().get(ec);
    }

    /// \brief Query for the number of all known localities.
    ///
    /// This function returns the number of localities known to the AGAS server
    /// or the number of localities having a registered factory for a given
    /// component type.
    ///
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    lcos::future<boost::uint32_t> get_num_localities_async(
        components::component_type type = components::component_invalid
        );

    boost::uint32_t get_num_localities(
        components::component_type type
      , error_code& ec = throws
        );

    boost::uint32_t get_num_localities(error_code& ec = throws)
    {
        return get_num_localities(components::component_invalid, ec);
    }

    lcos::future<boost::uint32_t> get_num_overall_threads_async();

    boost::uint32_t get_num_overall_threads(
        error_code& ec = throws
        );

    lcos::future<std::vector<boost::uint32_t> > get_num_threads_async();

    std::vector<boost::uint32_t> get_num_threads(
        error_code& ec = throws
        );

    /// \brief Return a unique id usable as a component type.
    ///
    /// This function returns the component type id associated with the
    /// given component name. If this is the first request for this
    /// component name a new unique id will be created.
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    components::component_type get_component_id(
        std::string const& name
      , error_code& ec = throws
        );

    void iterate_types(
        iterate_types_function_type const& f
      , error_code& ec = throws
        );

    std::string get_component_type_name(
        components::component_type id
      , error_code& ec = throws
        );

    /// \brief Register a factory for a specific component type
    ///
    /// This function allows to register a component factory for a given
    /// locality and component type.
    ///
    /// \param locality_id  [in] The locality value uniquely identifying the
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    components::component_type register_factory(
        naming::gid_type const& locality_id
      , std::string const& name
      , error_code& ec = throws
        )
    {
        return register_factory(naming::get_locality_id_from_gid(locality_id), name, ec);
    }

    components::component_type register_factory(
        boost::uint32_t locality_id
      , std::string const& name
      , error_code& ec = throws
        );

    /// \brief Get unique range of freely assignable global ids.
    ///
    /// Every locality needs to be able to assign global ids to different
    /// components without having to consult the AGAS server for every id
    /// to generate. This function can be called to preallocate a range of
    /// ids usable for this purpose.
    ///
    /// \param l          [in] The locality the locality id needs to be
    ///                   generated for. Repeating calls using the same
    ///                   locality results in identical locality_id values.
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool get_id_range(
        naming::locality const& l
      , boost::uint64_t count
      , naming::gid_type& lower_bound
      , naming::gid_type& upper_bound
      , error_code& ec = throws
        );

    /// \brief Bind a global address to a local address.
    ///
    /// Every element in the HPX namespace has a unique global address
    /// (global id). This global address has to be associated with a concrete
    /// local address to be able to address an instance of a component using
    /// it's global address.
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
    ///                   got associated with an local address. It returns
    ///                   \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             Binding a gid to a local address sets its global
    ///                   reference count to one.
    bool bind(
        naming::gid_type const& id
      , naming::address const& addr
      , error_code& ec = throws
        )
    {
        return bind_range(id, 1, addr, 0, ec);
    }

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
    /// \returns          This function returns \a true, if the given range
    ///                   was successfully bound. It returns \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             Binding a gid to a local address sets its global
    ///                   reference count to one.
    bool bind_range(
        naming::gid_type const& lower_id
      , boost::uint64_t count
      , naming::address const& baseaddr
      , boost::uint64_t offset
      , error_code& ec = throws
        );

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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             This function will raise an error if the global
    ///                   reference count of the given gid is not zero!
    ///                   TODO: confirm that this happens.
    bool unbind(
        naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        return unbind_range(id, 1, ec);
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             This function will raise an error if the global
    ///                   reference count of the given gid is not zero!
    ///                   TODO: confirm that this happens.
    bool unbind(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        return unbind_range(id, 1, addr, ec);
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             This function will raise an error if the global
    ///                   reference count of the given gid is not zero!
    ///                   TODO: confirm that this happens.
    bool unbind_range(
        naming::gid_type const& lower_id
      , boost::uint64_t count
      , error_code& ec = throws
        )
    {
        naming::address addr;
        return unbind_range(lower_id, count, addr, ec);
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    ///
    /// \note             This function will raise an error if the global
    ///                   reference count of the given gid is not zero!
    bool unbind_range(
        naming::gid_type const& lower_id
      , boost::uint64_t count
      , naming::address& addr
      , error_code& ec = throws
        );

    /// \brief Test whether the given address refers to a local object.
    ///
    /// This function will test whether the given address refers to an object
    /// living on the locality of the caller.
    ///
    /// \param addr       [in] The address to test.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    /// \returns          This function returns \a true if the passed address
    ///                   refers to an object which lives on the locality of
    ///                   the caller.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool is_local_address(
        naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        return is_local_address_cached(id, addr, ec);
    }

    bool is_local_address(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        return is_local_address_cached(id, addr, ec);
    }

    bool is_local_address_cached(
        naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        return is_local_address_cached(id, addr, ec);
    }

    bool is_local_address_cached(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        );

    bool is_local_lva_encoded_address(
        naming::gid_type const& id
        );

    // same, but bulk operation
    bool is_local_address(
        std::vector<naming::gid_type>& gids
      , std::vector<naming::address>& addrs
      , boost::dynamic_bitset<>& locals
      , error_code& ec = throws
        );

    /// \brief Resolve a given global address (\a id) to its associated local
    ///        address.
    ///
    /// This function returns the local address which is currently associated
    /// with the given global address (\a id).
    ///
    /// \param id         [in] The global address (\a id) for which the
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool resolve(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        // Try the cache
        if (caching_)
        {
            if (resolve_cached(id, addr, ec))
                return true;

            if (ec)
                return false;
        }

        return resolve_full(id, addr, ec);
    }

    bool resolve(
        naming::id_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        return resolve(id.get_gid(), addr, ec);
    }

    naming::address resolve(
        naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        resolve(id, addr, ec);
        return addr;
    }

    naming::address resolve(
        naming::id_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        resolve(id.get_gid(), addr, ec);
        return addr;
    }

    bool resolve_full(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        );

    bool resolve_full(
        naming::id_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        return resolve_full(id.get_gid(), addr, ec);
    }

    naming::address resolve_full(
        naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        resolve_full(id, addr, ec);
        return addr;
    }

    naming::address resolve_full(
        naming::id_type const& id
      , error_code& ec = throws
        )
    {
        naming::address addr;
        resolve_full(id.get_gid(), addr, ec);
        return addr;
    }

    bool resolve_cached(
        naming::gid_type const& id
      , naming::address& addr
      , error_code& ec = throws
        );

    bool resolve_cached(
        naming::id_type const& id
      , naming::address& addr
      , error_code& ec = throws
        )
    {
        return resolve_cached(id.get_gid(), addr, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Bulk version.
    // TODO: Add versions that take std::vector<id_type> for convenience.
    bool resolve(
        std::vector<naming::gid_type> const& gids
      , std::vector<naming::address>& addrs
      , boost::dynamic_bitset<>& locals
      , error_code& ec = throws
        )
    {
        // Try the cache.
        if (caching_)
        {
            bool all_resolved = resolve_cached(gids, addrs, locals, ec);
            if (ec)
                return false;
            if (all_resolved)
                return true; // Nothing more to do.
        }

        return resolve_full(gids, addrs, locals, ec);
    }

    bool resolve_full(
        std::vector<naming::gid_type> const& gids
      , std::vector<naming::address>& addrs
      , boost::dynamic_bitset<>& locals
      , error_code& ec = throws
        );

    bool resolve_cached(
        std::vector<naming::gid_type> const& gids
      , std::vector<naming::address>& addrs
      , boost::dynamic_bitset<>& locals
      , error_code& ec = throws
        );

    /// \brief Route the given parcel to the appropriate AGAS service instance
    ///
    /// This function sends the given parcel to the AGAS service instance which 
    /// manages the parcel's destination GID. This service instance will resolve
    /// the GID and either send (route) the parcel to the correct locality or
    /// it will deliver the parcel to the local action manager.
    ///
    /// \param p          [in] this is the parcel which has to be routed to the
    ///                   AGAS service instance managing the destination GID.
    ///
    /// \note             The route operation is asynchronous, thus it returns 
    ///                   before the parcel has been delivered to its 
    ///                   destination.
    void route(
        parcelset::parcel const& p
        );

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
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    void incref(
        naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t credits = 1
      , error_code& ec = throws
        );

    void incref(
        naming::gid_type const& id
      , boost::int64_t credits = 1
      , error_code& ec = throws
        )
    {
        return incref(id, id, credits, ec);
    }

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
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    void decref(
        naming::gid_type const& lower
      , naming::gid_type const& upper
      , boost::int64_t credits = 1
      , error_code& ec = throws
        );

    void decref(
        naming::gid_type const& id
      , boost::int64_t credits = 1
      , error_code& ec = throws
        )
    {
        return decref(id, id, credits, ec);
    }

#if !defined(HPX_NO_DEPRECATED)
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_DEPRECATED("This function is deprecated; use "
                   "hpx::agas::register_name instead.")
    bool registerid(
        std::string const& name
      , naming::gid_type const& id
      , error_code& ec = throws
        )
    {
        return register_name(name, id, ec);
    }

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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_DEPRECATED("This function is deprecated; use "
                   "hpx::agas::unregister_name instead.")
    bool unregisterid(
        std::string const& name
      , error_code& ec = throws
        )
    {
        return unregister_name(name, ec);
    }

    HPX_DEPRECATED("This function is deprecated; use "
                   "hpx::agas::unregister_name instead.")
    bool unregisterid(
        std::string const& name
      , naming::gid_type& id
      , error_code& ec = throws
        )
    {
        return unregister_name(name, id, ec);
    }

    /// \brief Query for the global address associated with a given global name.
    ///
    /// This function returns the global address associated with the given
    /// global name.
    ///
    /// \param name       [in] The global name (string) for which the
    ///                   currently associated global address has to be
    ///                   retrieved.
    /// \param id         [out] The id currently associated with the given
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_DEPRECATED("This function is deprecated; use "
                   "hpx::agas::resolve_name instead.")
    bool queryid(
        std::string const& name
      , naming::gid_type& id
      , error_code& ec = throws
        )
    {
        return resolve_name(name, id, ec);
    }
#endif

    /// \brief Invoke the supplied \a hpx#function for every registered global
    ///        name.
    ///
    /// This function iterates over all registered global ids and
    /// unconditionally invokes the supplied hpx#function for ever found entry.
    /// Any error results in an exception thrown (or reported) from this
    /// function.
    ///
    /// \param f          [in] a \a hpx#function encapsulating an action to be
    ///                   invoked for every currently registered global name.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool iterate_ids(
        iterate_names_function_type const& f
      , error_code& ec = throws
        );

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
    ///                   was registered. It returns false if the global name is
    ///                   not registered.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool register_name(
        std::string const& name
      , naming::gid_type const& id
      , error_code& ec = throws
        );

    lcos::future<bool> register_name_async(
        std::string const& name
      , naming::id_type const& id
        );

    bool register_name(
        std::string const& name
      , naming::id_type const& id
      , error_code& ec = throws
        )
    {
        return register_name_async(name, id).get(ec);
    }

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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool unregister_name(
        std::string const& name
      , naming::gid_type& id
      , error_code& ec = throws
        );

    lcos::future<naming::id_type> unregister_name_async(
        std::string const& name
        );

    naming::id_type unregister_name(
        std::string const& name
      , error_code& ec = throws
        )
    {
        return unregister_name_async(name).get(ec);
    }

    /// \brief Query for the global address associated with a given global name.
    ///
    /// This function returns the global address associated with the given
    /// global name.
    ///
    /// \param name       [in] The global name (string) for which the
    ///                   currently associated global address has to be
    ///                   retrieved.
    /// \param id         [out] The id currently associated with the given
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
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    bool resolve_name(
        std::string const& name
      , naming::gid_type& id
      , error_code& ec = throws
        );

    lcos::future<naming::id_type> resolve_name_async(
        std::string const& name
        );

    naming::id_type resolve_name(
        std::string const& name
      , error_code& ec = throws
        )
    {
        return resolve_name_async(name).get(ec);
    }

    /// \warning This function is for internal use only. It is dangerous and
    ///          may break your code if you use it.
    void insert_cache_entry(
        naming::gid_type const& gid
      , gva const& gva
      , error_code& ec = throws
        );

    /// \warning This function is for internal use only. It is dangerous and
    ///          may break your code if you use it.
    void insert_cache_entry(
        naming::gid_type const& gid
      , naming::address const& addr
      , error_code& ec = throws
        )
    {
        const gva g(addr.locality_, addr.type_, 1, addr.address_);
        insert_cache_entry(gid, g, ec);
    }

    /// \warning This function is for internal use only. It is dangerous and
    ///          may break your code if you use it.
    void update_cache_entry(
        naming::gid_type const& gid
      , gva const& gva
      , error_code& ec = throws
        );

    /// \warning This function is for internal use only. It is dangerous and
    ///          may break your code if you use it.
    void update_cache_entry(
        naming::gid_type const& gid
      , naming::address const& addr
      , error_code& ec = throws
        )
    {
        const gva g(addr.locality_, addr.type_, 1, addr.address_);
        update_cache_entry(gid, g, ec);
    }

    /// \warning This function is for internal use only. It is dangerous and
    ///          may break your code if you use it.
    void clear_cache(
        error_code& ec = throws
        );

    /// \brief Retrieve statistics performance counter
    bool retrieve_statistics_counter(
        std::string const& counter_name
      , naming::gid_type& counter
      , error_code& ec = throws
        );
};

}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

