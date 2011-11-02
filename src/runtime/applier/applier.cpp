//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace applier
{
    //
    lcos::promise<naming::id_type, naming::gid_type>
    create_async(naming::id_type const& targetgid,
        components::component_type type, std::size_t count)
    {
        // Create a promise, execute the required action,
        // we simply return the initialized promise, the caller needs
        // to call get() on the return value to obtain the result
        typedef
            components::server::runtime_support::create_component_action
        action_type;
        return lcos::eager_future<action_type, naming::id_type>(targetgid, type, count);
    }

    //
    naming::id_type create(naming::id_type const& targetgid,
        components::component_type type, std::size_t count)
    {
        return create_async(targetgid, type, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static inline threads::thread_state thread_function(
        HPX_STD_FUNCTION<void(threads::thread_state_ex)> const& func)
    {
        func(threads::thread_state_ex(threads::wait_signaled));
        return threads::thread_state(threads::terminated);
    }

    static inline threads::thread_state thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        func();
        return threads::thread_state(threads::terminated);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_id_type register_thread_nullary(
        HPX_STD_FUNCTION<void()> const& func, char const* desc,
        threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread_nullary",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function_nullary, func), desc ? desc : "<unknown>",
            0, priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread(
        HPX_STD_FUNCTION<void(threads::thread_state_ex)> const& func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function, func), desc ? desc : "<unknown>", 0,
            priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread_plain(
        HPX_STD_FUNCTION<threads::thread_function_type> const& func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread_plain",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(func, desc ? desc : "<unknown>",
            0, priority, os_thread);
        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    threads::thread_id_type register_thread_plain(
        threads::thread_init_data& data, threads::thread_state_enum state,
        bool run_now, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread_plain",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        return app->get_thread_manager().
            register_thread(data, state, run_now, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_work_nullary(
        HPX_STD_FUNCTION<void()> const& func, char const* desc,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_work_nullary",
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function_nullary, func),
            desc ? desc : "<unknown>", 0, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work(
        HPX_STD_FUNCTION<void(threads::thread_state_ex)> const& func,
        char const* desc, threads::thread_state_enum state,
        threads::thread_priority priority, std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_work",
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(
            boost::bind(&thread_function, func),
            desc ? desc : "<unknown>", 0, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        HPX_STD_FUNCTION<threads::thread_function_type> const& func,
        char const* desc, naming::address::address_type lva,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_work_plain",
                "global applier object is not accessible");
            return;
        }

        threads::thread_init_data data(func,
            desc ? desc : "<unknown>", lva, priority, os_thread);
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        threads::thread_init_data& data, threads::thread_state_enum state,
        error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_work_plain",
                "global applier object is not accessible");
            return;
        }

        app->get_thread_manager().register_work(data, state, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::util::thread_specific_ptr<applier*, applier::tls_tag> applier::applier_;

    applier::applier(parcelset::parcelhandler &ph, threads::threadmanager_base& tm,
                boost::uint64_t rts, boost::uint64_t mem)
      : parcel_handler_(ph), thread_manager_(tm),
        runtime_support_id_(parcel_handler_.get_prefix().get_msb(), rts,
            parcel_handler_.here(), components::component_runtime_support,
            rts, naming::id_type::unmanaged),
        memory_id_(parcel_handler_.get_prefix().get_msb(), mem,
            parcel_handler_.here(), components::component_runtime_support,
            mem, naming::id_type::unmanaged)
    {}

    naming::resolver_client& applier::get_agas_client()
    {
        return hpx::naming::get_agas_client();
    }

    parcelset::parcelhandler& applier::get_parcel_handler()
    {
        return parcel_handler_;
    }

    threads::threadmanager_base& applier::get_thread_manager()
    {
        return thread_manager_;
    }

    naming::locality const& applier::here() const
    {
        return hpx::get_locality();
    }

    naming::gid_type const& applier::get_prefix() const
    {
        return hpx::naming::get_agas_client().local_prefix();
    }

    boost::uint32_t applier::get_prefix_id() const
    {
        return naming::get_prefix_from_gid(get_prefix());
    }

    bool applier::get_raw_remote_prefixes(std::vector<naming::gid_type>& prefixes,
        components::component_type type) const
    {
        return parcel_handler_.get_raw_remote_prefixes(prefixes, type);
    }

    bool applier::get_remote_prefixes(std::vector<naming::id_type>& prefixes,
        components::component_type type) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_remote_prefixes(raw_prefixes, type))
            return false;

        BOOST_FOREACH(naming::gid_type& gid, raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    bool applier::get_raw_prefixes(std::vector<naming::gid_type>& prefixes,
        components::component_type type) const
    {
        return parcel_handler_.get_raw_prefixes(prefixes, type);
    }

    bool applier::get_prefixes(std::vector<naming::id_type>& prefixes,
        components::component_type type) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_prefixes(raw_prefixes, type))
            return false;

        BOOST_FOREACH(naming::gid_type& gid, raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    bool applier::address_is_local(naming::id_type const& id,
        naming::address& addr) const
    {
        if (id.is_local()) {    // address gets resolved if not already
            if (!id.get_local_address(addr)) {
                hpx::util::osstream strm;
                strm << "gid" << id.get_gid();
                HPX_THROW_EXCEPTION(invalid_status,
                    "applier::address_is_local",
                    hpx::util::osstream_get_string(strm));
            }
            hpx::applier::get_applier().get_agas_client().resolve(id, addr, true, throws);
            return true;
        }

        if (!id.is_resolved()) {
            hpx::util::osstream strm;
            strm << "gid" << id.get_gid();
            HPX_THROW_EXCEPTION(unknown_component_address,
                "applier::address_is_local", hpx::util::osstream_get_string(strm));
        }

        if (!id.get_address_cached(addr)) {
            hpx::util::osstream strm;
            strm << "gid" << id.get_gid();
            HPX_THROW_EXCEPTION(invalid_status,
                "applier::address_is_local",
                hpx::util::osstream_get_string(strm));
        }
        return false;   // non-local
    }

    bool applier::address_is_local_c_cache(naming::id_type const& id, 
        naming::address& addr) const
    {
        bool is_local = id.is_local_c_cache();
        bool in_cache = id.get_local_address_c_cache(addr);
        if(is_local) {     // check address if it is in cache
            if (!in_cache) {
                //object is local, but cannot be found in cache
                //or should local object be always be in cache?
                //TODO: validate this argument
                return false;
            }
            hpx::applier::get_applier().get_agas_client().resolve_cached(
                    id.get_gid(), addr, throws);
            return true;
        }

        if (!id.is_resolved() && is_local) {
            //object is local, but it is not resolved
            //object is not local, and id is not resolved
            //in both cases need to send parcel to agas to resolve address.
            //TODO: remove this test
        }

        if (!id.get_address_cached(addr) && in_cache) {
            //object is in cache, but cannot retrieve address from cache
            //should throw 
            hpx::util::osstream strm;
            strm << "gid" << id.get_gid();
            HPX_THROW_EXCEPTION(invalid_status, 
                "applier::address_is_local_c_cache", 
                hpx::util::osstream_get_string(strm));
        }
        return false;   // non-local
    }

    bool applier::address_is_local(naming::gid_type const& gid,
        naming::address& addr) const
    {
        // test if the gid is of one of the non-movable objects
        // this is certainly an optimization relying on the fact that the
        // lsb of the local objects is equal to their address
        if (naming::strip_credit_from_gid(gid.get_msb()) ==
                parcel_handler_.get_prefix().get_msb())
        {
            // a zero address references the local runtime support component
            if (0 != gid.get_lsb())
                addr.address_ = gid.get_lsb();
            else
                addr.address_ = runtime_support_id_.get_lsb();
            return true;
        }

        // Resolve the address of the gid
        if (!parcel_handler_.get_resolver().resolve(gid, addr))
        {
            hpx::util::osstream strm;
            strm << "gid" << gid;
            HPX_THROW_EXCEPTION(unknown_component_address,
                "applier::address_is_local", hpx::util::osstream_get_string(strm));
        }
        return addr.locality_ == parcel_handler_.here();
    }

    bool applier::address_is_local_c_cache(naming::gid_type const& gid, 
        naming::address& addr) const
    {
        // test if the gid is of one of the non-movable objects
        // this is certainly an optimization relying on the fact that the 
        // lsb of the local objects is equal to their address
        if (naming::strip_credit_from_gid(gid.get_msb()) == 
                parcel_handler_.get_prefix().get_msb())
        {
            // a zero address references the local runtime support component
            if (0 != gid.get_lsb())
                addr.address_ = gid.get_lsb();
            else 
                addr.address_ = runtime_support_id_.get_lsb();
            return true;
        }

        // Resolve the address of the gid
        if (!parcel_handler_.get_resolver().resolve_cached(gid, addr))
        {
            hpx::util::osstream strm;
            strm << "gid" << gid;
            HPX_THROW_EXCEPTION(unknown_component_address, 
                "applier::address_is_local", hpx::util::osstream_get_string(strm));
        }
        return addr.locality_ == parcel_handler_.here();
    }

    // parcel forwarding.
    bool applier::route(parcelset::parcel const& p)
    {
        return get_agas_client().route_parcel(p);
    }

    void applier::init_tss()
    {
        BOOST_ASSERT(NULL == applier::applier_.get());    // shouldn't be initialized yet
        applier::applier_.reset(new applier* (this));
    }

    void applier::deinit_tss()
    {
        applier::applier_.reset();
    }

    applier& get_applier()
    {
        BOOST_ASSERT(NULL != applier::applier_.get());   // should have been initialized
        return **applier::applier_;
    }

    applier* get_applier_ptr()
    {
        applier** appl = applier::applier_.get();
        return appl ? *appl : NULL;
    }

    // The function \a get_prefix_id returns the id of this locality
    boost::uint32_t get_prefix_id()
    {
        applier** appl = applier::applier_.get();
        return appl ? (*appl)->get_prefix_id() : 0;
    }
}}
