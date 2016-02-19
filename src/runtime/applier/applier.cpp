//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/include/async.hpp>
#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/capability.hpp>
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <memory>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    void dijkstra_make_black();     // forward declaration only
}}

namespace hpx { namespace applier
{
    ///////////////////////////////////////////////////////////////////////////
    static inline threads::thread_state_enum thread_function(
        util::unique_function_nonser<void(threads::thread_state_ex_enum)> func)
    {
        // execute the actual thread function
        func(threads::wait_signaled);

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    static inline threads::thread_state_enum thread_function_nullary(
        util::unique_function_nonser<void()> func)
    {
        // execute the actual thread function
        func();

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_id_type register_thread_nullary(
        util::unique_function_nonser<void()> && func, char const* desc,
        threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread,
        threads::thread_stacksize stacksize, error_code& ec)
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
            util::bind(util::one_shot(&thread_function_nullary), std::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));

        threads::thread_id_type id = threads::invalid_thread_id;
        app->get_thread_manager().register_thread(data, id, state, run_now, ec);
        return id;
    }

    threads::thread_id_type register_thread(
        util::unique_function_nonser<void(threads::thread_state_ex_enum)> && func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread,
        threads::thread_stacksize stacksize, error_code& ec)
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
            util::bind(util::one_shot(&thread_function), std::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));

        threads::thread_id_type id = threads::invalid_thread_id;
        app->get_thread_manager().register_thread(data, id, state, run_now, ec);
        return id;
    }

    threads::thread_id_type register_thread_plain(
        threads::thread_function_type && func,
        char const* desc, threads::thread_state_enum state, bool run_now,
        threads::thread_priority priority, std::size_t os_thread,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        hpx::applier::applier* app = hpx::applier::get_applier_ptr();
        if (NULL == app)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "hpx::applier::register_thread_plain",
                "global applier object is not accessible");
            return threads::invalid_thread_id;
        }

        threads::thread_init_data data(
            std::move(func), desc ? desc : "<unknown>", 0, priority,
            os_thread, threads::get_stack_size(stacksize));

        threads::thread_id_type id = threads::invalid_thread_id;
        app->get_thread_manager().register_thread(data, id, state, run_now, ec);
        return id;
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

        threads::thread_id_type id = threads::invalid_thread_id;
        app->get_thread_manager().register_thread(data, id, state, run_now, ec);
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_work_nullary(
        util::unique_function_nonser<void()> && func, char const* desc,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, threads::thread_stacksize stacksize,
        error_code& ec)
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
            util::bind(util::one_shot(&thread_function_nullary), std::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work(
        util::unique_function_nonser<void(threads::thread_state_ex_enum)> && func,
        char const* desc, threads::thread_state_enum state,
        threads::thread_priority priority, std::size_t os_thread,
        threads::thread_stacksize stacksize, error_code& ec)
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
            util::bind(util::one_shot(&thread_function), std::move(func)),
            desc ? desc : "<unknown>", 0, priority, os_thread,
            threads::get_stack_size(stacksize));
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        threads::thread_function_type && func,
        char const* desc, naming::address::address_type lva,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, threads::thread_stacksize stacksize,
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

        threads::thread_init_data data(std::move(func),
            desc ? desc : "<unknown>", lva, priority, os_thread,
            threads::get_stack_size(stacksize));
        app->get_thread_manager().register_work(data, state, ec);
    }

    void register_work_plain(
        threads::thread_function_type && func,
        naming::id_type const& target,
        char const* desc, naming::address::address_type lva,
        threads::thread_state_enum state, threads::thread_priority priority,
        std::size_t os_thread, threads::thread_stacksize stacksize,
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

        threads::thread_init_data data(std::move(func),
            desc ? desc : "<unknown>", lva, priority, os_thread,
            threads::get_stack_size(stacksize), target);
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

    applier::applier(parcelset::parcelhandler &ph, threads::threadmanager_base& tm)
      : parcel_handler_(ph), thread_manager_(tm)
#if defined(HPX_HAVE_SECURITY)
      , verify_capabilities_(false)
#endif
    {}

    void applier::initialize(boost::uint64_t rts, boost::uint64_t mem)
    {
        naming::resolver_client & agas_client = get_agas_client();
        runtime_support_id_ = naming::id_type(agas_client.get_local_locality().get_msb(),
                rts, naming::id_type::unmanaged);
        memory_id_ = naming::id_type(agas_client.get_local_locality().get_msb(),
            mem, naming::id_type::unmanaged);
    }

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

    naming::gid_type const& applier::get_raw_locality(error_code& ec) const
    {
        return hpx::naming::get_agas_client().get_local_locality(ec);
    }

    boost::uint32_t applier::get_locality_id(error_code& ec) const
    {
        return naming::get_locality_id_from_gid(get_raw_locality(ec));
    }

    bool applier::get_raw_remote_localities(std::vector<naming::gid_type>& prefixes,
        components::component_type type, error_code& ec) const
    {
        return parcel_handler_.get_raw_remote_localities(prefixes, type, ec);
    }

    bool applier::get_remote_localities(std::vector<naming::id_type>& prefixes,
        components::component_type type, error_code& ec) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_remote_localities(raw_prefixes, type, ec))
            return false;

        for (naming::gid_type& gid : raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    bool applier::get_raw_localities(std::vector<naming::gid_type>& prefixes,
        components::component_type type) const
    {
        return parcel_handler_.get_raw_localities(prefixes, type);
    }

    bool applier::get_localities(std::vector<naming::id_type>& prefixes,
        error_code& ec) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_localities(raw_prefixes,
            components::component_invalid, ec))
            return false;

        for (naming::gid_type& gid : raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    bool applier::get_localities(std::vector<naming::id_type>& prefixes,
        components::component_type type, error_code& ec) const
    {
        std::vector<naming::gid_type> raw_prefixes;
        if (!parcel_handler_.get_raw_localities(raw_prefixes, type, ec))
            return false;

        for (naming::gid_type& gid : raw_prefixes)
            prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

        return true;
    }

    void applier::init_tss()
    {
        if (NULL == applier::applier_.get())
            applier::applier_.reset(new applier* (this));
    }

    void applier::deinit_tss()
    {
        applier::applier_.reset();
    }

    namespace detail
    {
        // The original parcel-sent handler is wrapped to keep the parcel alive
        // until after the data has been reliably sent (which is needed for zero
        // copy serialization).
        void parcel_sent_handler(parcelset::parcelhandler& ph,
            boost::system::error_code const& ec,
            parcelset::parcel const& p)
        {
            // invoke the original handler
            ph.invoke_write_handler(ec, p);

            // inform termination detection of a sent message
            if (!p.does_termination_detection())
                hpx::detail::dijkstra_make_black();
        }
    }

    // schedule threads based on given parcel
    void applier::schedule_action(parcelset::parcel p, std::size_t num_thread)
    {
        // fetch the set of destinations
#if !defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        std::size_t const size = 1ul;
#else
        std::size_t const size = p.size();
#endif
        naming::id_type const* ids = p.destinations();
        naming::address const* addrs = p.addrs();

        // decode the action-type in the parcel
        std::unique_ptr<actions::continuation> cont = p.get_continuation();
        actions::base_action * act = p.get_action();

#if defined(HPX_HAVE_SECURITY)
        // we look up the certificate of the originating locality, no matter
        // whether this parcel was routed through another locality or not
        boost::uint32_t locality_id =
            naming::get_locality_id_from_gid(p.get_parcel_id());
        error_code ec(lightweight);
        components::security::signed_certificate const& cert =
            get_locality_certificate(locality_id, ec);

        if (verify_capabilities_ && ec) {
            // we should have received the sender's certificate by now
            HPX_THROW_EXCEPTION(security_error,
                "applier::schedule_action",
                boost::str(boost::format("couldn't extract sender's "
                    "certificate (sender locality id: %1%)") % locality_id));
            return;
        }

        components::security::capability caps_sender;
        if (verify_capabilities_)
            caps_sender = cert.get_type().get_capability();
#endif
        int comptype = act->get_component_type();
        naming::gid_type dest = p.destination_locality();

        // if the parcel carries a continuation it should be directed to a
        // single destination
        HPX_ASSERT(!cont || size == 1);

        naming::resolver_client& client = hpx::naming::get_agas_client();

        // schedule a thread for each of the destinations
        for (std::size_t i = 0; i != size; ++i)
        {
            naming::address const& addr = addrs[i];

            // make sure this parcel destination matches the proper locality
            HPX_ASSERT(dest == addr.locality_);

            // decode the local virtual address of the parcel
            naming::address::address_type lva = addr.address_;

            // by convention, a zero address references either the local
            // runtime support component or one of the AGAS components
            if (0 == lva)
            {
                switch(comptype)
                {
                case components::component_runtime_support:
                    lva = get_runtime_support_raw_gid().get_lsb();
                    break;

                case components::component_agas_primary_namespace:
                    lva = get_agas_client().get_primary_ns_lva();
                    break;

                case components::component_agas_symbol_namespace:
                    lva = get_agas_client().get_symbol_ns_lva();
                    break;

                case components::component_plain_function:
                    break;

                default:
                    HPX_ASSERT(false);
                }
            }
            else if (comptype == components::component_memory)
            {
                HPX_ASSERT(naming::refers_to_virtual_memory(ids[i].get_gid()));
                lva = get_memory_raw_gid().get_lsb();
            }

            // make sure the target has not been migrated away
            auto r = act->was_object_migrated(ids[i], lva);
            if (r.first)
            {
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
                // it's unclear at this point what could be done if there is
                // more than one destination
                HPX_ASSERT(size == 1);
#endif
                // set continuation in outgoing parcel
                if (cont)
                    p.set_continuation(std::move(cont));

                // route parcel to new locality of target
                client.route(
                    std::move(p),
                    util::bind(&detail::parcel_sent_handler,
                        boost::ref(parcel_handler_),
                        util::placeholders::_1, util::placeholders::_2),
                    threads::thread_priority_normal);
                break;
            }

#if defined(HPX_HAVE_SECURITY)
            if (verify_capabilities_) {
                components::security::capability caps_action =
                    act->get_required_capabilities(lva);

                if (caps_action.verify(caps_sender) == false) {
                    HPX_THROW_EXCEPTION(security_error,
                        "applier::schedule_action",
                        boost::str(boost::format("sender has insufficient capabilities "
                            "to execute the action (%1%, sender: %2%, action %3%)") %
                            act->get_action_name() % caps_sender % caps_action));
                    return;
                }
            }
#endif
            // make sure the component_type of the action matches the
            // component type in the destination address
            if (HPX_UNLIKELY(!components::types_are_compatible(
                addr.type_, comptype)))
            {
                std::ostringstream strm;
                strm << " types are not compatible: destination_type("
                      << addr.type_ << ") action_type(" << comptype
                      << ") parcel ("  << p << ")";
                HPX_THROW_EXCEPTION(bad_component_type,
                    "applier::schedule_action",
                    strm.str());
            }

            // dispatch action, register work item either with or without
            // continuation support
            if (!cont) {
                // No continuation is to be executed, register the plain
                // action and the local-virtual address.
                act->schedule_thread(ids[i], lva, threads::pending, num_thread);
            }
            else {
                // This parcel carries a continuation, register a wrapper
                // which first executes the original thread function as
                // required by the action and triggers the continuations
                // afterwards.
                act->schedule_thread(std::move(cont), ids[i], lva,
                    threads::pending, num_thread);
            }
        }
    }

    applier& get_applier()
    {
        // should have been initialized
        HPX_ASSERT(NULL != applier::applier_.get());
        return **applier::applier_;
    }

    applier* get_applier_ptr()
    {
        applier** appl = applier::applier_.get();
        return appl ? *appl : NULL;
    }

    // The function \a get_locality_id returns the id of this locality
    boost::uint32_t get_locality_id(error_code& ec) //-V659
    {
        applier** appl = applier::applier_.get();
        return appl ? (*appl)->get_locality_id(ec) : naming::invalid_locality_id;
    }
}}

