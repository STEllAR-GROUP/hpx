////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/apply.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>

namespace hpx { namespace agas
{

naming::gid_type bootstrap_symbol_namespace_gid()
{
    return naming::gid_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB);
}

naming::id_type bootstrap_symbol_namespace_id()
{
    return naming::id_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB
      , naming::id_type::unmanaged);
}

namespace server
{

// TODO: This isn't scalable, we have to update it every time we add a new
// AGAS request/response type.
response symbol_namespace::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    switch (req.get_action_code())
    {
        case symbol_ns_bind:
            {
                update_time_on_exit update(
                    counter_data_.bind_.time_
                );
                counter_data_.increment_bind_count();
                return bind(req, ec);
            }
        case symbol_ns_resolve:
            {
                update_time_on_exit update(
                    counter_data_.resolve_.time_
                );
                counter_data_.increment_resolve_count();
                return resolve(req, ec);
            }
        case symbol_ns_unbind:
            {
                update_time_on_exit update(
                    counter_data_.unbind_.time_
                );
                counter_data_.increment_unbind_count();
                return unbind(req, ec);
            }
        case symbol_ns_iterate_names:
            {
                update_time_on_exit update(
                    counter_data_.iterate_names_.time_
                );
                counter_data_.increment_iterate_names_count();
                return iterate(req, ec);
            }
        case symbol_ns_on_event:
            {
                update_time_on_exit update(
                    counter_data_.on_event_.time_
                );
                counter_data_.increment_on_event_count();
                return on_event(req, ec);
            }
        case symbol_ns_statistics_counter:
            return statistics_counter(req, ec);

        case locality_ns_allocate:
        case locality_ns_free:
        case locality_ns_localities:
        case locality_ns_num_localities:
        case locality_ns_num_threads:
        case locality_ns_resolve_locality:
        case locality_ns_resolved_localities:
        {
            LAGAS_(warning) <<
                "symbol_namespace::service, redirecting request to "
                "locality_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case primary_ns_route:
        case primary_ns_bind_gid:
        case primary_ns_resolve_gid:
        case primary_ns_unbind_gid:
        case primary_ns_increment_credit:
        case primary_ns_decrement_credit:
        case primary_ns_allocate:
        case primary_ns_begin_migration:
        case primary_ns_end_migration:
        {
            LAGAS_(warning) <<
                "symbol_namespace::service, redirecting request to "
                "primary_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case component_ns_bind_prefix:
        case component_ns_bind_name:
        case component_ns_resolve_id:
        case component_ns_unbind_name:
        case component_ns_iterate_types:
        case component_ns_get_component_type_name:
        case component_ns_num_localities:
        {
            LAGAS_(warning) <<
                "symbol_namespace::service, redirecting request to "
                "component_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case locality_ns_service:
        case component_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "symbol_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    };
} // }}}

// register all performance counter types exposed by this component
void symbol_namespace::register_counter_types(
    error_code& ec
    )
{
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    performance_counters::create_counter_func creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::symbol_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_symbol_namespace_services;
          ++i)
    {
        // global counters are handled elsewhere
        if (detail::symbol_namespace_services[i].code_ == symbol_ns_statistics_counter)
            continue;

        std::string name(detail::symbol_namespace_services[i].name_);
        std::string help;
        std::string::size_type p = name.find_last_of('/');
        HPX_ASSERT(p != std::string::npos);

        if (detail::symbol_namespace_services[i].target_ == detail::counter_target_count)
            help = boost::str(help_count % name.substr(p+1));
        else
            help = boost::str(help_time % name.substr(p+1));

        performance_counters::install_counter_type(
            agas::performance_counter_basename + name
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::symbol_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void symbol_namespace::register_global_counter_types(
    error_code& ec
    )
{
    performance_counters::create_counter_func creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::symbol_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_symbol_namespace_services;
          ++i)
    {
        // local counters are handled elsewhere
        if (detail::symbol_namespace_services[i].code_ != symbol_ns_statistics_counter)
            continue;

        std::string help;
        if (detail::symbol_namespace_services[i].target_ == detail::counter_target_count)
            help = "returns the overall number of invocations \
                    of all symbol AGAS services";
        else
            help = "returns the overall execution time of all symbol AGAS services";

        performance_counters::install_counter_type(
            std::string(agas::performance_counter_basename) +
                detail::symbol_namespace_services[i].name_
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::symbol_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void symbol_namespace::register_server_instance(
    char const* servicename
  , boost::uint32_t locality_id
  , error_code& ec
    )
{
    // set locality_id for this component
    if (locality_id == naming::invalid_locality_id)
        locality_id = 0;        // if not given, we're on the root

    this->base_type::set_locality_id(locality_id);

    // now register this AGAS instance with AGAS :-P
    instance_name_ = agas::service_name;
    instance_name_ += servicename;
    instance_name_ += agas::server::symbol_namespace_service_name;

    // register a gid (not the id) to avoid AGAS holding a reference to this
    // component
    agas::register_name_sync(instance_name_, get_unmanaged_id().get_gid(), ec);
}

void symbol_namespace::unregister_server_instance(
    error_code& ec
    )
{
    agas::unregister_name_sync(instance_name_, ec);
    this->base_type::finalize();
}

void symbol_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name_sync(instance_name_, ec);
    }
}

// TODO: do/undo semantics (e.g. transactions)
std::vector<response> symbol_namespace::bulk_service(
    std::vector<request> const& reqs
  , error_code& ec
    )
{
    std::vector<response> r;
    r.reserve(reqs.size());

    for (request const& req : reqs)
    {
        error_code ign;
        r.push_back(service(req, ign));
    }

    return r;
}

response symbol_namespace::bind(
    request const& req
  , error_code& ec
    )
{ // {{{ bind implementation
    // parameters
    std::string key = req.get_name();
    naming::gid_type gid = req.get_gid();

    boost::unique_lock<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it != end)
    {
        boost::int64_t const credits = naming::detail::get_credit_from_gid(gid);
        naming::gid_type raw_gid = *(it->second);

        naming::detail::strip_internal_bits_from_gid(raw_gid);
        naming::detail::strip_internal_bits_from_gid(gid);

        // increase reference count
        if (raw_gid == gid)
        {
            LAGAS_(info) << (boost::format(
                "symbol_namespace::bind, key(%1%), gid(%2%), old_credit(%3%), "
                "new_credit(%4%)")
                % key % gid
                % naming::detail::get_credit_from_gid(*(it->second))
                % (naming::detail::get_credit_from_gid(*(it->second)) + credits));

            // REVIEW: do we need to add the credit of the argument to the table?
            naming::detail::add_credit_to_gid(*(it->second), credits);

            if (&ec != &throws)
                ec = make_success_code();

            return response(symbol_ns_bind);
        }

        if (LAGAS_ENABLED(info))
        {
            naming::detail::add_credit_to_gid(gid, credits);
            LAGAS_(info) << (boost::format(
                "symbol_namespace::bind, key(%1%), gid(%2%), response(no_success)")
                % key % gid);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_bind, no_success);
    }

    if (HPX_UNLIKELY(!util::insert_checked(gids_.insert(
            std::make_pair(key, boost::make_shared<naming::gid_type>(gid))))))
    {
        l.unlock();

        HPX_THROWS_IF(ec, lock_error
          , "symbol_namespace::bind"
          , "GID table insertion failed due to a locking error or "
            "memory corruption");
        return response();
    }

    // handle registered events
    typedef on_event_data_map_type::iterator iterator;
    std::pair<std::string, namespace_action_code> evtkey(key, symbol_ns_bind);
    std::pair<iterator, iterator> p = on_event_data_.equal_range(evtkey);

    std::vector<hpx::id_type> lcos;
    if (p.first != p.second)
    {
        iterator it = p.first;
        while (it != p.second)
        {
            lcos.push_back((*it).second);
            ++it;
        }

        on_event_data_.erase(p.first, p.second);

        // notify all LCOS which were registered with this name
        for (hpx::id_type const& id : lcos)
        {
            // re-locate the entry in the GID table for each LCO anew, as we
            // need to unlock the mutex protecting the table for each iteration
            // below
            gid_table_type::iterator gid_it = gids_.find(key);
            if (gid_it == gids_.end())
            {
                l.unlock();

                HPX_THROWS_IF(ec, invalid_status
                  , "symbol_namespace::bind"
                  , "unable to re-locate the entry in the GID table");
                return response();
            }

            // hold on to the gid while the map is unlocked
            boost::shared_ptr<naming::gid_type> current_gid = gid_it->second;

            {
                util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);

                // split the credit as the receiving end will expect to keep the
                // object alive
                naming::gid_type new_gid =
                    naming::detail::split_gid_if_needed(*current_gid);

                // trigger the lco
                set_lco_value(id, new_gid);
            }
        }
    }

    l.unlock();

    LAGAS_(info) << (boost::format(
        "symbol_namespace::bind, key(%1%), gid(%2%)")
        % key % gid);

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_bind);
} // }}}

response symbol_namespace::resolve(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve implementation
    // parameters
    std::string key = req.get_name();

    boost::unique_lock<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "symbol_namespace::resolve, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_resolve
                      , naming::invalid_gid
                      , no_success);
    }

    if (&ec != &throws)
        ec = make_success_code();

    // hold on to gid before unlocking the map
    boost::shared_ptr<naming::gid_type> current_gid(it->second);

    l.unlock();
    naming::gid_type gid = naming::detail::split_gid_if_needed(*current_gid);

    LAGAS_(info) << (boost::format(
        "symbol_namespace::resolve, key(%1%), gid(%2%)")
        % key % gid);

    return response(symbol_ns_resolve, gid);
} // }}}

response symbol_namespace::unbind(
    request const& req
  , error_code& ec
    )
{ // {{{ unbind implementation
    // parameters
    std::string key = req.get_name();

    boost::lock_guard<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "symbol_namespace::unbind, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(symbol_ns_unbind, naming::invalid_gid, no_success);
    }

    naming::gid_type const gid = *(it->second);

    gids_.erase(it);

    LAGAS_(info) << (boost::format(
        "symbol_namespace::unbind, key(%1%), gid(%2%)")
        % key % gid);

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_unbind, gid);
} // }}}

// TODO: catch exceptions
response symbol_namespace::iterate(
    request const& req
  , error_code& ec
    )
{ // {{{ iterate implementation
    iterate_names_function_type f = req.get_iterate_names_function();

    boost::unique_lock<mutex_type> l(mutex_);

    for (gid_table_type::iterator it = gids_.begin(); it != gids_.end(); ++it)
    {
        std::string key(it->first);
        naming::gid_type gid = *(it->second);

        {
            util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
            f(key, gid);
        }

        // re-locate current entry
        it = gids_.find(key);
    }

    LAGAS_(info) << "symbol_namespace::iterate";

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_iterate_names);
} // }}}

response symbol_namespace::on_event(
    request const& req
  , error_code& ec
    )
{ // {{{ on_event implementation
    std::string name = req.get_name();
    namespace_action_code evt = req.get_on_event_event();
    bool call_for_past_events = req.get_on_event_call_for_past_event();
    hpx::id_type lco = req.get_on_event_result_lco();

    if (evt != symbol_ns_bind)
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "addressing_service::on_symbol_namespace_event",
            "invalid event type");
        return response(symbol_ns_on_event, no_success);
    }

    boost::unique_lock<mutex_type> l(mutex_);

    bool handled = false;
    if (call_for_past_events)
    {
        gid_table_type::iterator it = gids_.find(name);
        if (it != gids_.end())
        {
            // hold on to entry while map is unlocked
            boost::shared_ptr<naming::gid_type> current_gid(it->second);

            // split the credit as the receiving end will expect to keep the
            // object alive
            {
                util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                naming::gid_type new_gid = naming::detail::split_gid_if_needed(
                    *current_gid);

                // trigger the lco
                handled = true;

                // trigger LCO as name is already bound to an id
                set_lco_value(lco, new_gid);
            }
        }
    }

    if (!handled)
    {
        std::pair<std::string, namespace_action_code> key(name, evt);
        on_event_data_map_type::iterator it = on_event_data_.insert(
            on_event_data_map_type::value_type(std::move(key), lco));

        l.unlock();

        if (it == on_event_data_.end())
        {
            LAGAS_(info) << (boost::format(
                "symbol_namespace::on_event, name(%1%), response(no_success)")
                % name);

            if (&ec != &throws)
                ec = make_success_code();

            return response(symbol_ns_on_event, no_success);
        }
    }
    else
    {
        l.unlock();
    }

    LAGAS_(info) << "symbol_namespace::on_event";

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_on_event);
} // }}}

response symbol_namespace::statistics_counter(
    request const& req
  , error_code& ec
    )
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "symbol_namespace::statistics_counter";

    std::string name(req.get_statistics_counter_name());

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p, ec);
    if (ec) return response();

    if (p.objectname_ != "agas")
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "symbol_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    namespace_action_code code = invalid_request;
    detail::counter_target target = detail::counter_target_invalid;
    for (std::size_t i = 0;
          i != detail::num_symbol_namespace_services;
          ++i)
    {
        if (p.countername_ == detail::symbol_namespace_services[i].name_)
        {
            code = detail::symbol_namespace_services[i].code_;
            target = detail::symbol_namespace_services[i].target_;
            break;
        }
    }

    if (code == invalid_request || target == detail::counter_target_invalid)
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "symbol_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    typedef symbol_namespace::counter_data cd;

    util::function_nonser<boost::int64_t(bool)> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case symbol_ns_bind:
            get_data_func = boost::bind(&cd::get_bind_count, &counter_data_, ::_1);
            break;
        case symbol_ns_resolve:
            get_data_func = boost::bind(&cd::get_resolve_count, &counter_data_, ::_1);
            break;
        case symbol_ns_unbind:
            get_data_func = boost::bind(&cd::get_unbind_count, &counter_data_, ::_1);
            break;
        case symbol_ns_iterate_names:
            get_data_func = boost::bind(&cd::get_iterate_names_count,
                &counter_data_, ::_1);
            break;
        case symbol_ns_on_event:
            get_data_func = boost::bind(&cd::get_on_event_count, &counter_data_, ::_1);
            break;
        case symbol_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_count, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "symbol_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }
    else {
        HPX_ASSERT(detail::counter_target_time == target);
        switch (code) {
        case symbol_ns_bind:
            get_data_func = boost::bind(&cd::get_bind_time, &counter_data_, ::_1);
            break;
        case symbol_ns_resolve:
            get_data_func = boost::bind(&cd::get_resolve_time, &counter_data_, ::_1);
            break;
        case symbol_ns_unbind:
            get_data_func = boost::bind(&cd::get_unbind_time, &counter_data_, ::_1);
            break;
        case symbol_ns_iterate_names:
            get_data_func = boost::bind(&cd::get_iterate_names_time,
                &counter_data_, ::_1);
            break;
        case symbol_ns_on_event:
            get_data_func = boost::bind(&cd::get_on_event_time, &counter_data_, ::_1);
            break;
        case symbol_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_time, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "symbol_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }

    performance_counters::counter_info info;
    performance_counters::get_counter_type(name, info, ec);
    if (ec) return response();

    performance_counters::complement_counter_info(info, ec);
    if (ec) return response();

    using performance_counters::detail::create_raw_counter;
    naming::gid_type gid = create_raw_counter(info, get_data_func, ec);
    if (ec) return response();

    if (&ec != &throws)
        ec = make_success_code();

    return response(symbol_ns_statistics_counter, gid);
} // }}}

// access current counter values
boost::int64_t symbol_namespace::counter_data::get_bind_count(bool reset)
{
    return util::get_and_reset_value(bind_.count_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_resolve_count(bool reset)
{
    return util::get_and_reset_value(resolve_.count_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_unbind_count(bool reset)
{
    return util::get_and_reset_value(unbind_.count_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_iterate_names_count(bool reset)
{
    return util::get_and_reset_value(iterate_names_.count_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_on_event_count(bool reset)
{
    return util::get_and_reset_value(on_event_.count_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_overall_count(bool reset)
{
    return util::get_and_reset_value(bind_.count_, reset) +
        util::get_and_reset_value(resolve_.count_, reset) +
        util::get_and_reset_value(unbind_.count_, reset) +
        util::get_and_reset_value(iterate_names_.count_, reset) +
        util::get_and_reset_value(on_event_.count_, reset);
}

// access execution time counters
boost::int64_t symbol_namespace::counter_data::get_bind_time(bool reset)
{
    return util::get_and_reset_value(bind_.time_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_resolve_time(bool reset)
{
    return util::get_and_reset_value(resolve_.time_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_unbind_time(bool reset)
{
    return util::get_and_reset_value(unbind_.time_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_iterate_names_time(bool reset)
{
    return util::get_and_reset_value(iterate_names_.time_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_on_event_time(bool reset)
{
    return util::get_and_reset_value(on_event_.time_, reset);
}

boost::int64_t symbol_namespace::counter_data::get_overall_time(bool reset)
{
    return util::get_and_reset_value(bind_.time_, reset) +
        util::get_and_reset_value(resolve_.time_, reset) +
        util::get_and_reset_value(unbind_.time_, reset) +
        util::get_and_reset_value(iterate_names_.time_, reset) +
        util::get_and_reset_value(on_event_.time_, reset);
}

// increment counter values
void symbol_namespace::counter_data::increment_bind_count()
{
    ++bind_.count_;
}

void symbol_namespace::counter_data::increment_resolve_count()
{
    ++resolve_.count_;
}

void symbol_namespace::counter_data::increment_unbind_count()
{
    ++unbind_.count_;
}

void symbol_namespace::counter_data::increment_iterate_names_count()
{
    ++iterate_names_.count_;
}

void symbol_namespace::counter_data::increment_on_event_count()
{
    ++on_event_.count_;
}

}}}

