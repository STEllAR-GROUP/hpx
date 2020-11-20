////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2017 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/naming/split_gid.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/timing/scoped_timer.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/regex_from_pattern.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
#include <vector>

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

// register all performance counter types exposed by this component
void symbol_namespace::register_counter_types(
    error_code& ec
    )
{
    performance_counters::create_counter_func creator(
        util::bind_back(&performance_counters::agas_raw_counter_creator
      , agas::server::symbol_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_symbol_namespace_services;
          ++i)
    {
        // global counters are handled elsewhere
        if (detail::symbol_namespace_services[i].code_ ==
            symbol_ns_statistics_counter)
            continue;

        std::string name(detail::symbol_namespace_services[i].name_);
        std::string help;
        performance_counters::counter_type type;
        std::string::size_type p = name.find_last_of('/');
        HPX_ASSERT(p != std::string::npos);

        if (detail::symbol_namespace_services[i].target_ ==
            detail::counter_target_count)
        {
            help = hpx::util::format(
                "returns the number of invocations of the AGAS service '{}'",
                name.substr(p+1));
            type = performance_counters::counter_monotonically_increasing;
        }
        else
        {
            help = hpx::util::format(
                "returns the overall execution time of the AGAS service '{}'",
                name.substr(p+1));
            type = performance_counters::counter_elapsed_time;
        }

        performance_counters::install_counter_type(
            agas::performance_counter_basename + name
          , type
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
        util::bind_back(&performance_counters::agas_raw_counter_creator
      , agas::server::symbol_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_symbol_namespace_services;
          ++i)
    {
        // local counters are handled elsewhere
        if (detail::symbol_namespace_services[i].code_ !=
            symbol_ns_statistics_counter)
            continue;

        std::string help;
        performance_counters::counter_type type;
        if (detail::symbol_namespace_services[i].target_ ==
            detail::counter_target_count)
        {
            help = "returns the overall number of invocations of all symbol "
                   "AGAS services";
            type = performance_counters::counter_monotonically_increasing;
        }
        else
        {
            help = "returns the overall execution time of all symbol AGAS "
                   "services";
            type = performance_counters::counter_elapsed_time;
        }

        performance_counters::install_counter_type(
            std::string(agas::performance_counter_basename) +
                detail::symbol_namespace_services[i].name_
          , type
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
  , std::uint32_t locality_id
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
    agas::register_name(launch::sync, instance_name_,
        get_unmanaged_id().get_gid(), ec);
}

void symbol_namespace::unregister_server_instance(
    error_code& ec
    )
{
    agas::unregister_name(launch::sync, instance_name_, ec);
    this->base_type::finalize();
}

void symbol_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name(launch::sync, instance_name_, ec);
    }
}

bool symbol_namespace::bind(
    std::string key
  , naming::gid_type gid
    )
{ // {{{ bind implementation
    // parameters
    util::scoped_timer<std::atomic<std::int64_t> > update(
        counter_data_.bind_.time_,
        counter_data_.bind_.enabled_
    );
    counter_data_.increment_bind_count();

    std::unique_lock<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it != end)
    {
        std::int64_t const credits = naming::detail::get_credit_from_gid(gid);
        naming::gid_type raw_gid = *(it->second);

        naming::detail::strip_internal_bits_from_gid(raw_gid);
        naming::detail::strip_internal_bits_from_gid(gid);

        // increase reference count
        if (raw_gid == gid)
        {
            LAGAS_(info) << hpx::util::format(
                "symbol_namespace::bind, key({1}), gid({2}), old_credit({3}), "
                "new_credit({4})",
                key, gid,
                naming::detail::get_credit_from_gid(*(it->second)),
                naming::detail::get_credit_from_gid(*(it->second)) + credits);

            // REVIEW: do we need to add the credit of the argument to the table?
            naming::detail::add_credit_to_gid(*(it->second), credits);

            return true;
        }

        if (LAGAS_ENABLED(info))
        {
            naming::detail::add_credit_to_gid(gid, credits);
            LAGAS_(info) << hpx::util::format(
                "symbol_namespace::bind, key({1}), gid({2}), response(no_success)",
                key, gid);
        }

        return false;
    }

    if (HPX_UNLIKELY(!util::insert_checked(gids_.insert(
            std::make_pair(key, std::make_shared<naming::gid_type>(gid))))))
    {
        l.unlock();

        HPX_THROW_EXCEPTION(lock_error
          , "symbol_namespace::bind"
          , "GID table insertion failed due to a locking error or "
            "memory corruption");
    }

    // handle registered events
    typedef on_event_data_map_type::iterator iterator;
    std::pair<iterator, iterator> p = on_event_data_.equal_range(key);

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

                HPX_THROW_EXCEPTION(invalid_status
                  , "symbol_namespace::bind"
                  , "unable to re-locate the entry in the GID table");
            }

            // hold on to the gid while the map is unlocked
            std::shared_ptr<naming::gid_type> current_gid = gid_it->second;

            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);

                // split the credit as the receiving end will expect to keep the
                // object alive
                naming::gid_type new_gid =
                    naming::detail::split_gid_if_needed(*current_gid).get();

                // trigger the lco
                set_lco_value(id, std::move(new_gid));
            }
        }
    }

    l.unlock();

    LAGAS_(info) << hpx::util::format(
        "symbol_namespace::bind, key({1}), gid({2})",
        key, gid);

    return true;
} // }}}

naming::gid_type symbol_namespace::resolve(std::string const& key)
{ // {{{ resolve implementation
    // parameters
    util::scoped_timer<std::atomic<std::int64_t> > update(
        counter_data_.resolve_.time_,
        counter_data_.resolve_.enabled_
    );
    counter_data_.increment_resolve_count();

    std::unique_lock<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << hpx::util::format(
            "symbol_namespace::resolve, key({1}), response(no_success)",
            key);

        return naming::invalid_gid;
    }

    // hold on to gid before unlocking the map
    std::shared_ptr<naming::gid_type> current_gid(it->second);

    l.unlock();
    naming::gid_type gid = naming::detail::split_gid_if_needed(*current_gid).get();

    LAGAS_(info) << hpx::util::format(
        "symbol_namespace::resolve, key({1}), gid({2})",
        key, gid);

    return gid;
} // }}}

naming::gid_type symbol_namespace::unbind(std::string const& key)
{ // {{{ unbind implementation
    util::scoped_timer<std::atomic<std::int64_t> > update(
        counter_data_.unbind_.time_,
        counter_data_.unbind_.enabled_
    );
    counter_data_.increment_unbind_count();

    std::lock_guard<mutex_type> l(mutex_);

    gid_table_type::iterator it = gids_.find(key);
    gid_table_type::iterator end = gids_.end();

    if (it == end)
    {
        LAGAS_(info) << hpx::util::format(
            "symbol_namespace::unbind, key({1}), response(no_success)",
            key);

        return naming::invalid_gid;
    }

    naming::gid_type const gid = *(it->second);

    gids_.erase(it);

    LAGAS_(info) << hpx::util::format(
        "symbol_namespace::unbind, key({1}), gid({2})",
        key, gid);

    return gid;
} // }}}

// TODO: catch exceptions
symbol_namespace::iterate_names_return_type symbol_namespace::iterate(
    std::string const& pattern)
{ // {{{ iterate implementation
    util::scoped_timer<std::atomic<std::int64_t> > update(
        counter_data_.iterate_names_.time_,
        counter_data_.iterate_names_.enabled_
    );
    counter_data_.increment_iterate_names_count();

    std::map<std::string, naming::gid_type> found;

    if (pattern.find_first_of("*?[]") != std::string::npos)
    {
        std::string str_rx(util::regex_from_pattern(pattern, throws));
        std::regex rx(str_rx);

        std::unique_lock<mutex_type> l(mutex_);
        for (gid_table_type::iterator it = gids_.begin(); it != gids_.end();
             ++it)
        {
            if (!std::regex_match(it->first, rx))
                continue;

            // hold on to entry while map is unlocked
            std::shared_ptr<naming::gid_type> current_gid(it->second);
            util::unlock_guard<std::unique_lock<mutex_type> > ul(l);

            found[it->first] =
                naming::detail::split_gid_if_needed(*current_gid).get();
        }
    }
    else
    {
        std::unique_lock<mutex_type> l(mutex_);
        for (gid_table_type::iterator it = gids_.begin(); it != gids_.end();
             ++it)
        {
            if (!pattern.empty() && pattern != it->first)
                continue;

            // hold on to entry while map is unlocked
            std::shared_ptr<naming::gid_type> current_gid(it->second);
            util::unlock_guard<std::unique_lock<mutex_type> > ul(l);

            found[it->first] =
                naming::detail::split_gid_if_needed(*current_gid).get();
        }
    }

    LAGAS_(info) << "symbol_namespace::iterate";

    return found;
} // }}}

bool symbol_namespace::on_event(
    std::string const& name
  , bool call_for_past_events
  , hpx::id_type lco
    )
{ // {{{ on_event implementation
    util::scoped_timer<std::atomic<std::int64_t> > update(
        counter_data_.on_event_.time_,
        counter_data_.on_event_.enabled_
    );
    counter_data_.increment_on_event_count();

    std::unique_lock<mutex_type> l(mutex_);

    bool handled = false;
    if (call_for_past_events)
    {
        gid_table_type::iterator it = gids_.find(name);
        if (it != gids_.end())
        {
            // hold on to entry while map is unlocked
            std::shared_ptr<naming::gid_type> current_gid(it->second);

            // split the credit as the receiving end will expect to keep the
            // object alive
            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                naming::gid_type new_gid = naming::detail::split_gid_if_needed(
                    *current_gid).get();

                // trigger the lco
                handled = true;

                // trigger LCO as name is already bound to an id
                set_lco_value(lco, std::move(new_gid));
            }
        }
    }

    if (!handled)
    {
        on_event_data_map_type::iterator it = on_event_data_.insert(
            on_event_data_map_type::value_type(std::move(name), lco));

        // This overload of insert always returns the iterator pointing
        // to the inserted value. It should never point to end
        HPX_ASSERT(it != on_event_data_.end());
        HPX_UNUSED(it);
    }
    l.unlock();

    LAGAS_(info) << "symbol_namespace::on_event";

    return true;
} // }}}

naming::gid_type symbol_namespace::statistics_counter(std::string const& name)
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "symbol_namespace::statistics_counter";

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p);

    if (p.objectname_ != "agas")
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "symbol_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
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
        HPX_THROW_EXCEPTION(bad_parameter,
            "symbol_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
    }

    typedef symbol_namespace::counter_data cd;

    util::function_nonser<std::int64_t(bool)> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case symbol_ns_bind:
            get_data_func = util::bind_front(&cd::get_bind_count,
                &counter_data_);
            counter_data_.bind_.enabled_ = true;
            break;
        case symbol_ns_resolve:
            get_data_func = util::bind_front(&cd::get_resolve_count,
                &counter_data_);
            counter_data_.resolve_.enabled_ = true;
            break;
        case symbol_ns_unbind:
            get_data_func = util::bind_front(&cd::get_unbind_count,
                &counter_data_);
            counter_data_.unbind_.enabled_ = true;
            break;
        case symbol_ns_iterate_names:
            get_data_func = util::bind_front(&cd::get_iterate_names_count,
                &counter_data_);
            counter_data_.iterate_names_.enabled_ = true;
            break;
        case symbol_ns_on_event:
            get_data_func = util::bind_front(&cd::get_on_event_count,
                &counter_data_);
            counter_data_.on_event_.enabled_ = true;
            break;
        case symbol_ns_statistics_counter:
            get_data_func = util::bind_front(&cd::get_overall_count,
                &counter_data_);
            counter_data_.enable_all();
            break;
        default:
            HPX_THROW_EXCEPTION(bad_parameter
              , "symbol_namespace::statistics"
              , "bad action code while querying statistics");
        }
    }
    else {
        HPX_ASSERT(detail::counter_target_time == target);
        switch (code) {
        case symbol_ns_bind:
            get_data_func = util::bind_front(&cd::get_bind_time,
                &counter_data_);
            counter_data_.bind_.enabled_ = true;
            break;
        case symbol_ns_resolve:
            get_data_func = util::bind_front(&cd::get_resolve_time,
                &counter_data_);
            counter_data_.resolve_.enabled_ = true;
            break;
        case symbol_ns_unbind:
            get_data_func = util::bind_front(&cd::get_unbind_time,
                &counter_data_);
            counter_data_.unbind_.enabled_ = true;
            break;
        case symbol_ns_iterate_names:
            get_data_func = util::bind_front(&cd::get_iterate_names_time,
                &counter_data_);
            counter_data_.iterate_names_.enabled_ = true;
            break;
        case symbol_ns_on_event:
            get_data_func = util::bind_front(&cd::get_on_event_time,
                &counter_data_);
            counter_data_.on_event_.enabled_ = true;
            break;
        case symbol_ns_statistics_counter:
            get_data_func = util::bind_front(&cd::get_overall_time,
                &counter_data_);
            counter_data_.enable_all();
            break;
        default:
            HPX_THROW_EXCEPTION(bad_parameter
              , "symbol_namespace::statistics"
              , "bad action code while querying statistics");
        }
    }

    performance_counters::counter_info info;
    performance_counters::get_counter_type(name, info);

    performance_counters::complement_counter_info(info);

    using performance_counters::detail::create_raw_counter;
    naming::gid_type gid = create_raw_counter(info, get_data_func, hpx::throws);
    return naming::detail::strip_credits_from_gid(gid);
} // }}}

// access current counter values
std::int64_t symbol_namespace::counter_data::get_bind_count(bool reset)
{
    return util::get_and_reset_value(bind_.count_, reset);
}

std::int64_t symbol_namespace::counter_data::get_resolve_count(bool reset)
{
    return util::get_and_reset_value(resolve_.count_, reset);
}

std::int64_t symbol_namespace::counter_data::get_unbind_count(bool reset)
{
    return util::get_and_reset_value(unbind_.count_, reset);
}

std::int64_t symbol_namespace::counter_data::get_iterate_names_count(bool reset)
{
    return util::get_and_reset_value(iterate_names_.count_, reset);
}

std::int64_t symbol_namespace::counter_data::get_on_event_count(bool reset)
{
    return util::get_and_reset_value(on_event_.count_, reset);
}

std::int64_t symbol_namespace::counter_data::get_overall_count(bool reset)
{
    return util::get_and_reset_value(bind_.count_, reset) +
        util::get_and_reset_value(resolve_.count_, reset) +
        util::get_and_reset_value(unbind_.count_, reset) +
        util::get_and_reset_value(iterate_names_.count_, reset) +
        util::get_and_reset_value(on_event_.count_, reset);
}

void symbol_namespace::counter_data::enable_all()
{
    bind_.enabled_ = true;
    resolve_.enabled_ = true;
    unbind_.enabled_ = true;
    iterate_names_.enabled_ = true;
    on_event_.enabled_ = true;
}

// access execution time counters
std::int64_t symbol_namespace::counter_data::get_bind_time(bool reset)
{
    return util::get_and_reset_value(bind_.time_, reset);
}

std::int64_t symbol_namespace::counter_data::get_resolve_time(bool reset)
{
    return util::get_and_reset_value(resolve_.time_, reset);
}

std::int64_t symbol_namespace::counter_data::get_unbind_time(bool reset)
{
    return util::get_and_reset_value(unbind_.time_, reset);
}

std::int64_t symbol_namespace::counter_data::get_iterate_names_time(bool reset)
{
    return util::get_and_reset_value(iterate_names_.time_, reset);
}

std::int64_t symbol_namespace::counter_data::get_on_event_time(bool reset)
{
    return util::get_and_reset_value(on_event_.time_, reset);
}

std::int64_t symbol_namespace::counter_data::get_overall_time(bool reset)
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
    if (bind_.enabled_)
    {
        ++bind_.count_;
    }
}

void symbol_namespace::counter_data::increment_resolve_count()
{
    if (resolve_.enabled_)
    {
        ++resolve_.count_;
    }
}

void symbol_namespace::counter_data::increment_unbind_count()
{
    if (unbind_.enabled_)
    {
        ++unbind_.count_;
    }
}

void symbol_namespace::counter_data::increment_iterate_names_count()
{
    if (iterate_names_.enabled_)
    {
        ++iterate_names_.count_;
    }
}

void symbol_namespace::counter_data::increment_on_event_count()
{
    if (on_event_.enabled_)
    {
        ++on_event_.count_;
    }
}

}}}

