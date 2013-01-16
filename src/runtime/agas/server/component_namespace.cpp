////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/include/performance_counters.hpp>

#include <boost/foreach.hpp>

// TODO: Remove the use of the name "prefix"

namespace hpx { namespace agas
{

naming::gid_type bootstrap_component_namespace_gid()
{
    return naming::gid_type
        (HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB);
}

naming::id_type bootstrap_component_namespace_id()
{
    return naming::id_type
        (bootstrap_component_namespace_gid(), naming::id_type::unmanaged);
}

namespace server
{
// TODO: This isn't scalable, we have to update it every time we add a new
// AGAS request/response type.
response component_namespace::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    switch (req.get_action_code())
    {
        case component_ns_bind_prefix:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.bind_prefix_.time_
                );
                counter_data_.increment_bind_prefix_count();
                return bind_prefix(req, ec);
            }
        case component_ns_bind_name:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.bind_name_.time_
                );
                counter_data_.increment_bind_name_count();
                return bind_name(req, ec);
            }
        case component_ns_resolve_id:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.resolve_id_.time_
                );
                counter_data_.increment_resolve_id_count();
                return resolve_id(req, ec);
            }
        case component_ns_unbind_name:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.unbind_name_.time_
                );
                counter_data_.increment_unbind_name_ount();
                return unbind(req, ec);
            }
        case component_ns_iterate_types:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.iterate_types_.time_
                );
                counter_data_.increment_iterate_types_count();
                return iterate_types(req, ec);
            }
        case component_ns_get_component_type_name:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.get_component_type_name_.time_
                );
                counter_data_.increment_get_component_type_name_count();
                return get_component_type_name(req, ec);
            }
        case component_ns_num_localities:
            {
                update_time_on_exit update(
                    counter_data_
                  , counter_data_.num_localities_.time_
                );
                counter_data_.increment_num_localities_count();
                return get_num_localities(req, ec);
            }
        case component_ns_statistics_counter:
            return statistics_counter(req, ec);

        case primary_ns_allocate:
        case primary_ns_bind_gid:
        case primary_ns_resolve_gid:
        case primary_ns_free:
        case primary_ns_unbind_gid:
        case primary_ns_change_credit_non_blocking:
        case primary_ns_change_credit_sync:
        case primary_ns_localities:
        case primary_ns_num_localities:
        case primary_ns_num_threads:
        case primary_ns_resolved_localities:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "primary_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case symbol_ns_bind:
        case symbol_ns_resolve:
        case symbol_ns_unbind:
        case symbol_ns_iterate_names:
        {
            LAGAS_(warning) <<
                "component_namespace::service, redirecting request to "
                "symbol_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case component_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "component_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    };
} // }}}

// register all performance counter types exposed by this component
void component_namespace::register_counter_types(
    error_code& ec
    )
{
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    HPX_STD_FUNCTION<performance_counters::create_counter_func> creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::component_namespace_service_name));

    for (std::size_t i = 0;
          i < detail::num_component_namespace_services;
          ++i)
    {
        std::string name(detail::component_namespace_services[i].name_);
        std::string help;
        if (detail::component_namespace_services[i].target_ == detail::counter_target_count)
            help = boost::str(help_count % name.substr(name.find_last_of('/')+1));
        else
            help = boost::str(help_time % name.substr(name.find_last_of('/')+1));

        performance_counters::install_counter_type(
            "/agas/" + name
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::default_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::component_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void component_namespace::register_server_instance(
    char const* servicename
  , error_code& ec
    )
{
    // now register this AGAS instance with AGAS :-P
    instance_name_ = agas::service_name;
    instance_name_ += agas::server::component_namespace_service_name;
    instance_name_ += servicename;

    // register a gid (not the id) to avoid AGAS holding a reference to this
    // component
    agas::register_name(instance_name_, get_gid().get_gid(), ec);
}

void component_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name(instance_name_, ec);
    }
}

// TODO: do/undo semantics (e.g. transactions)
std::vector<response> component_namespace::bulk_service(
    std::vector<request> const& reqs
  , error_code& ec
    )
{
    std::vector<response> r;
    r.reserve(reqs.size());

    BOOST_FOREACH(request const& req, reqs)
    {
        error_code ign;
        r.push_back(service(req, ign));
    }

    return r;
}

response component_namespace::bind_prefix(
    request const& req
  , error_code& ec
    )
{ // {{{ bind_prefix implementation
    // parameters
    std::string key = req.get_name();
    boost::uint32_t prefix = req.get_locality_id();

    mutex_type::scoped_lock l(mutex_);

    component_id_table_type::left_map::iterator cit = component_ids_.left.find(key)
                                    , cend = component_ids_.left.end();

    // This is the first request, so we use the type counter, and then
    // increment it.
    if (component_ids_.left.find(key) == cend)
    {
        if (HPX_UNLIKELY(!util::insert_checked(component_ids_.left.insert(
                std::make_pair(key, type_counter)), cit)))
        {
            HPX_THROWS_IF(ec, lock_error
              , "component_namespace::bind_prefix"
              , "component id table insertion failed due to a locking "
                "error or memory corruption")
            return response();
        }

        // If the insertion succeeded, we need to increment the type
        // counter.
        ++type_counter;
    }

    factory_table_type::iterator fit = factories_.find(cit->second)
                               , fend = factories_.end();

    if (fit != fend)
    {
        prefixes_type& prefixes = fit->second;
        prefixes_type::iterator pit = prefixes.find(prefix);

        if (pit != prefixes.end())
        {
            // Duplicate type registration for this locality.
            HPX_THROWS_IF(ec, duplicate_component_id
              , "component_namespace::bind_prefix"
              , boost::str(boost::format(
                    "component id is already registered for the given "
                    "locality, key(%1%), prefix(%2%), ctype(%3%)")
                    % key % prefix % cit->second))
            return response();
        }

        fit->second.insert(prefix);

        // First registration for this locality, we still return no_success to
        // convey the fact that another locality already registered this
        // component type.
        LAGAS_(info) << (boost::format(
            "component_namespace::bind_prefix, key(%1%), prefix(%2%), "
            "ctype(%3%), response(no_success)")
            % key % prefix % cit->second);

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_bind_prefix
                      , cit->second
                      , no_success);
    }

    // Instead of creating a temporary and then inserting it, we insert
    // an empty set, then put the prefix into said set. This should
    // prevent a copy, though most compilers should be able to optimize
    // this without our help.
    if (HPX_UNLIKELY(!util::insert_checked(factories_.insert(
            std::make_pair(cit->second, prefixes_type())), fit)))
    {
        HPX_THROWS_IF(ec, lock_error
            , "component_namespace::bind_prefix"
            , "factory table insertion failed due to a locking "
              "error or memory corruption")
        return response();
    }

    fit->second.insert(prefix);

    LAGAS_(info) << (boost::format(
        "component_namespace::bind_prefix, key(%1%), prefix(%2%), ctype(%3%)")
        % key % prefix % cit->second);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_bind_prefix, cit->second);
} // }}}

response component_namespace::bind_name(
    request const& req
  , error_code& ec
    )
{ // {{{ bind_name implementation
    // parameters
    std::string key = req.get_name();

    mutex_type::scoped_lock l(mutex_);

    component_id_table_type::left_map::iterator it = component_ids_.left.find(key)
                                    , end = component_ids_.left.end();

    // If the name is not in the table, register it (this is only done so
    // we can implement a backwards compatible get_component_id).
    if (it == end)
    {
        if (HPX_UNLIKELY(!util::insert_checked(component_ids_.left.insert(
                std::make_pair(key, type_counter)), it)))
        {
            HPX_THROWS_IF(ec, lock_error
              , "component_namespace::bind_name"
              , "component id table insertion failed due to a locking "
                "error or memory corruption");
            return response();
        }

        // If the insertion succeeded, we need to increment the type
        // counter.
        ++type_counter;
    }

    LAGAS_(info) << (boost::format(
        "component_namespace::bind_name, key(%1%), ctype(%2%)")
        % key % it->second);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_bind_name, it->second);
} // }}}

response component_namespace::resolve_id(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve_id implementation
    // parameters
    component_id_type key = req.get_component_type();

    // If the requested component type is a derived type, use only its derived
    // part for the lookup.
    if (key != components::get_base_type(key))
        key = components::get_derived_type(key);

    mutex_type::scoped_lock l(mutex_);

    factory_table_type::const_iterator it = factories_.find(key)
                                     , end = factories_.end();

    // REVIEW: Should we differentiate between these two cases? Should we
    // throw an exception if it->second.empty()? It should be impossible.
    if (it == end || it->second.empty())
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_id, key(%1%), localities(0)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_resolve_id
                      , std::vector<boost::uint32_t>());
    }

    else
    {
        std::vector<boost::uint32_t> p;

        prefixes_type const& prefixes = it->second;
        prefixes_type::const_iterator pit = prefixes.begin()
                                    , pend = prefixes.end();

        for (; pit != pend; ++pit)
            p.push_back(*pit);

        LAGAS_(info) << (boost::format(
            "component_namespace::resolve_id, key(%1%), localities(%2%)")
            % key % prefixes.size());

        if (&ec != &throws)
            ec = make_success_code();

        return response(component_ns_resolve_id, p);
    }
} // }}}

response component_namespace::unbind(
    request const& req
  , error_code& ec
    )
{ // {{{ unbind implementation
    // parameters
    std::string key = req.get_name();

    mutex_type::scoped_lock l(mutex_);

    component_id_table_type::left_map::iterator it = component_ids_.left.find(key);

    // REVIEW: Should this be an error?
    if (it == component_ids_.left.end())
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::unbind, key(%1%), response(no_success)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

       return response(component_ns_unbind_name, no_success);
    }

    // REVIEW: If there are no localities with this type, should we throw
    // an exception here?
    factories_.erase(it->second);
    component_ids_.left.erase(it);

    LAGAS_(info) << (boost::format(
        "component_namespace::unbind, key(%1%)")
        % key);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_unbind_name);
} // }}}

// TODO: catch exceptions
response component_namespace::iterate_types(
    request const& req
  , error_code& ec
    )
{ // {{{ iterate implementation
    iterate_types_function_type f = req.get_iterate_types_function();

    mutex_type::scoped_lock l(mutex_);

    for (component_id_table_type::left_map::iterator it = component_ids_.left.begin()
                                         , end = component_ids_.left.end();
         it != end; ++it)
    {
        f(it->first, it->second);
    }

    LAGAS_(info) << "component_namespace::iterate_types";

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_iterate_types);
} // }}}

template <typename Map>
std::string get_component_name(Map const& m, components::component_type t)
{
    if (t < components::component_last)
        return components::get_component_type_name(t);

    typename Map::const_iterator it = m.find(t);
    if (it == m.end())
        return "";

    return (*it).second;
}

response component_namespace::get_component_type_name(
    request const& req
  , error_code& ec
    )
{ // {{{ get_component_type_name implementation
    components::component_type t = req.get_component_type();

    mutex_type::scoped_lock l(mutex_);

    std::string result;

    if (t == components::component_invalid) {
        result = "component_invalid";
    }
    else if (components::get_derived_type(t) == 0) {
        result = get_component_name(component_ids_.right, t);
    }
    else if (components::get_derived_type(t) != 0) {
        result = get_component_name(component_ids_.right, components::get_derived_type(t));
        result += "/";
        result += get_component_name(component_ids_.right, components::get_base_type(t));
    }

    if (result.empty())
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::get_component_typename, key(%1%), response(no_success)")
            % int(t));

        if (&ec != &throws)
            ec = make_success_code();

       return response(component_ns_get_component_type_name, no_success);
    }

    LAGAS_(info) << (boost::format(
        "component_namespace::get_component_typename, key(%1%), response(%2%)")
        % int(t) % result);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_get_component_type_name, result);
} // }}}

response component_namespace::get_num_localities(
    request const& req
  , error_code& ec
    )
{ // {{{ get_num_localities implementation
    component_id_type key = req.get_component_type();

    // If the requested component type is a derived type, use only its derived
    // part for the lookup.
    if (key != components::get_base_type(key))
        key = components::get_derived_type(key);

    mutex_type::scoped_lock l(mutex_);

    factory_table_type::const_iterator it = factories_.find(key)
                                     , end = factories_.end();
    if (it == end)
    {
        LAGAS_(info) << (boost::format(
            "component_namespace::get_num_localities, key(%1%), localities(0)")
            % key);

        if (&ec != &throws)
            ec = make_success_code();

        response(component_ns_num_localities, boost::uint32_t(0), no_success);
    }

    boost::uint32_t num_localities = static_cast<boost::uint32_t>(it->second.size());

    LAGAS_(info) << (boost::format(
        "component_namespace::get_num_localities, key(%1%), localities(%2%)")
        % key % num_localities);

    if (&ec != &throws)
        ec = make_success_code();

    return response(component_ns_num_localities, num_localities);
} // }}}

response component_namespace::statistics_counter(
    request const& req
  , error_code& ec
    )
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "component_namespace::statistics_counter";

    std::string name(req.get_statistics_counter_name());

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p, ec);
    if (ec) return response();

    if (p.objectname_ != "agas")
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "component_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    namespace_action_code code = invalid_request;
    detail::counter_target target = detail::counter_target_invalid;
    for (std::size_t i = 0;
          i < detail::num_component_namespace_services;
          ++i)
    {
        if (p.countername_ == detail::component_namespace_services[i].name_)
        {
            code = detail::component_namespace_services[i].code_;
            target = detail::component_namespace_services[i].target_;
            break;
        }
    }

    if (code == invalid_request || target == detail::counter_target_invalid)
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "component_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    typedef component_namespace::counter_data cd;

    HPX_STD_FUNCTION<boost::int64_t()> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case component_ns_bind_prefix:
            get_data_func = boost::bind(&cd::get_bind_prefix_count, &counter_data_);
            break;
        case component_ns_bind_name:
            get_data_func = boost::bind(&cd::get_bind_name_count, &counter_data_);
            break;
        case component_ns_resolve_id:
            get_data_func = boost::bind(&cd::get_resolve_id_count, &counter_data_);
            break;
        case component_ns_unbind_name:
            get_data_func = boost::bind(&cd::get_unbind_name_count, &counter_data_);
            break;
        case component_ns_iterate_types:
            get_data_func = boost::bind(&cd::get_iterate_types_count, &counter_data_);
            break;
        case component_ns_get_component_type_name:
            get_data_func = boost::bind(&cd::get_component_type_name_count, &counter_data_);
            break;
        case component_ns_num_localities:
            get_data_func = boost::bind(&cd::get_num_localities_count, &counter_data_);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "component_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }
    else {
        switch (code) {
        case component_ns_bind_prefix:
            get_data_func = boost::bind(&cd::get_bind_prefix_time, &counter_data_);
            break;
        case component_ns_bind_name:
            get_data_func = boost::bind(&cd::get_bind_name_time, &counter_data_);
            break;
        case component_ns_resolve_id:
            get_data_func = boost::bind(&cd::get_resolve_id_time, &counter_data_);
            break;
        case component_ns_unbind_name:
            get_data_func = boost::bind(&cd::get_unbind_name_time, &counter_data_);
            break;
        case component_ns_iterate_types:
            get_data_func = boost::bind(&cd::get_iterate_types_time, &counter_data_);
            break;
        case component_ns_get_component_type_name:
            get_data_func = boost::bind(&cd::get_component_type_name_time, &counter_data_);
            break;
        case component_ns_num_localities:
            get_data_func = boost::bind(&cd::get_num_localities_time, &counter_data_);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "component_namespace::statistics"
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

    return response(component_ns_statistics_counter, gid);
} // }}}

// access current counter values
boost::int64_t component_namespace::counter_data::get_bind_prefix_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return bind_prefix_.count_;
}

boost::int64_t component_namespace::counter_data::get_bind_name_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return bind_name_.count_;
}

boost::int64_t component_namespace::counter_data::get_resolve_id_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return resolve_id_.count_;
}

boost::int64_t component_namespace::counter_data::get_unbind_name_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return unbind_name_.count_;
}

boost::int64_t component_namespace::counter_data::get_iterate_types_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return iterate_types_.count_;
}

boost::int64_t component_namespace::counter_data::get_component_type_name_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return get_component_type_name_.count_;
}

boost::int64_t component_namespace::counter_data::get_num_localities_count() const
{
    mutex_type::scoped_lock l(mtx_);
    return num_localities_.count_;
}

// access execution time counters
boost::int64_t component_namespace::counter_data::get_bind_prefix_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return bind_prefix_.time_;
}

boost::int64_t component_namespace::counter_data::get_bind_name_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return bind_name_.time_;
}

boost::int64_t component_namespace::counter_data::get_resolve_id_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return resolve_id_.time_;
}

boost::int64_t component_namespace::counter_data::get_unbind_name_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return unbind_name_.time_;
}

boost::int64_t component_namespace::counter_data::get_iterate_types_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return iterate_types_.time_;
}

boost::int64_t component_namespace::counter_data::get_component_type_name_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return get_component_type_name_.time_;
}

boost::int64_t component_namespace::counter_data::get_num_localities_time() const
{
    mutex_type::scoped_lock l(mtx_);
    return num_localities_.time_;
}

// increment counter values
void component_namespace::counter_data::increment_bind_prefix_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++bind_prefix_.count_;
}

void component_namespace::counter_data::increment_bind_name_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++bind_name_.count_;
}

void component_namespace::counter_data::increment_resolve_id_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++resolve_id_.count_;
}

void component_namespace::counter_data::increment_unbind_name_ount()
{
    mutex_type::scoped_lock l(mtx_);
    ++unbind_name_.count_;
}

void component_namespace::counter_data::increment_iterate_types_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++iterate_types_.count_;
}

void component_namespace::counter_data::increment_get_component_type_name_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++get_component_type_name_.count_;
}

void component_namespace::counter_data::increment_num_localities_count()
{
    mutex_type::scoped_lock l(mtx_);
    ++num_localities_.count_;
}

}}}

