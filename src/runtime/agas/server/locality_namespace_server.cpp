////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <list>
#include <string>
#include <vector>

#include <boost/thread/locks.hpp>

namespace hpx { namespace agas
{

naming::gid_type bootstrap_locality_namespace_gid()
{
    return naming::gid_type(HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB);
}

naming::id_type bootstrap_locality_namespace_id()
{
    return naming::id_type( HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB
      , naming::id_type::unmanaged);
}

namespace server
{

// TODO: This isn't scalable, we have to update it every time we add a new
// AGAS request/response type.
response locality_namespace::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    switch (req.get_action_code())
    {
        case locality_ns_allocate:
            {
                update_time_on_exit update(
                    counter_data_.allocate_.time_
                );
                counter_data_.increment_allocate_count();
                return allocate(req, ec);
            }
        case locality_ns_free:
            {
                update_time_on_exit update(
                    counter_data_.free_.time_
                );
                counter_data_.increment_free_count();
                return free(req, ec);
            }
        case locality_ns_localities:
            {
                update_time_on_exit update(
                    counter_data_.localities_.time_
                );
                counter_data_.increment_localities_count();
                return localities(req, ec);
            }
        case locality_ns_resolve_locality:
            {
                update_time_on_exit update(
                    counter_data_.resolve_locality_.time_
                );
                counter_data_.increment_resolve_locality_count();
                return resolve_locality(req, ec);
            }
        case locality_ns_resolved_localities:
            {
                update_time_on_exit update(
                    counter_data_.resolved_localities_.time_
                );
                counter_data_.increment_resolved_localities_count();
                return resolved_localities(req, ec);
            }
        case locality_ns_num_localities:
            {
                update_time_on_exit update(
                    counter_data_.num_localities_.time_
                );
                counter_data_.increment_num_localities_count();
                return get_num_localities(req, ec);
            }
        case locality_ns_num_threads:
            {
                update_time_on_exit update(
                    counter_data_.num_threads_.time_
                );
                counter_data_.increment_num_threads_count();
                return get_num_threads(req, ec);
            }
        case locality_ns_statistics_counter:
            return statistics_counter(req, ec);

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
                "locality_namespace::service, redirecting request to "
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
                "locality_namespace::service, redirecting request to "
                "component_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        case symbol_ns_bind:
        case symbol_ns_resolve:
        case symbol_ns_unbind:
        case symbol_ns_iterate_names:
        case symbol_ns_on_event:
        {
            LAGAS_(warning) <<
                "locality_namespace::service, redirecting request to "
                "symbol_namespace";
            return naming::get_agas_client().service(req, ec);
        }

        default:
        case component_ns_service:
        case locality_ns_service:
        case primary_ns_service:
        case symbol_ns_service:
        case invalid_request:
        {
            HPX_THROWS_IF(ec, bad_action_code
              , "locality_namespace::service"
              , boost::str(boost::format(
                    "invalid action code encountered in request, "
                    "action_code(%x)")
                    % boost::uint16_t(req.get_action_code())));
            return response();
        }
    };
} // }}}

// register all performance counter types exposed by this component
void locality_namespace::register_counter_types(
    error_code& ec
    )
{
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    performance_counters::create_counter_func creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::locality_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_locality_namespace_services;
          ++i)
    {
        // global counters are handled elsewhere
        if (detail::locality_namespace_services[i].code_ ==
            locality_ns_statistics_counter)
            continue;

        std::string name(detail::locality_namespace_services[i].name_);
        std::string help;
        std::string::size_type p = name.find_last_of('/');
        HPX_ASSERT(p != std::string::npos);

        if (detail::locality_namespace_services[i].target_ ==
            detail::counter_target_count)
            help = boost::str(help_count % name.substr(p+1));
        else
            help = boost::str(help_time % name.substr(p+1));

        performance_counters::install_counter_type(
            agas::performance_counter_basename + name
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality0_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::locality_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void locality_namespace::register_global_counter_types(
    error_code& ec
    )
{
    performance_counters::create_counter_func creator(
        boost::bind(&performance_counters::agas_raw_counter_creator, _1, _2
      , agas::server::locality_namespace_service_name));

    for (std::size_t i = 0;
          i != detail::num_locality_namespace_services;
          ++i)
    {
        // local counters are handled elsewhere
        if (detail::locality_namespace_services[i].code_ !=
            locality_ns_statistics_counter)
            continue;

        std::string help;
        if (detail::locality_namespace_services[i].target_ ==
            detail::counter_target_count)
            help = "returns the overall number of invocations \
                    of all locality AGAS services";
        else
            help = "returns the overall execution time of all locality AGAS services";

        performance_counters::install_counter_type(
            std::string(agas::performance_counter_basename) +
                detail::locality_namespace_services[i].name_
          , performance_counters::counter_raw
          , help
          , creator
          , &performance_counters::locality0_counter_discoverer
          , HPX_PERFORMANCE_COUNTER_V1
          , detail::locality_namespace_services[i].uom_
          , ec
          );
        if (ec) return;
    }
}

void locality_namespace::register_server_instance(
    char const* servicename
  , error_code& ec
    )
{
    // now register this AGAS instance with AGAS :-P
    instance_name_ = agas::service_name;
    instance_name_ += servicename;
    instance_name_ += agas::server::locality_namespace_service_name;

    // register a gid (not the id) to avoid AGAS holding a reference to this
    // component
    agas::register_name_sync(instance_name_, get_unmanaged_id().get_gid(), ec);
}

void locality_namespace::unregister_server_instance(
    error_code& ec
    )
{
    agas::unregister_name_sync(instance_name_, ec);
    this->base_type::finalize();
}

void locality_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name_sync(instance_name_, ec);
    }
}

// TODO: do/undo semantics (e.g. transactions)
std::vector<response> locality_namespace::bulk_service(
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

response locality_namespace::allocate(
    request const& req
  , error_code& ec
    )
{ // {{{ allocate implementation
    using hpx::util::get;

    // parameters
    parcelset::endpoints_type endpoints = req.get_endpoints();
    boost::uint64_t const count = req.get_count();
    boost::uint32_t const num_threads = req.get_num_threads();
    naming::gid_type const suggested_prefix = req.get_suggested_prefix();

    boost::unique_lock<mutex_type> l(mutex_);

#if defined(HPX_DEBUG)
    for (partition_table_type::value_type const& partition : partitions_)
    {
        HPX_ASSERT(get<0>(partition.second) != endpoints);
    }
#endif
    // Check for address space exhaustion.
    if (HPX_UNLIKELY(0xFFFFFFFE < partitions_.size())) //-V104
    {
        l.unlock();

        HPX_THROWS_IF(ec, internal_server_error
          , "locality_namespace::allocate"
          , "primary namespace has been exhausted");
        return response();
    }

    // Compute the locality's prefix.
    boost::uint32_t prefix = naming::invalid_locality_id;

    // check if the suggested prefix can be used instead of the next
    // free one
    boost::uint32_t suggested_locality_id =
        naming::get_locality_id_from_gid(suggested_prefix);

    partition_table_type::iterator it = partitions_.end();
    if (suggested_locality_id != naming::invalid_locality_id)
    {
        it = partitions_.find(suggested_locality_id);

        if(it == partitions_.end())
        {
            prefix = suggested_locality_id;
        }
        else
        {
            do {
                prefix = prefix_counter_++;
                it = partitions_.find(prefix);
            } while (it != partitions_.end());
        }
    }
    else
    {
        do {
            prefix = prefix_counter_++;
            it = partitions_.find(prefix);
        } while (it != partitions_.end());
    }

    // We need to create an entry in the partition table for this
    // locality.
    if(HPX_UNLIKELY(!util::insert_checked(partitions_.insert(
        std::make_pair(prefix, partition_type(endpoints, num_threads))), it)))
    {
        l.unlock();

        HPX_THROWS_IF(ec, lock_error
          , "locality_namespace::allocate"
          , boost::str(boost::format(
                "partition table insertion failed due to a locking "
                "error or memory corruption, endpoint(%1%), "
                "prefix(%2%)") % endpoints % prefix));
        return response();
    }


    // Now that we've inserted the locality into the partition table
    // successfully, we need to put the locality's GID into the GVA
    // table so that parcels can be sent to the memory of a locality.
    if (primary_)
    {
        naming::gid_type id(naming::get_gid_from_locality_id(prefix));
        gva const g(id, components::component_runtime_support, count);

        request req(primary_ns_bind_gid, id, g, id);
        response resp = primary_->service(req, ec);
        if (ec) return resp;
    }

    LAGAS_(info) << (boost::format(
        "locality_namespace::allocate, ep(%1%), count(%2%), "
        "prefix(%3%)")
        % endpoints % count % prefix);

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_allocate, prefix);
} // }}}

response locality_namespace::resolve_locality(
    request const& req
  , error_code& ec
    )
{ // {{{ resolve_locality implementation

    using hpx::util::get;
    boost::uint32_t prefix = naming::get_locality_id_from_gid(req.get_gid());

    boost::lock_guard<mutex_type> l(mutex_);
    partition_table_type::iterator it = partitions_.find(prefix);

    if(it != partitions_.end())
    {
        return response(locality_ns_resolve_locality, get<0>(it->second));
    }

    return response(locality_ns_resolve_locality, parcelset::endpoints_type(),
        no_success);
} // }}}

response locality_namespace::free(
    request const& req
  , error_code& ec
    )
{ // {{{ free implementation
    using hpx::util::get;

    // parameters
    naming::gid_type locality = req.get_gid();
    boost::uint32_t prefix = naming::get_locality_id_from_gid(locality);

    boost::unique_lock<mutex_type> l(mutex_);

    partition_table_type::iterator pit = partitions_.find(prefix)
                                 , pend = partitions_.end();

    if (pit != pend)
    {
        /*
        // Wipe the locality from the tables.
        naming::gid_type locality =
            naming::get_gid_from_locality_id(get<0>(pit->second));

        // first remove entry from reverse partition table
        prefixes_.erase(get<0>(pit->second));
        */

        // now remove it from the main partition table
        partitions_.erase(pit);

        if (primary_)
        {
            l.unlock();

            boost::uint32_t locality_id =
                naming::get_locality_id_from_gid(locality);

            // remove primary namespace
            {
                naming::gid_type service(HPX_AGAS_PRIMARY_NS_MSB,
                    HPX_AGAS_PRIMARY_NS_LSB);
                request req(primary_ns_unbind_gid,
                    naming::replace_locality_id(service, locality_id), 1);
                response resp = primary_->service(req, ec);
                if (ec) return resp;
            }

            // remove symbol namespace
            {
                naming::gid_type service(HPX_AGAS_SYMBOL_NS_MSB,
                    HPX_AGAS_SYMBOL_NS_LSB);
                request req(primary_ns_unbind_gid,
                    naming::replace_locality_id(service, locality_id), 1);
                response resp = primary_->service(req, ec);
                if (ec) return resp;
            }

            // remove locality itself
            {
                request req(primary_ns_unbind_gid, locality, 0);
                response resp = primary_->service(req, ec);
                if (ec) return resp;
            }
        }

        /*
        LAGAS_(info) << (boost::format(
            "locality_namespace::free, ep(%1%)")
            % ep);
        */

        if (&ec != &throws)
            ec = make_success_code();

        return response(locality_ns_free);
    }

    /*
    LAGAS_(info) << (boost::format(
        "locality_namespace::free, ep(%1%), "
        "response(no_success)")
        % ep);
    */

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_free
                       , no_success);
} // }}}

response locality_namespace::localities(
    request const& req
  , error_code& ec
    )
{ // {{{ localities implementation
    using hpx::util::get;

    boost::lock_guard<mutex_type> l(mutex_);

    std::vector<boost::uint32_t> p;

    partition_table_type::const_iterator it = partitions_.begin()
                                       , end = partitions_.end();

    for (/**/; it != end; ++it)
        p.push_back(it->first);

    LAGAS_(info) << (boost::format(
        "locality_namespace::localities, localities(%1%)")
        % p.size());

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_localities, p);
} // }}}

response locality_namespace::resolved_localities(
    request const& req
  , error_code& ec
    )
{ // {{{ localities implementation
    using hpx::util::get;

    boost::lock_guard<mutex_type> l(mutex_);

    std::map<naming::gid_type, parcelset::endpoints_type> localities;

    partition_table_type::const_iterator it = partitions_.begin()
                                       , end = partitions_.end();

    for (; it != end; ++it)
    {
        localities.insert(
            std::make_pair(
                naming::get_gid_from_locality_id(it->first)
              , get<0>(it->second)
            )
        );
    }

    LAGAS_(info) << (boost::format(
        "locality_namespace::resolved_localities, localities(%1%)")
        % localities.size());

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_resolved_localities, localities);
} // }}}

response locality_namespace::get_num_localities(
    request const& req
  , error_code& ec
    )
{ // {{{ get_num_localities implementation
    boost::lock_guard<mutex_type> l(mutex_);

    boost::uint32_t num_localities =
        static_cast<boost::uint32_t>(partitions_.size());

    LAGAS_(info) << (boost::format(
        "locality_namespace::get_num_localities, localities(%1%)")
        % num_localities);

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_num_localities, num_localities);
} // }}}

response locality_namespace::get_num_threads(
    request const& req
  , error_code& ec
    )
{ // {{{ get_num_threads implementation
    boost::lock_guard<mutex_type> l(mutex_);

    std::vector<boost::uint32_t> num_threads;

    partition_table_type::iterator end = partitions_.end();
    for (partition_table_type::iterator it = partitions_.begin();
         it != end; ++it)
    {
        using hpx::util::get;
        num_threads.push_back(get<1>(it->second));
    }

    LAGAS_(info) << (boost::format(
        "locality_namespace::get_num_threads, localities(%1%)")
        % num_threads.size());

    if (&ec != &throws)
        ec = make_success_code();

    return response(locality_ns_num_threads, num_threads);
} // }}}

response locality_namespace::statistics_counter(
    request const& req
  , error_code& ec
    )
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "locality_namespace::statistics_counter";

    std::string name(req.get_statistics_counter_name());

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p, ec);
    if (ec) return response();

    if (p.objectname_ != "agas")
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "locality_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    namespace_action_code code = invalid_request;
    detail::counter_target target = detail::counter_target_invalid;
    for (std::size_t i = 0;
          i != detail::num_locality_namespace_services;
          ++i)
    {
        if (p.countername_ == detail::locality_namespace_services[i].name_)
        {
            code = detail::locality_namespace_services[i].code_;
            target = detail::locality_namespace_services[i].target_;
            break;
        }
    }

    if (code == invalid_request || target == detail::counter_target_invalid)
    {
        HPX_THROWS_IF(ec, bad_parameter,
            "locality_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
        return response();
    }

    typedef locality_namespace::counter_data cd;

    util::function_nonser<boost::int64_t(bool)> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case locality_ns_allocate:
            get_data_func = boost::bind(&cd::get_allocate_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_resolve_locality:
            get_data_func = boost::bind(&cd::get_resolve_locality_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_free:
            get_data_func = boost::bind(&cd::get_free_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_localities:
            get_data_func = boost::bind(&cd::get_localities_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_resolved_localities:
            get_data_func = boost::bind(&cd::get_resolved_localities_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_num_localities:
            get_data_func = boost::bind(&cd::get_num_localities_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_num_threads:
            get_data_func = boost::bind(&cd::get_num_threads_count,
                &counter_data_, ::_1);
            break;
        case locality_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_count, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "locality_namespace::statistics"
              , "bad action code while querying statistics");
            return response();
        }
    }
    else {
        HPX_ASSERT(detail::counter_target_time == target);
        switch (code) {
        case locality_ns_allocate:
            get_data_func = boost::bind(&cd::get_allocate_time,
                &counter_data_, ::_1);
            break;
        case locality_ns_resolve_locality:
            get_data_func = boost::bind(&cd::get_resolve_locality_time,
                &counter_data_, ::_1);
            break;
        case locality_ns_free:
            get_data_func = boost::bind(&cd::get_free_time, &counter_data_, ::_1);
            break;
        case locality_ns_localities:
            get_data_func = boost::bind(&cd::get_localities_time,
                &counter_data_, ::_1);
            break;
        case locality_ns_resolved_localities:
            get_data_func = boost::bind(&cd::get_resolved_localities_time,
                &counter_data_, ::_1);
            break;
        case locality_ns_num_localities:
            get_data_func = boost::bind(&cd::get_num_localities_time,
                &counter_data_, ::_1);
            break;
        case locality_ns_num_threads:
            get_data_func = boost::bind(&cd::get_num_threads_time, &counter_data_, ::_1);
            break;
        case locality_ns_statistics_counter:
            get_data_func = boost::bind(&cd::get_overall_time, &counter_data_, ::_1);
            break;
        default:
            HPX_THROWS_IF(ec, bad_parameter
              , "locality_namespace::statistics"
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
}

// access current counter values
boost::int64_t locality_namespace::counter_data::get_allocate_count(bool reset)
{
    return util::get_and_reset_value(allocate_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_resolve_locality_count(bool reset)
{
    return util::get_and_reset_value(resolve_locality_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_free_count(bool reset)
{
    return util::get_and_reset_value(free_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_localities_count(bool reset)
{
    return util::get_and_reset_value(localities_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_num_localities_count(bool reset)
{
    return util::get_and_reset_value(num_localities_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_num_threads_count(bool reset)
{
    return util::get_and_reset_value(num_threads_.count_, reset);
}

boost::int64_t locality_namespace::counter_data
        ::get_resolved_localities_count(bool reset)
{
    return util::get_and_reset_value(resolved_localities_.count_, reset);
}

boost::int64_t locality_namespace::counter_data::get_overall_count(bool reset)
{
    return util::get_and_reset_value(allocate_.count_, reset) +
        util::get_and_reset_value(resolve_locality_.count_, reset) +
        util::get_and_reset_value(free_.count_, reset) +
        util::get_and_reset_value(localities_.count_, reset) +
        util::get_and_reset_value(num_localities_.count_, reset) +
        util::get_and_reset_value(num_threads_.count_, reset) +
        util::get_and_reset_value(resolved_localities_.count_, reset);
}

// access execution time counters
boost::int64_t locality_namespace::counter_data::get_allocate_time(bool reset)
{
    return util::get_and_reset_value(allocate_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_resolve_locality_time(bool reset)
{
    return util::get_and_reset_value(resolve_locality_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_free_time(bool reset)
{
    return util::get_and_reset_value(free_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_localities_time(bool reset)
{
    return util::get_and_reset_value(localities_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_num_localities_time(bool reset)
{
    return util::get_and_reset_value(num_localities_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_num_threads_time(bool reset)
{
    return util::get_and_reset_value(num_threads_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_resolved_localities_time(bool reset)
{
    return util::get_and_reset_value(resolved_localities_.time_, reset);
}

boost::int64_t locality_namespace::counter_data::get_overall_time(bool reset)
{
    return util::get_and_reset_value(allocate_.time_, reset) +
        util::get_and_reset_value(resolve_locality_.time_, reset) +
        util::get_and_reset_value(free_.time_, reset) +
        util::get_and_reset_value(localities_.time_, reset) +
        util::get_and_reset_value(num_localities_.time_, reset) +
        util::get_and_reset_value(num_threads_.time_, reset) +
        util::get_and_reset_value(resolved_localities_.time_, reset);
}

// increment counter values
void locality_namespace::counter_data::increment_allocate_count()
{
    ++allocate_.count_;
}

void locality_namespace::counter_data::increment_resolve_locality_count()
{
    ++resolve_locality_.count_;
}

void locality_namespace::counter_data::increment_free_count()
{
    ++free_.count_;
}

void locality_namespace::counter_data::increment_localities_count()
{
    ++localities_.count_;
}

void locality_namespace::counter_data::increment_num_localities_count()
{
    ++num_localities_.count_;
}

void locality_namespace::counter_data::increment_num_threads_count()
{
    ++num_threads_.count_;
}

void locality_namespace::counter_data::increment_resolved_localities_count()
{
    ++resolved_localities_.count_;
}

}}}

