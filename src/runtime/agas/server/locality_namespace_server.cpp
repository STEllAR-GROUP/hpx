////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2015 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/insert_checked.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/scoped_timer.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace server
{

// register all performance counter types exposed by this component
void locality_namespace::register_counter_types(
    error_code& ec
    )
{
    using util::placeholders::_1;
    using util::placeholders::_2;
    boost::format help_count(
        "returns the number of invocations of the AGAS service '%s'");
    boost::format help_time(
        "returns the overall execution time of the AGAS service '%s'");
    performance_counters::create_counter_func creator(
        util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
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
    using util::placeholders::_1;
    using util::placeholders::_2;
    performance_counters::create_counter_func creator(
        util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
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
    agas::register_name(launch::sync, instance_name_,
        get_unmanaged_id().get_gid(), ec);
}

void locality_namespace::unregister_server_instance(
    error_code& ec
    )
{
    agas::unregister_name(launch::sync, instance_name_, ec);
    this->base_type::finalize();
}

void locality_namespace::finalize()
{
    if (!instance_name_.empty())
    {
        error_code ec(lightweight);
        agas::unregister_name(launch::sync, instance_name_, ec);
    }
}

std::uint32_t locality_namespace::allocate(
    parcelset::endpoints_type const& endpoints
  , std::uint64_t count
  , std::uint32_t num_threads
  , naming::gid_type suggested_prefix
    )
{ // {{{ allocate implementation
    util::scoped_timer<boost::atomic<std::int64_t> > update(
        counter_data_.allocate_.time_
    );
    counter_data_.increment_allocate_count();

    using hpx::util::get;

    std::unique_lock<mutex_type> l(mutex_);

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

        HPX_THROW_EXCEPTION(internal_server_error
          , "locality_namespace::allocate"
          , "primary namespace has been exhausted");
    }

    // Compute the locality's prefix.
    std::uint32_t prefix = naming::invalid_locality_id;

    // check if the suggested prefix can be used instead of the next
    // free one
    std::uint32_t suggested_locality_id =
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

        HPX_THROW_EXCEPTION(lock_error
          , "locality_namespace::allocate"
          , boost::str(boost::format(
                "partition table insertion failed due to a locking "
                "error or memory corruption, endpoint(%1%), "
                "prefix(%2%)") % endpoints % prefix));
    }


    // Now that we've inserted the locality into the partition table
    // successfully, we need to put the locality's GID into the GVA
    // table so that parcels can be sent to the memory of a locality.
    if (primary_)
    {
        naming::gid_type id(naming::get_gid_from_locality_id(prefix));
        gva const g(id, components::component_runtime_support, count);

        if(!primary_->bind_gid(g, id, id))
        {
            HPX_THROW_EXCEPTION(bad_request
              , "locality_namespace::allocate"
              , boost::str(boost::format(
                    "unable to bind prefix(%1%) to a gid") % prefix));
        }
        return prefix;
    }

    LAGAS_(info) << (boost::format(
        "locality_namespace::allocate, ep(%1%), count(%2%), "
        "prefix(%3%)")
        % endpoints % count % prefix);

    return prefix;
} // }}}

parcelset::endpoints_type locality_namespace::resolve_locality(
    naming::gid_type locality)
{ // {{{ resolve_locality implementation
    util::scoped_timer<boost::atomic<std::int64_t> > update(
        counter_data_.resolve_locality_.time_
    );
    counter_data_.increment_resolve_locality_count();

    using hpx::util::get;
    std::uint32_t prefix = naming::get_locality_id_from_gid(locality);

    std::lock_guard<mutex_type> l(mutex_);
    partition_table_type::iterator it = partitions_.find(prefix);

    if(it != partitions_.end())
    {
        return get<0>(it->second);
    }

    return parcelset::endpoints_type();
} // }}}

void locality_namespace::free(naming::gid_type locality)
{ // {{{ free implementation
    util::scoped_timer<boost::atomic<std::int64_t> > update(
        counter_data_.free_.time_
    );
    counter_data_.increment_free_count();

    using hpx::util::get;

    // parameters
    std::uint32_t prefix = naming::get_locality_id_from_gid(locality);

    std::unique_lock<mutex_type> l(mutex_);

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

            std::uint32_t locality_id =
                naming::get_locality_id_from_gid(locality);

            // remove primary namespace
            {
                naming::gid_type service(HPX_AGAS_PRIMARY_NS_MSB,
                    HPX_AGAS_PRIMARY_NS_LSB);
                primary_->unbind_gid(
                    1, naming::replace_locality_id(service, locality_id));
            }

            // remove symbol namespace
            {
                naming::gid_type service(HPX_AGAS_SYMBOL_NS_MSB,
                    HPX_AGAS_SYMBOL_NS_LSB);
                primary_->unbind_gid(
                    1, naming::replace_locality_id(service, locality_id));
            }

            // remove locality itself
            {
                primary_->unbind_gid(0, locality);
            }
        }

        /*
        LAGAS_(info) << (boost::format(
            "locality_namespace::free, ep(%1%)")
            % ep);
        */
    }

    /*
    LAGAS_(info) << (boost::format(
        "locality_namespace::free, ep(%1%), "
        "response(no_success)")
        % ep);
    */
} // }}}

std::vector<std::uint32_t> locality_namespace::localities()
{ // {{{ localities implementation
    util::scoped_timer<boost::atomic<std::int64_t> > update(
        counter_data_.localities_.time_
    );
    counter_data_.increment_localities_count();

    std::lock_guard<mutex_type> l(mutex_);

    std::vector<std::uint32_t> p;

    partition_table_type::const_iterator it = partitions_.begin()
                                       , end = partitions_.end();

    for (/**/; it != end; ++it)
        p.push_back(it->first);

    LAGAS_(info) << (boost::format(
        "locality_namespace::localities, localities(%1%)")
        % p.size());

    return p;
} // }}}

std::uint32_t locality_namespace::get_num_localities()
{ // {{{ get_num_localities implementation
    util::scoped_timer<boost::atomic<std::int64_t> > update(
        counter_data_.num_localities_.time_
    );
    counter_data_.increment_num_localities_count();
    std::lock_guard<mutex_type> l(mutex_);

    std::uint32_t num_localities =
        static_cast<std::uint32_t>(partitions_.size());

    LAGAS_(info) << (boost::format(
        "locality_namespace::get_num_localities, localities(%1%)")
        % num_localities);

    return num_localities;
} // }}}

std::vector<std::uint32_t> locality_namespace::get_num_threads()
{ // {{{ get_num_threads implementation
    std::lock_guard<mutex_type> l(mutex_);

    std::vector<std::uint32_t> num_threads;

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

    return num_threads;
} // }}}

std::uint32_t locality_namespace::get_num_overall_threads()
{
    std::lock_guard<mutex_type> l(mutex_);

    std::uint32_t num_threads = 0;

    partition_table_type::iterator end = partitions_.end();
    for (partition_table_type::iterator it = partitions_.begin();
         it != end; ++it)
    {
        using hpx::util::get;
        num_threads += get<1>(it->second);
    }

    LAGAS_(info) << (boost::format(
        "locality_namespace::get_num_overall_threads, localities(%1%)")
        % num_threads);

    return num_threads;
}

naming::gid_type locality_namespace::statistics_counter(std::string name)
{ // {{{ statistics_counter implementation
    LAGAS_(info) << "locality_namespace::statistics_counter";

    performance_counters::counter_path_elements p;
    performance_counters::get_counter_path_elements(name, p);

    if (p.objectname_ != "agas")
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "locality_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
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
        HPX_THROW_EXCEPTION(bad_parameter,
            "locality_namespace::statistics_counter",
            "unknown performance counter (unrelated to AGAS)");
    }

    typedef locality_namespace::counter_data cd;

    using util::placeholders::_1;
    util::function_nonser<std::int64_t(bool)> get_data_func;
    if (target == detail::counter_target_count)
    {
        switch (code) {
        case locality_ns_allocate:
            get_data_func = util::bind(&cd::get_allocate_count,
                &counter_data_, _1);
            break;
        case locality_ns_resolve_locality:
            get_data_func = util::bind(&cd::get_resolve_locality_count,
                &counter_data_, _1);
            break;
        case locality_ns_free:
            get_data_func = util::bind(&cd::get_free_count,
                &counter_data_, _1);
            break;
        case locality_ns_localities:
            get_data_func = util::bind(&cd::get_localities_count,
                &counter_data_, _1);
            break;
        case locality_ns_num_localities:
            get_data_func = util::bind(&cd::get_num_localities_count,
                &counter_data_, _1);
            break;
        case locality_ns_num_threads:
            get_data_func = util::bind(&cd::get_num_threads_count,
                &counter_data_, _1);
            break;
        case locality_ns_statistics_counter:
            get_data_func = util::bind(&cd::get_overall_count, &counter_data_, _1);
            break;
        default:
            HPX_THROW_EXCEPTION(bad_parameter
              , "locality_namespace::statistics"
              , "bad action code while querying statistics");
        }
    }
    else {
        HPX_ASSERT(detail::counter_target_time == target);
        switch (code) {
        case locality_ns_allocate:
            get_data_func = util::bind(&cd::get_allocate_time,
                &counter_data_, _1);
            break;
        case locality_ns_resolve_locality:
            get_data_func = util::bind(&cd::get_resolve_locality_time,
                &counter_data_, _1);
            break;
        case locality_ns_free:
            get_data_func = util::bind(&cd::get_free_time, &counter_data_, _1);
            break;
        case locality_ns_localities:
            get_data_func = util::bind(&cd::get_localities_time,
                &counter_data_, _1);
            break;
        case locality_ns_num_localities:
            get_data_func = util::bind(&cd::get_num_localities_time,
                &counter_data_, _1);
            break;
        case locality_ns_num_threads:
            get_data_func = util::bind(&cd::get_num_threads_time, &counter_data_, _1);
            break;
        case locality_ns_statistics_counter:
            get_data_func = util::bind(&cd::get_overall_time, &counter_data_, _1);
            break;
        default:
            HPX_THROW_EXCEPTION(bad_parameter
              , "locality_namespace::statistics"
              , "bad action code while querying statistics");
        }
    }

    performance_counters::counter_info info;
    performance_counters::get_counter_type(name, info);

    performance_counters::complement_counter_info(info);

    using performance_counters::detail::create_raw_counter;
    naming::gid_type gid = create_raw_counter(info, get_data_func, hpx::throws);

    return naming::detail::strip_credits_from_gid(gid);
}

// access current counter values
std::int64_t locality_namespace::counter_data::get_allocate_count(bool reset)
{
    return util::get_and_reset_value(allocate_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_resolve_locality_count(bool reset)
{
    return util::get_and_reset_value(resolve_locality_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_free_count(bool reset)
{
    return util::get_and_reset_value(free_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_localities_count(bool reset)
{
    return util::get_and_reset_value(localities_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_num_localities_count(bool reset)
{
    return util::get_and_reset_value(num_localities_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_num_threads_count(bool reset)
{
    return util::get_and_reset_value(num_threads_.count_, reset);
}

std::int64_t locality_namespace::counter_data::get_overall_count(bool reset)
{
    return util::get_and_reset_value(allocate_.count_, reset) +
        util::get_and_reset_value(resolve_locality_.count_, reset) +
        util::get_and_reset_value(free_.count_, reset) +
        util::get_and_reset_value(localities_.count_, reset) +
        util::get_and_reset_value(num_localities_.count_, reset) +
        util::get_and_reset_value(num_threads_.count_, reset);
}

// access execution time counters
std::int64_t locality_namespace::counter_data::get_allocate_time(bool reset)
{
    return util::get_and_reset_value(allocate_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_resolve_locality_time(bool reset)
{
    return util::get_and_reset_value(resolve_locality_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_free_time(bool reset)
{
    return util::get_and_reset_value(free_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_localities_time(bool reset)
{
    return util::get_and_reset_value(localities_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_num_localities_time(bool reset)
{
    return util::get_and_reset_value(num_localities_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_num_threads_time(bool reset)
{
    return util::get_and_reset_value(num_threads_.time_, reset);
}

std::int64_t locality_namespace::counter_data::get_overall_time(bool reset)
{
    return util::get_and_reset_value(allocate_.time_, reset) +
        util::get_and_reset_value(resolve_locality_.time_, reset) +
        util::get_and_reset_value(free_.time_, reset) +
        util::get_and_reset_value(localities_.time_, reset) +
        util::get_and_reset_value(num_localities_.time_, reset) +
        util::get_and_reset_value(num_threads_.time_, reset);
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
}}}

