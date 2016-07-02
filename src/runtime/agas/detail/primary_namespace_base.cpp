//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/runtime/agas/detail/primary_namespace_base.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/logging.hpp>

#include <string>

namespace hpx { namespace agas { namespace detail
{
    void primary_namespace_base::finalize()
    {
        if (!instance_name_.empty())
        {
            error_code ec(lightweight);
            agas::unregister_name_sync(instance_name_, ec);
        }
    }

    // Parcel routing forwards the message handler request to the routed action
    parcelset::policies::message_handler*
    primary_namespace_base::get_message_handler(parcelset::parcelhandler* ph,
        parcelset::locality const& loc, parcelset::parcel const& p)
    {
        typedef hpx::actions::transfer_action<
                primary_namespace_base::route_action
            > action_type;

        action_type * act = static_cast<action_type *>(p.get_action());

        parcelset::parcel const& routed_p = hpx::actions::get<0>(*act);
        return routed_p.get_message_handler(ph, loc);
    }

    serialization::binary_filter*
    primary_namespace_base::get_serialization_filter(parcelset::parcel const& p)
    {
        typedef hpx::actions::transfer_action<
                primary_namespace_base::route_action
            > action_type;

        action_type * act = static_cast<action_type *>(p.get_action());

        parcelset::parcel const& routed_p = hpx::actions::get<0>(*act);
        return routed_p.get_serialization_filter();
    }

    ///////////////////////////////////////////////////////////////////////////
    response primary_namespace_base::statistics_counter(request const& req,
        error_code& ec)
    {
        LAGAS_(info) << "primary_namespace::statistics_counter";

        std::string name(req.get_statistics_counter_name());

        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec) return response();

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "primary_namespace_base::statistics_counter",
                "unknown performance counter (unrelated to AGAS)");
            return response();
        }

        namespace_action_code code = invalid_request;
        detail::counter_target target = detail::counter_target_invalid;
        for (std::size_t i = 0;
              i != detail::num_primary_namespace_services;
              ++i)
        {
            if (p.countername_ == detail::primary_namespace_services[i].name_)
            {
                code = detail::primary_namespace_services[i].code_;
                target = detail::primary_namespace_services[i].target_;
                break;
            }
        }

        if (code == invalid_request || target == detail::counter_target_invalid)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "primary_namespace_base::statistics_counter",
                "unknown performance counter (unrelated to AGAS?)");
            return response();
        }

        util::function_nonser<boost::int64_t(bool)> get_data_func =
            counter_data_.get_counter_function(target, code, ec);
        if (ec) return response();

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

    // register all performance counter types exposed by this component
    void primary_namespace_base::register_counter_types(error_code& ec)
    {
        using util::placeholders::_1;
        using util::placeholders::_2;
        boost::format help_count(
            "returns the number of invocations of the AGAS service '%s'");
        boost::format help_time(
            "returns the overall execution time of the AGAS service '%s'");
        performance_counters::create_counter_func creator(
            util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
          , agas::detail::primary_namespace_service_name));

        for (std::size_t i = 0;
             i != detail::num_primary_namespace_services;
             ++i)
        {
            // global counters are handled elsewhere
            if (detail::primary_namespace_services[i].code_ ==
                primary_ns_statistics_counter)
            {
                continue;
            }

            std::string name(detail::primary_namespace_services[i].name_);
            std::string help;
            std::string::size_type p = name.find_last_of('/');
            HPX_ASSERT(p != std::string::npos);

            if (detail::primary_namespace_services[i].target_ ==
                detail::counter_target_count)
            {
                help = boost::str(help_count % name.substr(p+1));
            }
            else
            {
                help = boost::str(help_time % name.substr(p+1));
            }

            performance_counters::install_counter_type(
                agas::performance_counter_basename + name,
                performance_counters::counter_raw, help, creator,
                &performance_counters::locality_counter_discoverer,
                HPX_PERFORMANCE_COUNTER_V1,
                detail::primary_namespace_services[i].uom_, ec);
            if (ec) return;
        }
    }

    void primary_namespace_base::register_global_counter_types(error_code& ec)
    {
        using util::placeholders::_1;
        using util::placeholders::_2;

        performance_counters::create_counter_func creator(
            util::bind(&performance_counters::agas_raw_counter_creator, _1, _2
          , agas::detail::primary_namespace_service_name));

        for (std::size_t i = 0;
             i != detail::num_primary_namespace_services;
             ++i)
        {
            // local counters are handled elsewhere
            if (detail::primary_namespace_services[i].code_
                != primary_ns_statistics_counter)
            {
                continue;
            }

            std::string help;
            if (detail::primary_namespace_services[i].target_
                == detail::counter_target_count)
            {
                help = "returns the overall number of invocations "
                       " of all primary AGAS services";
            }
            else
            {
                help = "returns the overall execution time of all primary AGAS "
                       "services";
            }

            performance_counters::install_counter_type(
                std::string(agas::performance_counter_basename) +
                    detail::primary_namespace_services[i].name_,
                performance_counters::counter_raw, help, creator,
                &performance_counters::locality_counter_discoverer,
                HPX_PERFORMANCE_COUNTER_V1,
                detail::primary_namespace_services[i].uom_, ec);
            if (ec) return;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void primary_namespace_base::register_server_instance(
        char const* servicename, boost::uint32_t locality_id, error_code& ec)
    {
        // set locality_id for this component
        if (locality_id == naming::invalid_locality_id)
            locality_id = 0;        // if not given, we're on the root

        this->base_type::set_locality_id(locality_id);

        // now register this AGAS instance with AGAS :-P
        instance_name_ = agas::service_name;
        instance_name_ += servicename;
        instance_name_ += agas::detail::primary_namespace_service_name;

        // register a gid (not the id) to avoid AGAS holding a reference to this
        // component
        agas::register_name_sync(instance_name_, get_unmanaged_id().get_gid(), ec);
    }

    void primary_namespace_base::unregister_server_instance(error_code& ec)
    {
        agas::unregister_name_sync(instance_name_, ec);
        this->base_type::finalize();
    }
}}}

