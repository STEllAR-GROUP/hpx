//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/detail/primary_namespace_base.hpp>

namespace hpx { namespace agas { namespace detail
{
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
            get_counter_function(target, code, ec);
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
}}}

