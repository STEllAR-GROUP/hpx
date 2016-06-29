//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/detail/local_primary_namespace.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/assert.hpp>

#include <vector>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

namespace hpx { namespace agas { namespace detail
{
    response local_primary_namespace::service(request const& req, error_code& ec)
    {
        switch (req.get_action_code())
        {
            case primary_ns_route:
                {
                    HPX_ASSERT(false);
                    return response();
                }
            case primary_ns_bind_gid:
                {
                    update_time_on_exit update(
                        counter_data_.bind_gid_.time_
                    );
                    counter_data_.increment_bind_gid_count();
//                     return bind_gid(req, ec);
                }
            case primary_ns_resolve_gid:
                {
                    update_time_on_exit update(
                        counter_data_.resolve_gid_.time_
                    );
                    counter_data_.increment_resolve_gid_count();
//                     return resolve_gid(req, ec);
                }
            case primary_ns_unbind_gid:
                {
                    update_time_on_exit update(
                        counter_data_.unbind_gid_.time_
                    );
                    counter_data_.increment_unbind_gid_count();
//                     return unbind_gid(req, ec);
                }
            case primary_ns_increment_credit:
                {
                    update_time_on_exit update(
                        counter_data_.increment_credit_.time_
                    );
                    counter_data_.increment_increment_credit_count();
//                     return increment_credit(req, ec);
                }
            case primary_ns_decrement_credit:
                {
                    update_time_on_exit update(
                        counter_data_.decrement_credit_.time_
                    );
                    counter_data_.increment_decrement_credit_count();
//                     return decrement_credit(req, ec);
                }
            case primary_ns_allocate:
                {
                    update_time_on_exit update(
                        counter_data_.allocate_.time_
                    );
                    counter_data_.increment_allocate_count();
                    return allocate(req, ec);
                }
            case primary_ns_begin_migration:
                {
                    update_time_on_exit update(
                        counter_data_.begin_migration_.time_
                    );
                    counter_data_.increment_begin_migration_count();
//                     return begin_migration(req, ec);
                }
            case primary_ns_end_migration:
                {
                    update_time_on_exit update(
                        counter_data_.end_migration_.time_
                    );
                    counter_data_.increment_end_migration_count();
//                     return end_migration(req, ec);
                }
            case primary_ns_statistics_counter:
//                 return statistics_counter(req, ec);
                break;

            case locality_ns_allocate:
            case locality_ns_free:
            case locality_ns_localities:
            case locality_ns_num_localities:
            case locality_ns_num_threads:
            case locality_ns_resolve_locality:
            case locality_ns_resolved_localities:
            {
                LAGAS_(warning) <<
                    "local_primary_namespace::service, redirecting request to "
                    "locality_namespace";
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
                    "local_primary_namespace::service, redirecting request to "
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
                    "local_primary_namespace::service, redirecting request to "
                    "symbol_namespace";
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
                  , "local_primary_namespace::service"
                  , boost::str(boost::format(
                        "invalid action code encountered in request, "
                        "action_code(%x)")
                        % boost::uint16_t(req.get_action_code())));
                return response();
            }
        }

        return response();
    }

    std::vector<response> local_primary_namespace::bulk_service(
        std::vector<request> const& reqs, error_code& ec)
    {
        std::vector<response> r;
        r.reserve(reqs.size());

        for (request const& req : reqs)
        {
            r.push_back(service(req, ec));
            if (ec)
                break;      // on error: for now stop iterating
        }

        return r;
    }

    ///////////////////////////////////////////////////////////////////////////
    response local_primary_namespace::allocate(request const& req, error_code& ec)
    {
        boost::uint64_t const count = req.get_count();
        boost::uint64_t const real_count = (count) ? (count - 1) : (0);

//     // Just return the prefix
//     // REVIEW: Should this be an error?
//     if (0 == count)
//     {
//         LAGAS_(info) << (boost::format(
//             "primary_namespace::allocate, count(%1%), "
//             "lower(%1%), upper(%3%), prefix(%4%), response(repeated_request)")
//             % count % next_id_ % next_id_
//             % naming::get_locality_id_from_gid(next_id_));
//
//         if (&ec != &throws)
//             ec = make_success_code();
//
//         return response(primary_ns_allocate, next_id_, next_id_
//           , naming::get_locality_id_from_gid(next_id_), success);
//     }

        // Compute the new allocation.
        naming::gid_type lower;
        naming::gid_type upper(lower + real_count);

//     // Check for overflow.
//     if (upper.get_msb() != lower.get_msb())
//     {
//         // Check for address space exhaustion (we currently use 86 bits of
//         // the gid for the actual id)
//         if (HPX_UNLIKELY(
//             (lower.get_msb() & naming::gid_type::virtual_memory_mask) ==
//                 naming::gid_type::virtual_memory_mask)
//            )
//         {
//             HPX_THROWS_IF(ec, internal_server_error
//                 , "locality_namespace::allocate"
//                 , "primary namespace has been exhausted");
//             return response();
//         }
//
//         // Otherwise, correct
//         lower = naming::gid_type(upper.get_msb(), 0);
//         upper = lower + real_count;
//     }
//
//     // Store the new upper bound.
//     next_id_ = upper;
//
//     // Set the initial credit count.
//     naming::detail::set_credit_for_gid(lower, boost::int64_t(HPX_GLOBALCREDIT_INITIAL));
//     naming::detail::set_credit_for_gid(upper, boost::int64_t(HPX_GLOBALCREDIT_INITIAL));
//
//     LAGAS_(info) << (boost::format(
//         "primary_namespace::allocate, count(%1%), "
//         "lower(%2%), upper(%3%), prefix(%4%), response(repeated_request)")
//         % count % lower % upper % naming::get_locality_id_from_gid(next_id_));

        if (&ec != &throws)
            ec = make_success_code();

        return response(locality_ns_allocate, lower, upper, 0, success);
    }
}}}

