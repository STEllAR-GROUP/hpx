//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/format.hpp>

namespace hpx { namespace agas { namespace server
{
    response primary_namespace::route(request const& req, error_code& ec)
    { // {{{ route implementation
        parcelset::parcel p = req.get_parcel();

        std::size_t size = p.size();
        naming::id_type const* ids = p.get_destinations();
        naming::address* addrs = p.get_destination_addrs();
        std::vector<resolved_type> cache_addresses;

        runtime& rt = get_runtime();

        // resolve destination addresses, we should be able to resolve all of
        // them, otherwise it's an error
        {
            mutex_type::scoped_lock l(mutex_);

            cache_addresses.reserve(size);
            for (std::size_t i = 0; i != size; ++i)
            {
                if (!addrs[i])
                {
                    naming::gid_type gid(ids[i].get_gid());

                    // wait for any migration to be completed
                    wait_for_migration_locked(l, gid, ec);

                    cache_addresses.push_back(resolve_gid_locked(gid, ec));
                    resolved_type const& r = cache_addresses.back();

                    if (ec || boost::fusion::at_c<0>(r) == naming::invalid_gid)
                    {
                        id_type const id = ids[i];
                        l.unlock();

                        HPX_THROWS_IF(ec, no_success,
                            "primary_namespace::route",
                            boost::str(boost::format(
                                    "can't route parcel to unknown gid: %s"
                                ) % id));

                        return response(primary_ns_route, no_success);
                    }

                    gva const g = boost::fusion::at_c<1>(r).resolve(
                        ids[i].get_gid(), boost::fusion::at_c<0>(r));

                    addrs[i].locality_ = g.prefix;
                    addrs[i].type_ = g.type;
                    addrs[i].address_ = g.lva();
                }
            }
        }

        // either send the parcel on its way or execute actions locally
        if (addrs[0].locality_ == get_locality())
        {
            // destination is local
            rt.get_applier().schedule_action(p);
        }
        else
        {
            // destination is remote
            rt.get_parcel_handler().put_parcel(p);
        }

        if (rt.get_state() < state_pre_shutdown)
        {
            // asynchronously update cache on source locality
            naming::id_type source = p.get_source();
            for (std::size_t i = 0; i != cache_addresses.size(); ++i)
            {
                resolved_type const& r = cache_addresses[i];
                if (boost::fusion::at_c<0>(r))
                {
                    gva const& g = boost::fusion::at_c<1>(r);
                    naming::address addr(g.prefix, g.type, g.lva());

                    using components::stubs::runtime_support;
                    runtime_support::update_agas_cache_entry_colocated(
                        source, boost::fusion::at_c<0>(r), addr, g.count, g.offset);
                }
            }
        }

        return response(primary_ns_route, success);
    } // }}}
}}}

