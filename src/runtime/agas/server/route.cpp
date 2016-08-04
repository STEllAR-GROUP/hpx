//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/scoped_timer.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace server
{
    void primary_namespace::route(parcelset::parcel && p)
    { // {{{ route implementation
        util::scoped_timer<boost::atomic<boost::int64_t> > update(
            counter_data_.route_.time_
        );
        counter_data_.increment_route_count();

        std::size_t size = p.size();
        naming::id_type const* ids = p.destinations();
        naming::address* addrs = p.addrs();
        std::vector<resolved_type> cache_addresses;

        runtime& rt = get_runtime();

        // resolve destination addresses, we should be able to resolve all of
        // them, otherwise it's an error
        {
            std::unique_lock<mutex_type> l(mutex_);

            cache_addresses.reserve(size);
            for (std::size_t i = 0; i != size; ++i)
            {
                naming::gid_type gid(ids[i].get_gid());

                // wait for any migration to be completed
                wait_for_migration_locked(l, gid, hpx::throws);

                cache_addresses.push_back(resolve_gid_locked(l, gid, hpx::throws));
                resolved_type& r = cache_addresses.back();

                if (hpx::util::get<0>(r) == naming::invalid_gid)
                {
                    id_type const id = ids[i];
                    l.unlock();

                    HPX_THROW_EXCEPTION(no_success,
                        "primary_namespace::route",
                        boost::str(boost::format(
                                "can't route parcel to unknown gid: %s"
                            ) % id));
                }

                // retain don't store in cache flag
                if (!naming::detail::store_in_cache(gid))
                {
                    naming::detail::set_dont_store_in_cache(
                        hpx::util::get<0>(r));
                }

                gva const g = hpx::util::get<1>(r).resolve(
                    ids[i].get_gid(), hpx::util::get<0>(r));

                addrs[i].locality_ = g.prefix;
                addrs[i].type_ = g.type;
                addrs[i].address_ = g.lva();
            }
        }

        naming::id_type source = p.source_id();

        // either send the parcel on its way or execute actions locally
        if (addrs[0].locality_ == get_locality())
        {
            // destination is local
            rt.get_applier().schedule_action(std::move(p));
        }
        else
        {
            // destination is remote
            rt.get_parcel_handler().put_parcel(std::move(p));
        }

        if (rt.get_state() < state_pre_shutdown)
        {
            // asynchronously update cache on source locality
            for (std::size_t i = 0; i != cache_addresses.size(); ++i)
            {
                // update remote cache if the id is not flagged otherwise
                resolved_type const& r = cache_addresses[i];
                naming::gid_type const& id = hpx::util::get<0>(r);
                if (id && naming::detail::store_in_cache(id))
                {
                    gva const& g = hpx::util::get<1>(r);
                    naming::address addr(g.prefix, g.type, g.lva());

                    using components::stubs::runtime_support;
                    runtime_support::update_agas_cache_entry_colocated(
                        source, id, addr, g.count, g.offset);
                }
            }
        }
    } // }}}
}}}

