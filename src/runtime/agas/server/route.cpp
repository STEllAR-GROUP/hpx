//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2007-2013 Hartmut Kaiser
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
        std::vector<boost::fusion::vector2<naming::gid_type, gva> > cache_addresses;

        // resolve destination addresses, we should be able to resolve all of
        // them, otherwise it's an error
        {
            mutex_type::scoped_lock l(mutex_);

            if (!locality_)
                locality_ = get_runtime().here();

            cache_addresses.reserve(size);
            for (std::size_t i = 0; i != size; ++i)
            {
                if (!addrs[i])
                {
                    cache_addresses.push_back(resolve_gid_locked(ids[i].get_gid(), ec));
                    boost::fusion::vector2<naming::gid_type, gva> const& r =
                        cache_addresses.back();

                    if (ec || boost::fusion::at_c<0>(r) == naming::invalid_gid)
                    {
                        HPX_THROWS_IF(ec, no_success,
                            "primary_namespace::route",
                            boost::str(boost::format(
                                    "can't route parcel to unknown gid: %s"
                                ) % ids[i]));
                        return response(primary_ns_route, naming::invalid_gid,
                            gva(), no_success);
                    }

                    gva const g = boost::fusion::at_c<1>(r).resolve(
                        ids[i].get_gid(), boost::fusion::at_c<0>(r));

                    addrs[i].locality_ = g.endpoint;
                    addrs[i].type_ = g.type;
                    addrs[i].address_ = g.lva();
                }
                else
                {
                    cache_addresses.push_back(
                        boost::fusion::make_vector(naming::gid_type(), gva()));
                }
            }
        }

        // either send the parcel on its way or execute actions locally
        if (addrs[0].locality_ == locality_)
        {
            // destination is local
            get_runtime().get_applier().schedule_action(p);
        }
        else
        {
            // destination is remote
            get_runtime().get_parcel_handler().put_parcel(p);
        }

        // asynchronously update cache on source locality
        naming::id_type source = get_colocation_id_sync(p.get_source());
        for (std::size_t i = 0; i != size; ++i)
        {
            boost::fusion::vector2<naming::gid_type, gva> const& r =
                cache_addresses[i];

            if (boost::fusion::at_c<0>(r))
            {
                gva const& g = boost::fusion::at_c<1>(r);
                naming::address addr(g.endpoint, g.type, g.lva());

                components::stubs::runtime_support::update_agas_cache_entry(
                    source, boost::fusion::at_c<0>(r), addr, g.count, g.offset);
            }
        }

        return response(primary_ns_route, success);
    } // }}}
}}}

