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

namespace hpx { namespace agas { namespace server
{
    bool primary_namespace::resolve_gid_locked2(naming::gid_type const& gid, 
        naming::address& addr, error_code& ec)
    {
        boost::fusion::vector2<naming::gid_type, gva> r =
            resolve_gid_locked(gid, ec);
        if (ec || boost::fusion::at_c<0>(r) == naming::invalid_gid)
            return false;

        gva const g = boost::fusion::at_c<1>(r).resolve(gid, boost::fusion::at_c<0>(r));

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva();
        return true;
    }

    response primary_namespace::route(request const& req, error_code& ec)
    { // {{{ route implementation
        parcelset::parcel p = req.get_parcel();

        std::vector<naming::gid_type> const& gids = p.get_destinations();
        std::vector<naming::address>& addrs = p.get_destination_addrs();

        // resolve destination addresses, we should be able to resolve all of 
        // them, otherwise it's an error
        {
            mutex_type::scoped_lock l(mutex_);

            if (!locality_)
                locality_ = get_runtime().here();

            for (std::size_t i = 0; i != gids.size(); ++i)
            {
                if (!addrs[i])
                {
                    if (!resolve_gid_locked2(gids[i], addrs[i], ec) || ec)
                    {
                        return response(primary_ns_route, naming::invalid_gid,
                            gva(), no_success);
                    }
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
        naming::id_type source = get_colocation_id(p.get_source());
        for (std::size_t i = 0; i != gids.size(); ++i)
        {
            components::stubs::runtime_support::insert_agas_cache_entry(
                source, gids[i], addrs[i]);
        }

        return response(primary_ns_route, success);
    } // }}}
}}}

