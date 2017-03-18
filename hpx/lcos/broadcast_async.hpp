//  Copyright (c) 2013-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file broadcast_async.hpp

#ifndef HPX_LCOS_BROADCAST_ASYNC_HPP
#define HPX_LCOS_BROADCAST_ASYNC_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/calculate_fanout.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <boost/format.hpp>
#include <boost/preprocessor/cat.hpp>

#if !defined(HPX_BROADCAST_FANOUT)
#define HPX_BROADCAST_FANOUT 16
#endif

namespace hpx { namespace lcos
{
    template <typename T>
    hpx::future<T> broadcast_here(char const* basename,
        std::size_t this_site = std::size_t(-1),
        std::size_t generation = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        hpx::lcos::promise<T> p;
        hpx::future<T> f = p.get_future();

        // register promise using symbolic name
        hpx::future<bool> was_registered =
            hpx::register_with_basename(name, p.get_id(), this_site);

        return hpx::dataflow(
            [](hpx::future<T> f, hpx::future<bool> was_registered,
                std::string && name, std::size_t this_site)
            {
                was_registered.get();       // rethrow exceptions

                // make sure promise gets unregistered after use
                hpx::unregister_with_basename(name, this_site).get();

                return f.get();
            },
            std::move(f), std::move(was_registered), std::move(name), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        std::map<hpx::id_type, std::vector<std::size_t> >
        generate_locality_indices(std::string const& name, std::size_t num_sites)
        {
            std::map<hpx::id_type, std::vector<std::size_t> > indices;
            for (std::size_t i = 0; i != num_sites; ++i)
            {
                hpx::id_type service_locality =
                    agas::symbol_namespace::symbol_namespace_locality(
                        hpx::detail::name_from_basename(name, i));
                indices[service_locality].push_back(i);
            }
            return indices;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        hpx::future<void> broadcast_there_invoke(std::string const& name,
            std::size_t site, T const& t)
        {
            // this should be always executed on the locality responsible for
            // resolving the given name
            HPX_ASSERT(
                naming::get_locality_id_from_id(
                    agas::symbol_namespace::symbol_namespace_locality(
                        hpx::detail::name_from_basename(name, site))) ==
                hpx::get_locality_id()
            );

            // find_from_basename is always a local operation (see assert above)
            hpx::future<hpx::id_type> f = hpx::find_from_basename(name, site);
            return f.then(
                [&](hpx::future<hpx::id_type> f)
                {
                    return set_lco_value(f.get(), t);
                });
        }

        template <typename T>
        struct broadcast_there
        {
            static hpx::future<void> call(std::string const& name,
                std::size_t num_sites, std::vector<std::size_t> const& sites,
                T const& t)
            {
                // first apply actual broadcast operation to first set of sites
                std::vector<hpx::future<void> > futures;
                futures.reserve(sites.size());
                for(std::size_t i : sites)
                {
                    futures.push_back(detail::broadcast_there_invoke(name, i, t));
                }
                return hpx::when_all(futures);
            }
        };

        template <typename T>
        struct make_broadcast_there_action
        {
            typedef typename HPX_MAKE_ACTION(broadcast_there<T>::call)::type type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<void> broadcast_there(char const* basename, T const& t,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1))
    {
        if (num_sites == std::size_t(-1))
            num_sites = hpx::get_num_localities(hpx::launch::sync);

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        std::map<hpx::id_type, std::vector<std::size_t> > locality_indices =
            detail::generate_locality_indices(name, num_sites);

        typedef
            typename lcos::detail::make_broadcast_there_action<T>::type
            broadcast_there_action;

        std::vector<hpx::future<void> > futures;
        futures.resize(locality_indices.size());
        for (auto const& p : locality_indices)
        {
            futures.push_back(hpx::detail::async_colocated(
                broadcast_there_action(), p.first, name, num_sites, p.second, t));
        }
        return hpx::when_all(futures);
    }
}}

#define HPX_BROADCAST_ASYNC_DECLARATION(Type)                                 \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::detail::make_broadcast_there_action<Type>::type,           \
        BOOST_PP_CAT(broadcast_async_, Type))                                 \
/**/
#define HPX_BROADCAST_ASYNC(Type)                                             \
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::detail::make_broadcast_there_action<Type>::type,           \
        BOOST_PP_CAT(broadcast_async_, Type))                                 \
/**/

#endif
