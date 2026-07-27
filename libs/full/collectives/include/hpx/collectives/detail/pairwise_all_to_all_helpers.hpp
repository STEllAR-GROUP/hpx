//  Copyright (c) 2026 Anshuman Agrawal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file detail/pairwise_all_to_all_helpers.hpp
/// Direct site-to-site exchange for the all_to_all collective. The default
/// all_to_all routes every contribution through one communicator site, which
/// concentrates the whole exchange on a single locality. This variant sends
/// each row straight to the site it belongs to, trading two messages per site
/// for num_sites - 1 of them. That trade only pays for large rows, so the
/// caller decides which path to take.

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/channel_communicator.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace hpx::collectives::detail {

    ///////////////////////////////////////////////////////////////////////////
    // pairwise_all_to_all
    //
    // Exchanges one row per participating site over a channel communicator,
    // without a root site relaying the data.
    //
    // local_result must hold exactly num_sites rows, where row d is destined
    // for site d. The returned vector holds one row per source site, so entry
    // s is what site s contributed for this site. That is the same contract
    // the communicator-based all_to_all offers, which lets either path serve
    // the same call.
    //
    // Sends are staggered: site h starts at peer h + 1 rather than at peer 0,
    // so the sites do not all target the same peer at the same time. The
    // receives are staggered in the opposite direction for the same reason.
    //
    // The tag separates concurrent exchanges on one channel communicator, so
    // the caller must pass a distinct tag per generation.
    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename T>
    hpx::future<std::vector<T>> pairwise_all_to_all(channel_communicator comm,
        std::vector<T>&& local_result, std::size_t const num_sites,
        std::size_t const this_site, tag_arg const tag)
    {
        if (num_sites == 0)
        {
            return hpx::make_exceptional_future<std::vector<T>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::detail::pairwise_all_to_all",
                    "the number of participating sites must not be zero"));
        }

        if (this_site >= num_sites)
        {
            return hpx::make_exceptional_future<std::vector<T>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::detail::pairwise_all_to_all",
                    "the local site must be smaller than the number of "
                    "participating sites"));
        }

        if (local_result.size() != num_sites)
        {
            return hpx::make_exceptional_future<std::vector<T>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::detail::pairwise_all_to_all",
                    "each participating site must contribute exactly "
                    "num_sites elements"));
        }

        // A single site exchanges with itself only.
        if (num_sites == 1)
        {
            return hpx::make_ready_future(HPX_MOVE(local_result));
        }

        std::vector<hpx::future<void>> sends;
        sends.reserve(num_sites - 1);

        std::vector<hpx::future<T>> receives;
        receives.reserve(num_sites - 1);

        for (std::size_t k = 1; k != num_sites; ++k)
        {
            std::size_t const destination = (this_site + k) % num_sites;
            sends.push_back(set(comm, that_site_arg(destination),
                HPX_MOVE(local_result[destination]), tag));

            std::size_t const source = (this_site + num_sites - k) % num_sites;
            receives.push_back(get<T>(comm, that_site_arg(source), tag));
        }

        // The diagonal never travels; it is this site's own contribution.
        T diagonal = HPX_MOVE(local_result[this_site]);

        return hpx::when_all(HPX_MOVE(sends), HPX_MOVE(receives))
            .then(hpx::launch::sync,
                [comm = HPX_MOVE(comm), diagonal = HPX_MOVE(diagonal),
                    num_sites, this_site](auto&& f) mutable {
                    HPX_UNUSED(comm);

                    auto [sent, received] = HPX_MOVE(f).get();

                    // Report a failed send rather than a truncated result.
                    for (auto& send : sent)
                    {
                        send.get();
                    }

                    std::vector<T> result(num_sites);
                    result[this_site] = HPX_MOVE(diagonal);

                    // received[i] came from the peer i + 1 hops behind this
                    // site, matching the receive order posted above.
                    for (std::size_t k = 1; k != num_sites; ++k)
                    {
                        std::size_t const source =
                            (this_site + num_sites - k) % num_sites;
                        result[source] = received[k - 1].get();
                    }

                    return result;
                });
    }
}    // namespace hpx::collectives::detail

#endif    // !HPX_COMPUTE_DEVICE_CODE
