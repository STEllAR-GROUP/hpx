//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/barrier.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/type_support/construct_at.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // support for barrier
    namespace communication {
        struct barrier_tag;
    }    // namespace communication

    template <typename Communicator>
    struct communication_operation<Communicator, communication::barrier_tag>
    {
        template <typename Result>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation)
        {
            return communicator.template handle_data<void>(which, generation);
        }
    };
}}    // namespace hpx::traits

#endif

namespace hpx { namespace distributed {

    namespace coll = hpx::collectives;

    barrier::barrier(std::string const& basename,
        coll::generation_arg generation, coll::root_site_arg root_site)
      : generation_(0)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      , comm_(coll::create_communicator(
            basename.c_str(), {}, {}, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::traits::detail::get_shared_state(comm_)->set_on_completed(
            [this]() {
                comm_.set_info(coll::num_sites_arg(
                                   agas::get_num_localities(hpx::launch::sync)),
                    coll::this_site_arg(agas::get_locality_id()));
            });
#else
        HPX_UNUSED(basename);
        HPX_UNUSED(generation);
        HPX_UNUSED(root_site);
#endif
    }

    barrier::barrier(std::string const& basename, coll::num_sites_arg num,
        coll::generation_arg generation, coll::root_site_arg root_site)
      : generation_(0)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      , comm_(coll::create_communicator(
            basename.c_str(), num, {}, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::traits::detail::get_shared_state(comm_)->set_on_completed(
            [this, num]() {
                comm_.set_info(
                    num, coll::this_site_arg(agas::get_locality_id()));
            });
#else
        HPX_UNUSED(basename);
        HPX_UNUSED(num);
        HPX_UNUSED(generation);
        HPX_UNUSED(root_site);
#endif
    }

    barrier::barrier(std::string const& basename, coll::num_sites_arg num,
        coll::this_site_arg rank, coll::generation_arg generation,
        coll::root_site_arg root_site)
      : generation_(0)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      , comm_(coll::create_communicator(
            basename.c_str(), num, rank, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::traits::detail::get_shared_state(comm_)->set_on_completed(
            [this, num, rank]() { this->comm_.set_info(num, rank); });
#else
        HPX_UNUSED(basename);
        HPX_UNUSED(num);
        HPX_UNUSED(rank);
        HPX_UNUSED(generation);
        HPX_UNUSED(root_site);
#endif
    }

    static std::size_t find_rank(
        std::vector<std::size_t> const& ranks, std::size_t rank)
    {
        auto const rank_it = std::find(ranks.begin(), ranks.end(), rank);
        HPX_ASSERT(rank_it != ranks.end());

        return static_cast<std::size_t>(std::distance(ranks.begin(), rank_it));
    }

    barrier::barrier(std::string const& base_name,
        std::vector<std::size_t> const& ranks, std::size_t rank)
      : barrier(base_name, coll::num_sites_arg(ranks.size()),
            coll::this_site_arg(find_rank(ranks, rank)))
    {
    }

    barrier::barrier() = default;

    void barrier::detach()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        comm_.free();
#endif
    }

    void barrier::wait(coll::generation_arg generation) const
    {
        wait(hpx::launch::async, generation).get();
    }

    hpx::future<void> barrier::wait(
        hpx::launch::async_policy, coll::generation_arg generation) const
    {
        if (generation == 0)
        {
            return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                hpx::bad_parameter, "hpx::distributed::barrier",
                "the generation number shouldn't be zero"));
        }

        if (generation != std::size_t(-1))
        {
            std::size_t prev_generation = generation_.exchange(generation);
            if (prev_generation >= generation)
            {
                return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                    hpx::bad_parameter, "hpx::distributed::barrier",
                    "the generation number be must continuously increasing"));
            }
        }
        else
        {
            // use the next available gneration number, if none is given
            generation = ++generation_;
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        auto barrier_data = [this, generation](auto&&) -> hpx::future<void> {
            using action_type = typename coll::detail::communicator_server::
                template communication_get_action<
                    traits::communication::barrier_tag, hpx::future<void>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<void> result = hpx::async(
                action_type(), comm_, comm_.get_info().second, generation);

            if (!result.is_ready())
            {
                traits::detail::get_shared_state(result)->set_on_completed(
                    [comm = comm_]() { HPX_UNUSED(comm); });
            }

            return result;
        };

        return comm_.then(hpx::launch::sync, HPX_MOVE(barrier_data));
#else
        return hpx::make_ready_future<void>();
#endif
    }

    static barrier create_global_barrier()
    {
        runtime& rt = get_runtime();
        util::runtime_configuration const& cfg = rt.get_config();
        barrier b1("/0/hpx/global_barrier0",
            coll::num_sites_arg(cfg.get_num_localities()),
            coll::this_site_arg(cfg.get_locality()));
        barrier b2("/0/hpx/global_barrier1",
            static_cast<std::size_t>(cfg.get_num_localities()));
        return {{HPX_MOVE(b1), HPX_MOVE(b2)}};
    }

    struct barrier_tag
    {
    };

    barrier& barrier::get_global_barrier()
    {
        using static_type =
            hpx::util::reinitializable_static<barrier, barrier_tag, 1,
                hpx::util::reinitializable_static_init_mode::function>;

        static_type b(&create_global_barrier);
        return b.get();
    }

    void barrier::synchronize(coll::generation_arg generation)
    {
        get_global_barrier().wait(generation);
    }
}}    // namespace hpx::distributed
