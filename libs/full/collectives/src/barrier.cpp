//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/type_support.hpp>

#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/barrier.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

///////////////////////////////////////////////////////////////////////////////
namespace hpx::traits {

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
}    // namespace hpx::traits

#endif

namespace hpx::distributed {

    barrier::barrier([[maybe_unused]] std::string const& basename,
        [[maybe_unused]] hpx::collectives::generation_arg generation,
        [[maybe_unused]] hpx::collectives::root_site_arg root_site)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      : comm_(hpx::collectives::create_communicator(basename.c_str(),
            hpx::collectives::num_sites_arg{},
            hpx::collectives::this_site_arg{}, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        auto state = hpx::traits::detail::get_shared_state(comm_);
        auto f = [state]() mutable {
            using hpx::collectives::detail::communicator_data;
            state->get_extra_data<communicator_data>() =
                communicator_data{agas::get_num_localities(hpx::launch::sync),
                    agas::get_locality_id()};
        };
        state->set_on_completed(HPX_MOVE(f));
#endif
    }

    barrier::barrier([[maybe_unused]] std::string const& basename,
        [[maybe_unused]] hpx::collectives::num_sites_arg num,
        [[maybe_unused]] hpx::collectives::generation_arg generation,
        [[maybe_unused]] hpx::collectives::root_site_arg root_site)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      : comm_(hpx::collectives::create_communicator(basename.c_str(), num,
            hpx::collectives::this_site_arg{}, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        auto state = hpx::traits::detail::get_shared_state(comm_);
        auto f = [num, state]() mutable {
            using hpx::collectives::detail::communicator_data;
            state->get_extra_data<communicator_data>() =
                communicator_data{num, agas::get_locality_id()};
        };
        state->set_on_completed(HPX_MOVE(f));
#endif
    }

    barrier::barrier([[maybe_unused]] std::string const& basename,
        [[maybe_unused]] hpx::collectives::num_sites_arg num,
        [[maybe_unused]] hpx::collectives::this_site_arg rank,
        [[maybe_unused]] hpx::collectives::generation_arg generation,
        [[maybe_unused]] hpx::collectives::root_site_arg root_site)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      : comm_(hpx::collectives::create_communicator(
            basename.c_str(), num, rank, generation, root_site))
#endif
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        auto state = hpx::traits::detail::get_shared_state(comm_);
        auto f = [num, rank, state]() mutable {
            using hpx::collectives::detail::communicator_data;
            state->get_extra_data<communicator_data>() =
                communicator_data{num, rank};
        };
        state->set_on_completed(HPX_MOVE(f));
#endif
    }

    barrier::barrier(barrier const& rhs)
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      : comm_(rhs.comm_)
#endif
    {
    }
    barrier::barrier(barrier&& rhs) noexcept
      : generation_(rhs.generation_.load())
#if !defined(HPX_COMPUTE_DEVICE_CODE)
      , comm_(HPX_MOVE(rhs.comm_))
#endif
    {
    }

    barrier& barrier::operator=(barrier const& rhs)
    {
        generation_ = 0;
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        comm_ = rhs.comm_;
#endif
        return *this;
    }
    barrier& barrier::operator=(barrier&& rhs) noexcept
    {
        generation_.store(rhs.generation_.load());
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        comm_ = HPX_MOVE(rhs.comm_);
#endif
        return *this;
    }

    barrier::~barrier() = default;

    namespace {

        std::size_t find_rank(
            std::vector<std::size_t> const& ranks, std::size_t rank)
        {
            auto const rank_it = std::find(ranks.begin(), ranks.end(), rank);
            HPX_ASSERT(rank_it != ranks.end());

            return static_cast<std::size_t>(
                std::distance(ranks.begin(), rank_it));
        }
    }    // namespace

    barrier::barrier(std::string const& base_name,
        std::vector<std::size_t> const& ranks, std::size_t rank)
      : barrier(base_name, hpx::collectives::num_sites_arg(ranks.size()),
            hpx::collectives::this_site_arg(find_rank(ranks, rank)))
    {
    }

    barrier::barrier() = default;

    void barrier::detach()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        comm_.free();
#endif
        generation_ = 0;
    }

    void barrier::wait(hpx::collectives::generation_arg generation)
    {
        std::size_t this_site = 0;
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        this_site = comm_.get_info().second;
#endif
        LRT_(info).format(
            "hpx::distributed::barrier::wait: rank({}), generation({}): "
            "entering barrier",
            this_site, generation);

        wait(hpx::launch::async, generation).get();

        LRT_(info).format(
            "hpx::distributed::barrier::wait: rank({}), generation({}): "
            "exiting barrier",
            this_site, generation);
    }

    hpx::future<void> barrier::wait(
        hpx::launch::async_policy, hpx::collectives::generation_arg generation)
    {
        if (generation == 0)
        {
            return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::distributed::barrier",
                "the generation number shouldn't be zero"));
        }

        if (generation != static_cast<std::size_t>(-1))
        {
            std::size_t const prev_generation =
                generation_.exchange(generation);
            if (prev_generation >= generation)
            {
                return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                    hpx::error::bad_parameter, "hpx::distributed::barrier",
                    "the generation number must be continuously increasing"));
            }
        }
        else
        {
            // use the next available generation number, if none is given
            generation = ++generation_;
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        auto barrier_data = [this, generation](auto&&) -> hpx::future<void> {
            using action_type = hpx::collectives::detail::communicator_server::
                communication_get_action<traits::communication::barrier_tag,
                    hpx::future<void>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            auto const this_site = comm_.get_info().second;
            hpx::future<void> result =
                hpx::async(action_type(), comm_, this_site, generation);

            if (!result.is_ready())
            {
                traits::detail::get_shared_state(result)->set_on_completed(
                    [comm = comm_] { HPX_UNUSED(comm); });
            }

            return result;
        };

        if (comm_.is_ready())
        {
            return barrier_data(comm_);
        }
        return comm_.then(hpx::launch::sync, HPX_MOVE(barrier_data));
#else
        return hpx::make_ready_future<void>();
#endif
    }

    void barrier::create_global_barrier()
    {
        util::runtime_configuration const& cfg = get_runtime().get_config();
        get_global_barrier() = barrier("/0/hpx/global_barrier",
            hpx::collectives::num_sites_arg(cfg.get_num_localities()),
            hpx::collectives::this_site_arg(cfg.get_locality()));
    }

    barrier& barrier::get_global_barrier()
    {
        static barrier b;
        return b;
    }

    void barrier::synchronize(hpx::collectives::generation_arg generation)
    {
        get_global_barrier().wait(generation);
    }
}    // namespace hpx::distributed
