//  Copyright (c) 2019-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  basename    The base name identifying the exclusive_scan operation
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on
    ///                     the given base name. This is optional and needs to
    ///                     be supplied only if the exclusive_scan operation on
    ///                     the given base name has to be performed more than
    ///                     once. The generation number (if given) must be a
    ///                     positive number greater than zero.
    /// \param  root_site   The site that is responsible for creating the
    ///                     exclusive_scan support object. This value is
    ///                     optional and defaults to '0' (zero).
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns a future
    ///             the reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on participating site on process 1 is always
    ///             the value passed in by participating site 1. The returned
    ///             future will become ready once the exclusive_scan operation
    ///             has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the exclusive_scan operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns a future
    ///             the reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on participating site on process 1 is always
    ///             the value passed in by participating site 1. The returned
    ///             future will become ready once the exclusive_scan operation
    ///             has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(
        communicator comm, T&& result, F&& op,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the exclusive_scan operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns a future
    ///             the reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on participating site on process 1 is always
    ///             the value passed in by participating site 1. The returned
    ///             future will become ready once the exclusive_scan operation
    ///             has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(
        communicator comm, T&& result, F&& op,
        generation_arg generation,
        this_site_arg this_site = this_site_arg());

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  basename    The base name identifying the exclusive_scan operation
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the exclusive_scan operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param  root_site   The site that is responsible for creating the
    ///                     exclusive_scan support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns a future
    ///             the reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on participating site on process 1 is always
    ///             the value passed in by participating site 1. The returned
    ///             future will become ready once the exclusive_scan operation
    ///             has been completed.
    ///
    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy,
        char const* basename, T&& result, F&& op,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on
    ///                     the given base name. This is optional and needs to
    ///                     be supplied only if the exclusive_scan operation on
    ///                     the given base name has to be performed more than
    ///                     once. The generation number (if given) must be a
    ///                     positive number greater than zero.
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns a future
    ///             the reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on participating site on process 1 is always
    ///             the value passed in by participating site 1. The returned
    ///             future will become ready once the exclusive_scan operation
    ///             has been completed.
    ///
    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator comm,
        T&& result, F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Exclusive scan a set of values from different call sites
    ///
    /// This function performs an exclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the exclusive_scan operation performed on
    ///                     the given base name. This is optional and needs to
    ///                     be supplied only if the exclusive_scan operation on
    ///                     the given base name has to be performed more than
    ///                     once. The generation number (if given) must be a
    ///                     positive number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       The result returned on the root_site is always the same as
    ///             the result returned on thus_site == 1 and is the same as the
    ///             value provided by the root_site.
    ///
    /// \returns    For the participating site i this function returns the
    ///             reduction (calculated according to the function op) of
    ///             the values passed in by the participating sites 0, ..., i-1.
    ///             The value returned on participating site 0 is undefined. The
    ///             value returned on
    ///             participating site  on process 1 is always the value passed
    ///             in by participating site 1. The returned future will become
    ///             ready once the exclusive_scan operation has been completed.
    ///
    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator comm,
        T&& result, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg());
}}    // namespace hpx::collectives

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::collectives::detail {

    template <typename T, typename InIter, typename Sent, typename OutIter,
        typename Op>
    static constexpr void exclusive_scan(
        InIter first, Sent last, OutIter dest, Op&& op)
    {
        // the first value given goes to the second destination
        T temp = *first++;
        ++dest;    // the first output is ignored
        for (/* */; first != last; (void) ++first, ++dest)
        {
            T next = HPX_INVOKE(op, temp, *first);
            *dest = temp;
            temp = next;
        }
    }
}    // namespace hpx::collectives::detail

namespace hpx::traits {

    namespace communication {

        struct exclusive_scan_tag;
        struct exclusive_scan_init_tag;

        template <>
        struct communicator_data<exclusive_scan_tag>
        {
            HPX_EXPORT static char const* name() noexcept;
        };

        template <>
        struct communicator_data<exclusive_scan_init_tag>
        {
            HPX_EXPORT static char const* name() noexcept;
        };
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for exclusive_scan
    template <typename Communicator>
    struct communication_operation<Communicator,
        communication::exclusive_scan_tag>
    {
        template <typename Result, typename T, typename F>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t, F&& op)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                communication::communicator_data<
                    communication::exclusive_scan_tag>::name(),
                which, generation,
                // step function (invoked for each get)
                [&t](auto& data, std::size_t which) {
                    data[which] = HPX_FORWARD(T, t);
                },
                // finalizer (invoked non-concurrently after all data has been
                // received)
                [op = HPX_FORWARD(F, op)](auto& data, bool& data_available,
                    std::size_t which) mutable {
                    if (!data_available)
                    {
                        using T_ = std::decay_t<T>;

                        std::vector<T_> dest;
                        dest.resize(data.size());

                        if constexpr (!std::is_same_v<T_, bool>)
                        {
                            collectives::detail::exclusive_scan<T_>(
                                data.begin(), data.end(), dest.begin(),
                                HPX_FORWARD(F, op));
                        }
                        else
                        {
                            collectives::detail::exclusive_scan<bool>(
                                data.begin(), data.end(), dest.begin(),
                                [&](auto lhs, auto rhs) {
                                    return HPX_FORWARD(F, op)(
                                        static_cast<bool>(lhs),
                                        static_cast<bool>(rhs));
                                });
                        }
                        std::swap(data, dest);
                        data_available = true;
                    }
                    return Communicator::template handle_bool<std::decay_t<T>>(
                        HPX_MOVE(data[which]));
                });
        }
    };

    template <typename Communicator>
    struct communication_operation<Communicator,
        communication::exclusive_scan_init_tag>
    {
        template <typename Result, typename T, typename Init, typename F>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t, Init&& init, F&& op)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                communication::communicator_data<
                    communication::exclusive_scan_init_tag>::name(),
                which, generation,
                // step function (invoked for each get)
                [&t](auto& data, std::size_t which) {
                    data[which] = HPX_FORWARD(T, t);
                },
                // finalizer (invoked non-concurrently after all data has been
                // received)
                [op = HPX_FORWARD(F, op), init = HPX_FORWARD(Init, init)](
                    auto& data, bool& data_available,
                    std::size_t which) mutable {
                    if (!data_available)
                    {
                        using T_ = std::decay_t<T>;

                        std::vector<T_> dest;
                        dest.resize(data.size());

                        if constexpr (!std::is_same_v<T_, bool>)
                        {
                            hpx::exclusive_scan(data.begin(), data.end(),
                                dest.begin(), HPX_FORWARD(Init, init),
                                HPX_FORWARD(F, op));
                        }
                        else
                        {
                            hpx::exclusive_scan(data.begin(), data.end(),
                                dest.begin(), static_cast<bool>(init),
                                [&](auto lhs, auto rhs) {
                                    return HPX_FORWARD(F, op)(
                                        static_cast<bool>(lhs),
                                        static_cast<bool>(rhs));
                                });
                        }
                        std::swap(data, dest);
                        data_available = true;
                    }
                    return Communicator::template handle_bool<std::decay_t<T>>(
                        HPX_MOVE(data[which]));
                });
        }
    };
}    // namespace hpx::traits

namespace hpx::collectives {

    ////////////////////////////////////////////////////////////////////////////
    // exclusive_scan with semantics similar to MPI_Exscan
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(communicator fid,
        T&& local_result, F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site.is_default())
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::exclusive_scan",
                "the generation number shouldn't be zero"));
        }

        // Handle operation right away if there is only one value.
        if (auto [num_sites, comm_site] = fid.get_info(); num_sites == 1)
        {
            if (this_site != comm_site)
            {
                return hpx::make_exceptional_future<arg_type>(
                    HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                        "hpx::collectives::exclusive_scan",
                        "the local site should be zero if only one site is "
                        "involved"));
            }
            return hpx::make_ready_future(arg_type{});
        }

        auto exclusive_scan_data =
            [local_result = HPX_FORWARD(T, local_result),
                op = HPX_FORWARD(F, op), this_site,
                generation](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type =
                detail::communicator_server::communication_get_direct_action<
                    traits::communication::exclusive_scan_tag,
                    hpx::future<arg_type>, arg_type, func_type>;

            // explicitly unwrap returned future
            hpx::future<arg_type> result = hpx::async(action_type(), c,
                this_site, generation, HPX_MOVE(local_result), HPX_MOVE(op));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)] { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(exclusive_scan_data));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(communicator fid,
        T&& local_result, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation);
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> exclusive_scan(char const* basename,
        T&& local_result, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return exclusive_scan(create_communicator(basename, num_sites,
                                  this_site, generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site);
    }

    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator fid,
        T&& local_result, F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator fid,
        T&& local_result, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename F>
    decltype(auto) exclusive_scan(hpx::launch::sync_policy,
        char const* basename, T&& local_result, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return exclusive_scan(create_communicator(basename, num_sites,
                                  this_site, generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site)
            .get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Version of exclusive scan that takes an initial value for element 0.
    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<this_site_arg, std::decay_t<F>>)
    hpx::future<std::decay_t<T>> exclusive_scan(communicator fid,
        T&& local_result, Init&& init, F&& op,
        this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site.is_default())
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::exclusive_scan",
                "the generation number shouldn't be zero"));
        }

        // Handle operation right away if there is only one value.
        if (auto [num_sites, comm_site] = fid.get_info(); num_sites == 1)
        {
            if (this_site != comm_site)
            {
                return hpx::make_exceptional_future<arg_type>(
                    HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                        "hpx::collectives::exclusive_scan",
                        "the local site should be zero if only one site is "
                        "involved"));
            }
            return hpx::make_ready_future<arg_type>(HPX_FORWARD(Init, init));
        }

        auto exclusive_scan_data =
            [local_result = HPX_FORWARD(T, local_result),
                init = HPX_FORWARD(Init, init), op = HPX_FORWARD(F, op),
                this_site,
                generation](communicator&& c) mutable -> hpx::future<arg_type> {
            using init_type = std::decay_t<Init>;
            using func_type = std::decay_t<F>;
            using action_type =
                detail::communicator_server::communication_get_direct_action<
                    traits::communication::exclusive_scan_init_tag,
                    hpx::future<arg_type>, arg_type, init_type, func_type>;

            // explicitly unwrap returned future
            hpx::future<arg_type> result =
                hpx::async(action_type(), c, this_site, generation,
                    HPX_MOVE(local_result), HPX_MOVE(init), HPX_MOVE(op));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)] { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(exclusive_scan_data));
    }

    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<generation_arg, std::decay_t<F>>)
    hpx::future<std::decay_t<T>> exclusive_scan(communicator fid,
        T&& local_result, Init&& init, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(Init, init), HPX_FORWARD(F, op), this_site, generation);
    }

    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<num_sites_arg, std::decay_t<F>>)
    hpx::future<std::decay_t<T>> exclusive_scan(char const* basename,
        T&& local_result, Init&& init, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return exclusive_scan(create_communicator(basename, num_sites,
                                  this_site, generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(Init, init),
            HPX_FORWARD(F, op), this_site);
    }

    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<this_site_arg, std::decay_t<F>>)
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator fid,
        T&& local_result, Init&& init, F&& op,
        this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(Init, init), HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<generation_arg, std::decay_t<F>>)
    decltype(auto) exclusive_scan(hpx::launch::sync_policy, communicator fid,
        T&& local_result, Init&& init, F&& op, generation_arg const generation,
        this_site_arg const this_site = this_site_arg())
    {
        return exclusive_scan(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(Init, init), HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename Init, typename F>
        requires(!std::is_same_v<num_sites_arg, std::decay_t<F>>)
    decltype(auto) exclusive_scan(hpx::launch::sync_policy,
        char const* basename, T&& local_result, Init&& init, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return exclusive_scan(create_communicator(basename, num_sites,
                                  this_site, generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(Init, init),
            HPX_FORWARD(F, op), this_site)
            .get();
    }
}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
