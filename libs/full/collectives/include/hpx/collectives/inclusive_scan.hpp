//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file inclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Inclusive inclusive_scan a set of values from different call sites
    ///
    /// This function performs an inclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  basename    The base name identifying the inclusive_scan operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the inclusive_scan operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the inclusive_scan operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     inclusive_scan support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the inclusive_scan operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> inclusive_scan(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Inclusive inclusive_scan a set of values from different call sites
    ///
    /// This function performs an inclusive scan operation on a set of values
    /// received from all call sites operating on the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the inclusive_scan operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> inclusive_scan(communicator comm,
        T&& result, F&& op, this_site_arg this_site = this_site_arg());
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
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    namespace communication {
        struct inclusive_scan_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for inclusive_scan
    template <typename Communicator>
    struct communication_operation<Communicator,
        communication::inclusive_scan_tag>
      : std::enable_shared_from_this<communication_operation<Communicator,
            communication::inclusive_scan_tag>>
    {
        explicit communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T, typename F>
        Result get(std::size_t which, T&& t, F&& op)
        {
            using arg_type = std::decay_t<T>;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready =
                [which, this_ = std::move(this_), op = std::forward<F>(op)](
                    hpx::shared_future<void> f) mutable -> arg_type {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);
                util::ignore_while_checking<lock_type> il(&l);

                auto& data = communicator.template access_data<arg_type>(l);
                if (!communicator.data_available_)
                {
                    std::vector<arg_type> dest;
                    dest.resize(data.size());

                    hpx::inclusive_scan(hpx::execution::seq, data.begin(),
                        data.end(), dest.begin(), std::forward<F>(op));

                    std::swap(data, dest);
                    communicator.data_available_ = true;
                }
                return data[which];
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<arg_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<arg_type>(l);
            data[which] = std::forward<T>(t);

            if (communicator_.gate_.set(which, std::move(l)))
            {
                l = lock_type(communicator_.mtx_);
                communicator_.invalidate_data(l);
            }

            return f;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace collectives {

    ////////////////////////////////////////////////////////////////////////////
    // inclusive_scan plain values
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> inclusive_scan(communicator fid,
        T&& local_result, F&& op, this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto scan_data_direct =
            [op = std::forward<F>(op),
                local_result = std::forward<T>(local_result),
                this_site](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::inclusive_scan_tag,
                    hpx::future<arg_type>, arg_type, func_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<arg_type> result = async(action_type(), c, this_site,
                std::forward<T>(local_result), std::forward<F>(op));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(scan_data_direct));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> inclusive_scan(char const* basename,
        T&& local_result, F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        return inclusive_scan(create_communicator(basename, num_sites,
                                  this_site, generation, root_site),
            std::forward<T>(local_result), std::forward<F>(op), this_site);
    }
}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
