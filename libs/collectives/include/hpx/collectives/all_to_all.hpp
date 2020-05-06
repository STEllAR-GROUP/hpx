//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_to_all.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace lcos {

    /// AllToAll a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_to_all operation
    /// \param  local_result A future referring to the value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_to_all operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_to_all operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_to_all support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each all_to_all operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLTOALL macro to define the
    ///             necessary internal facilities used by \a all_to_all.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_to_all operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<T>> all_to_all(char const* basename,
        hpx::future<T> result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

    /// AllToAll a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_to_all operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_to_all operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_to_all operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_to_all support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each all_to_all operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLTOALL macro to define the
    ///             necessary internal facilities used by \a all_to_all.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_to_all operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<typename std::decay<T>::type>> all_to_all(
        char const* basename, T&& result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

/// \def HPX_REGISTER_ALLTOALL_DECLARATION(type, name)
///
/// \brief Declare a all_to_all object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_ALLTOALL_DECLARATION can be used to declare
/// all facilities necessary for a (possibly remote) all_to_all operation.
///
/// The parameter \a type specifies for which data type the all_to_all
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// that will be internally used to identify a particular all_to_all operation.
/// If this defaults to \a \<type\>_all_to_all if not specified.
///
/// \note The macro \a HPX_REGISTER_ALLTOALL_DECLARATION can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_all_to_all.
///
#define HPX_REGISTER_ALLTOALL_DECLARATION(type, name)

/// \def HPX_REGISTER_ALLTOALL(type, name)
///
/// \brief Define a all_to_all object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_ALLTOALL can be used to define
/// all facilities necessary for a (possibly remote) all_to_all operation.
///
/// The parameter \a type specifies for which data type the all_to_all
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// that will be internally used to identify a particular all_to_all operation.
/// If this defaults to \a \<type\>_all_to_all if not specified.
///
/// \note The macro \a HPX_REGISTER_ALLTOALL can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_all_to_all.
///
#define HPX_REGISTER_ALLTOALL(type, name)
}}    // namespace hpx::lcos
// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async/dataflow.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    namespace communication {
        struct all_to_all_tag;
    }

    ///////////////////////////////////////////////////////////////////////////
    // support for all_to_all
    template <typename Communicator>
    struct communication_operation<Communicator, communication::all_to_all_tag>
      : std::enable_shared_from_this<communication_operation<Communicator,
            communication::all_to_all_tag>>
    {
        communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T>
        Result get(std::size_t which, T&& t)
        {
            using arg_type = typename std::decay<T>::type;
            using mutex_type = typename Communicator::mutex_type;

            auto this_ = this->shared_from_this();
            auto on_ready =
                [this_ = std::move(this_)](
                    shared_future<void>&& f) -> std::vector<arg_type> {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                std::vector<arg_type> data;
                {
                    std::unique_lock<mutex_type> l(communicator.mtx_);
                    data = communicator.data_;
                }
                return data;
            };

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            hpx::future<std::vector<arg_type>> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, on_ready);

            communicator_.gate_.synchronize(1, l);
            communicator_.data_[which] = std::forward<T>(t);
            if (communicator_.gate_.set(which, l))
            {
                // this is a one-shot object (generations counters are not
                // supported), unregister ourselves (but only once)
                hpx::unregister_with_basename(
                    std::move(communicator_.name_), communicator_.site_)
                    .get();
            }
            return f;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace lcos {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_all_to_all(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return detail::create_communicator<T>(
            basename, num_sites, generation, this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<std::vector<T>> all_to_all(hpx::future<hpx::id_type>&& fid,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        auto all_to_all_data =
            [this_site](hpx::future<hpx::id_type>&& f,
                hpx::future<T>&& local_result) -> hpx::future<std::vector<T>> {
            using action_type = typename detail::communicator_server<T>::
                template communication_get_action<
                    traits::communication::all_to_all_tag,
                    hpx::future<std::vector<T>>, T>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(all_to_all_data),
            std::move(fid), std::move(local_result));
    }

    template <typename T>
    hpx::future<std::vector<T>> all_to_all(char const* basename,
        hpx::future<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        if (this_site == root_site)
        {
            return all_to_all(create_all_to_all<T>(
                                  basename, num_sites, generation, root_site),
                std::move(local_result), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return all_to_all(hpx::find_from_basename(std::move(name), root_site),
            std::move(local_result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // all_to_all plain values
    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type>> all_to_all(
        hpx::future<hpx::id_type>&& fid, T&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        using arg_type = typename util::decay<T>::type;

        auto all_to_all_data_direct =
            [local_result = std::forward<T>(local_result), this_site](
                hpx::future<hpx::id_type>&& f)
            -> hpx::future<std::vector<arg_type>> {
            using action_type = typename detail::communicator_server<arg_type>::
                template communication_get_action<
                    traits::communication::all_to_all_tag,
                    hpx::future<std::vector<arg_type>>, arg_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(all_to_all_data_direct));
    }

    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type>> all_to_all(
        char const* basename, T&& local_result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        if (this_site == root_site)
        {
            return all_to_all(create_all_to_all<T>(
                                  basename, num_sites, generation, root_site),
                std::forward<T>(local_result), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return all_to_all(hpx::find_from_basename(std::move(name), root_site),
            std::forward<T>(local_result), this_site);
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
namespace hpx {
    using lcos::all_to_all;
    using lcos::create_all_to_all;
}    // namespace hpx

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLTOALL_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLTOALL(...)                                             \
    HPX_REGISTER_ALLTOALL_(__VA_ARGS__)                                        \
    /**/

#define HPX_REGISTER_ALLTOALL_(...)                                            \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_REGISTER_ALLTOALL_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))       \
    /**/

#define HPX_REGISTER_ALLTOALL_1(type)                                          \
    HPX_REGISTER_ALLTOALL_2(type, HPX_PP_CAT(type, _all_to_all))               \
    /**/

#define HPX_REGISTER_ALLTOALL_2(type, name)                                    \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::communicator_server<type>>                          \
        HPX_PP_CAT(all_to_all_, name);                                         \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(all_to_all_, name))                      \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
