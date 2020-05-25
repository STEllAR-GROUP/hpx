//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file gather.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace lcos {

    /// Gather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central gather point from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       Each gather operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_GATHER macro to define the
    ///             necessary internal facilities used by \a gather_here and
    ///             \a gather_there
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<T>> gather_here(char const* basename,
        hpx::future<T> result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1));

    /// Gather a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central gather
    /// site (where the corresponding \a gather_here is executed)
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central gather point from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The sequence number of the central gather point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    ///
    /// \note       Each gather operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_GATHER macro to define the
    ///             necessary internal facilities used by \a gather_here and
    ///             \a gather_there
    ///
    /// \returns    This function returns a future which will become ready once
    ///             the gather operation has been completed.
    ///
    template <typename T>
    hpx::future<void> gather_there(char const* basename, hpx::future<T> result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

    /// Gather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       Each gather operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_GATHER macro to define the
    ///             necessary internal facilities used by \a gather_here and
    ///             \a gather_there
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<typename std::decay<T>::type>> gather_here(
        char const* basename, T&& result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1));

    /// Gather a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central gather
    /// site (where the corresponding \a gather_here is executed)
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The sequence number of the central gather point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    ///
    /// \note       Each gather operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_GATHER macro to define the
    ///             necessary internal facilities used by \a gather_here and
    ///             \a gather_there
    ///
    /// \returns    This function returns a future which will become ready once
    ///             the gather operation has been completed.
    ///
    template <typename T>
    hpx::future<void> gather_there(char const* basename, T&& result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

/// \def HPX_REGISTER_GATHER_DECLARATION(type, name)
///
/// \brief Declare a gather object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_GATHER_DECLARATION can be used to declare
/// all facilities necessary for a (possibly remote) gather operation.
///
/// The parameter \a type specifies for which data type the gather
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// which will be internally used to identify a particular gather operation.
/// If this defaults to \a \<type\>_gather if not specified.
///
/// \note The macro \a HPX_REGISTER_GATHER_DECLARATION can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_gather.
///
#define HPX_REGISTER_GATHER_DECLARATION(type, name)

/// \def HPX_REGISTER_GATHER(type, name)
///
/// \brief Define a gather object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_GATHER can be used to define
/// all facilities necessary for a (possibly remote) gather operation.
///
/// The parameter \a type specifies for which data type the gather
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// which will be internally used to identify a particular gather operation.
/// If this defaults to \a \<type\>_gather if not specified.
///
/// \note The macro \a HPX_REGISTER_GATHER can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_gather.
///
#define HPX_REGISTER_GATHER(type, name)
}}    // namespace hpx::lcos
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async/dataflow.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/naming/id_type.hpp>
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

    ///////////////////////////////////////////////////////////////////////////
    // support for gather
    namespace communication {
        struct gather_tag;
    }

    template <typename Communicator>
    struct communication_operation<Communicator, communication::gather_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::gather_tag>>
    {
        communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T>
        Result get(std::size_t which, T&& t)
        {
            using arg_type = typename Communicator::arg_type;
            using mutex_type = typename Communicator::mutex_type;

            auto this_ = this->shared_from_this();
            auto on_ready =
                [this_ = std::move(this_)](
                    shared_future<void>&& f) -> std::vector<arg_type> {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                std::vector<T> data(communicator.num_sites_);
                {
                    std::unique_lock<mutex_type> l(communicator.mtx_);
                    std::swap(data, communicator.data_);
                }
                return data;
            };

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            hpx::future<std::vector<arg_type>> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

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

        template <typename T>
        void set(std::size_t which, T&& t)
        {
            using mutex_type = typename Communicator::mutex_type;

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

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
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace lcos {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_gatherer(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return detail::create_communicator<T>(
            basename, num_sites, generation, this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<std::vector<T>> gather_here(hpx::future<hpx::id_type>&& fid,
        hpx::future<T>&& result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        auto gather_data =
            [this_site](hpx::future<hpx::id_type>&& fid,
                hpx::future<T>&& local_result) -> hpx::future<std::vector<T>> {
            using action_type = typename detail::communicator_server<T>::
                template communication_get_action<
                    traits::communication::gather_tag,
                    hpx::future<std::vector<T>>, T>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(gather_data),
            std::move(fid), std::move(result));
    }

    template <typename T>
    hpx::future<std::vector<T>> gather_here(char const* basename,
        hpx::future<T> result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        return gather_here(
            create_gatherer<T>(basename, num_sites, generation, this_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // gather plain values
    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type>> gather_here(
        hpx::future<hpx::id_type>&& fid, T&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }
        typedef typename std::decay<T>::type arg_type;

        auto gather_data_direct =
            [this_site](hpx::future<hpx::id_type>&& fid,
                T&& local_result) -> hpx::future<std::vector<arg_type>> {
            using action_type = typename detail::communicator_server<arg_type>::
                template communication_get_action<
                    traits::communication::gather_tag,
                    hpx::future<std::vector<arg_type>>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result = async(
                action_type(), id, this_site, std::forward<T>(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(gather_data_direct),
            std::move(fid), std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type>> gather_here(
        char const* basename, T&& result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
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

        return gather_here(
            create_gatherer<T>(basename, num_sites, generation, this_site),
            std::forward<T>(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<void> gather_there(hpx::future<hpx::id_type>&& fid,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto gather_there_data =
            [this_site](hpx::future<hpx::id_type>&& fid,
                hpx::future<T>&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server<T>::
                template communication_set_action<
                    traits::communication::gather_tag, T>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(gather_there_data),
            std::move(fid), std::move(local_result));
    }

    template <typename T>
    hpx::future<void> gather_there(char const* basename,
        hpx::future<T>&& result, std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return gather_there(hpx::find_from_basename(std::move(name), root_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // gather plain values
    template <typename T>
    hpx::future<void> gather_there(hpx::future<hpx::id_type>&& fid,
        T&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        using arg_type = typename util::decay<T>::type;

        auto gather_there_data_direct =
            [this_site](hpx::future<hpx::id_type>&& fid,
                arg_type&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server<T>::
                template communication_get_action<
                    traits::communication::gather_tag, hpx::future<T>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result = async(
                action_type(), id, this_site, std::forward<T>(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(std::move(gather_there_data_direct), std::move(fid),
            std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<void> gather_there(char const* basename, T&& local_result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return gather_there(hpx::find_from_basename(std::move(name), root_site),
            std::forward<T>(local_result), this_site);
    }
}}    // namespace hpx::lcos

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_GATHER_DECLARATION(...) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_GATHER(...)                                               \
    HPX_REGISTER_GATHER_(__VA_ARGS__)                                          \
    /**/

#define HPX_REGISTER_GATHER_(...)                                              \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_GATHER_, HPX_PP_NARGS(__VA_ARGS__))( \
        __VA_ARGS__))                                                          \
    /**/

#define HPX_REGISTER_GATHER_1(type)                                            \
    HPX_REGISTER_GATHER_2(type, HPX_PP_CAT(type, _gather))                     \
    /**/

#define HPX_REGISTER_GATHER_2(type, name)                                      \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::communicator_server<type>>                          \
        HPX_PP_CAT(gather_, name);                                             \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(gather_, name))                          \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
