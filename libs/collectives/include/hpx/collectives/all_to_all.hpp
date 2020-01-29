//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_to_all.hpp

#if !defined(HPX_COLLECTIVES_ALL_TO_ALL_JUN_23_2019_0900AM)
#define HPX_COLLECTIVES_ALL_TO_ALL_JUN_23_2019_0900AM

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

#include <hpx/assertion.hpp>
#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/local_lcos/and_gate.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/unmanaged.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace lcos {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class all_to_all_server
          : public hpx::components::component_base<all_to_all_server<T>>
        {
            using mutex_type = lcos::local::spinlock;

        public:
            all_to_all_server()    //-V730
            {
                HPX_ASSERT(false);    // shouldn't ever be called
            }

            all_to_all_server(std::size_t num_sites, std::string const& name,
                std::size_t site)
              : num_sites_(num_sites)
              , data_(num_sites)
              , gate_(num_sites)
              , name_(name)
              , site_(site)
            {
            }

            hpx::future<std::vector<T>> get_result(std::size_t which, T t)
            {
                std::unique_lock<mutex_type> l(mtx_);

                auto on_ready = [this](
                                    shared_future<void>&& f) -> std::vector<T> {
                    f.get();    // propagate any exceptions

                    std::vector<T> data;
                    std::string name;

                    {
                        std::unique_lock<mutex_type> l(mtx_);
                        util::ignore_while_checking<
                            std::unique_lock<mutex_type>>
                            il(&l);
                        data = data_;
                        std::swap(name, name_);
                    }

                    // this is a one-shot object (generations counters are not
                    // supported), unregister ourselves (but only once)
                    if (!name.empty())
                    {
                        hpx::unregister_with_basename(name, site_).get();
                    }

                    return data;
                };

                hpx::future<std::vector<T>> f = gate_.get_shared_future(l).then(
                    hpx::launch::async, on_ready);

                gate_.synchronize(1, l);
                data_[which] = std::move(t);
                gate_.set(which, l);

                return f;
            }

            HPX_DEFINE_COMPONENT_ACTION(
                all_to_all_server, get_result, get_result_action);

        private:
            mutex_type mtx_;
            std::size_t num_sites_;
            std::vector<T> data_;
            lcos::local::and_gate gate_;
            std::string name_;
            std::size_t site_;
        };

        ///////////////////////////////////////////////////////////////////////
        inline hpx::future<hpx::id_type> register_all_to_all_name(
            hpx::future<hpx::id_type>&& f, std::string basename,
            std::size_t site)
        {
            hpx::id_type target = f.get();

            // Register unmanaged id to avoid cyclic dependencies, unregister
            // is done after all data has been collected in the component above.
            hpx::future<bool> result = hpx::register_with_basename(
                basename, hpx::unmanaged(target), site);

            return result.then(hpx::launch::sync,
                [target = std::move(target), basename = std::move(basename)](
                    hpx::future<bool>&& f) -> hpx::id_type {
                    bool result = f.get();
                    if (!result)
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "hpx::lcos::detail::register_all_to_all_name",
                            "the given base name for the all_to_all operation "
                            "was already registered: " +
                                basename);
                    }
                    return target;
                });
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_all_to_all(char const* basename,
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
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        // create a new all_to_all_server
        using result_type = typename util::decay<T>::type;
        hpx::future<hpx::id_type> id =
            hpx::new_<detail::all_to_all_server<result_type>>(
                hpx::find_here(), num_sites, name, this_site);

        // register the all_to_all's id using the given basename
        return id.then(hpx::launch::sync,
            util::bind_back(
                &detail::register_all_to_all_name, std::move(name), this_site));
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<std::vector<T>> all_to_all(hpx::future<hpx::id_type>&& f,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        auto all_to_all_data =
            [this_site](hpx::future<hpx::id_type>&& f,
                hpx::future<T>&& local_result) -> hpx::future<std::vector<T>> {
            using action_type =
                typename detail::all_to_all_server<T>::get_result_action;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            return result.then(hpx::launch::sync,
                [id = std::move(id)](
                    hpx::future<std::vector<T>>&& f) -> std::vector<T> {
                    HPX_UNUSED(id);
                    return f.get();
                });
        };

        return dataflow(hpx::launch::sync, std::move(all_to_all_data),
            std::move(f), std::move(local_result));
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
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        if (this_site == 0)
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
        hpx::future<hpx::id_type>&& f, T&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using arg_type = typename util::decay<T>::type;

        auto all_to_all_data_direct =
            [local_result = std::forward<T>(local_result), this_site](
                hpx::future<hpx::id_type>&& f)
            -> hpx::future<std::vector<arg_type>> {
            using action_type =
                typename detail::all_to_all_server<arg_type>::get_result_action;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, std::move(local_result));

            return result.then(hpx::launch::sync,
                [id = std::move(id)](hpx::future<std::vector<arg_type>>&& f)
                    -> std::vector<arg_type> {
                    HPX_UNUSED(id);
                    return f.get();
                });
        };

        return dataflow(
            hpx::launch::sync, std::move(all_to_all_data_direct), std::move(f));
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
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

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
#define HPX_REGISTER_ALLTOALL_DECLARATION(...)                                 \
    HPX_REGISTER_ALLTOALL_DECLARATION_(__VA_ARGS__)                            \
    /**/

#define HPX_REGISTER_ALLTOALL_DECLARATION_(...)                                \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_ALLTOALL_DECLARATION_,               \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_REGISTER_ALLTOALL_DECLARATION_1(type)                              \
    HPX_REGISTER_ALLTOALL_DECLARATION_2(type, HPX_PP_CAT(type, _all_to_all))   \
    /**/

#define HPX_REGISTER_ALLTOALL_DECLARATION_2(type, name)                        \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::detail::all_to_all_server<type>::get_result_action,         \
        HPX_PP_CAT(gather_get_result_action_, name));                          \
    /**/

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
    HPX_REGISTER_ACTION(                                                       \
        hpx::lcos::detail::all_to_all_server<type>::get_result_action,         \
        HPX_PP_CAT(gather_get_result_action_, name));                          \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::all_to_all_server<type>>                            \
        HPX_PP_CAT(all_to_all_, name);                                         \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(all_to_all_, name))                      \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
#endif
