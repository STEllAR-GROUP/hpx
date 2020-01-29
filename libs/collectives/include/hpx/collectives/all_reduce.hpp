//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_reduce.hpp

#if !defined(HPX_COLLECTIVES_ALL_REDUCE_JUL_01_2019_0313PM)
#define HPX_COLLECTIVES_ALL_REDUCE_JUL_01_2019_0313PM

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace lcos {

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values that are the result of applying
    /// a given operator on values supplied from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  local_result A future referring to the value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_reduce support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each all_reduce operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLREDUCE macro to define the
    ///             necessary internal facilities used by \a all_reduce.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<T> all_reduce(char const* basename, hpx::future<T> result,
        F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_reduce support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each all_reduce operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLREDUCE macro to define the
    ///             necessary internal facilities used by \a all_reduce.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(char const* basename, T&& result,
        F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

/// \def HPX_REGISTER_ALLREDUCE_DECLARATION(type, name)
///
/// \brief Declare a all_reduce object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_ALLREDUCE_DECLARATION can be used to declare
/// all facilities necessary for a (possibly remote) all_reduce operation.
///
/// The parameter \a type specifies for which data type the all_reduce
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// that will be internally used to identify a particular all_reduce operation.
/// If this defaults to \a \<type\>_all_reduce if not specified.
///
/// \note The macro \a HPX_REGISTER_ALLREDUCE_DECLARATION can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_all_reduce.
///
#define HPX_REGISTER_ALLREDUCE_DECLARATION(type, name)

    /// \def HPX_REGISTER_ALLREDUCE(type, name)
    ///
    /// \brief Define a all_reduce object named \a name for a given data type \a type.
    ///
    /// The macro \a HPX_REGISTER_ALLREDUCE can be used to define
    /// all facilities necessary for a (possibly remote) all_reduce operation.
    ///
    /// The parameter \a type specifies for which data type the all_reduce
    /// operations should be enabled.
    ///
    /// The (optional) parameter \a name should be a unique C-style identifier
    /// that will be internally used to identify a particular all_reduce operation.
    /// If this defaults to \a \<type\>_all_reduce if not specified.
    ///
    /// \note The macro \a HPX_REGISTER_ALLREDUCE can be used with 1
    ///       or 2 arguments. The second argument is optional and defaults to
    ///       \a \<type\>_all_reduce.
    ///
    #define HPX_REGISTER_ALLREDUCE(type, name)
}}    // namespace hpx::lcos
// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assertion.hpp>
#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/local_lcos/and_gate.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
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

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        class all_reduce_server
          : public hpx::components::component_base<all_reduce_server<T>>
        {
            using mutex_type = lcos::local::spinlock;

        public:
            all_reduce_server()    //-V730
            {
                HPX_ASSERT(false);    // shouldn't ever be called
            }

            all_reduce_server(std::size_t num_sites, std::string const& name,
                std::size_t site)
              : num_sites_(num_sites)
              , data_(num_sites)
              , gate_(num_sites)
              , name_(name)
              , site_(site)
            {
            }

            template <typename F>
            hpx::future<T> get_result(std::size_t which, T t, F op)
            {
                std::unique_lock<mutex_type> l(mtx_);

                auto on_ready = [this, op = std::move(op)](
                                    hpx::shared_future<void> f) mutable -> T {
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

                    // this is a one-shot object (generations counters are
                    // not supported), unregister ourselves, but only once)
                    if (!name.empty())
                    {
                        hpx::unregister_with_basename(name, site_).get();
                    }

                    HPX_ASSERT(!data.empty());

                    auto it = data.begin();
                    return hpx::parallel::reduce(hpx::parallel::execution::par,
                        ++it, data.end(), *data.begin(), op);
                };

                hpx::future<T> f = gate_.get_shared_future(l).then(
                    hpx::launch::async, std::move(on_ready));

                gate_.synchronize(1, l);
                data_[which] = std::move(t);
                gate_.set(which, l);

                return f;
            }

            template <typename F>
            struct get_result_action
              : hpx::actions::make_action<hpx::future<T> (all_reduce_server::*)(
                                              std::size_t, T, F),
                    &all_reduce_server::template get_result<F>,
                    get_result_action<F>>::type
            {
            };

        private:
            mutex_type mtx_;
            std::size_t num_sites_;
            std::vector<T> data_;
            lcos::local::and_gate gate_;
            std::string name_;
            std::size_t site_;
        };

        ////////////////////////////////////////////////////////////////////////
        inline hpx::future<hpx::id_type> register_all_reduce_name(
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
                            "hpx::lcos::detail::register_all_reduce_name",
                            "the given base name for the all_reduce operation "
                            "was already registered: " +
                                basename);
                    }
                    return target;
                });
        }
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_all_reduce(char const* basename,
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

        // create a new all_reduce_server
        using result_type = typename util::decay<T>::type;
        hpx::future<hpx::id_type> id =
            hpx::new_<detail::all_reduce_server<result_type>>(
                hpx::find_here(), num_sites, name, this_site);

        // register the all_reduce's id using the given basename
        return id.then(hpx::launch::sync,
            util::bind_back(
                &detail::register_all_reduce_name, std::move(name), this_site));
    }

    ////////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T, typename F>
    hpx::future<T> all_reduce(hpx::future<hpx::id_type>&& f,
        hpx::future<T>&& local_result, F&& op,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        auto all_reduce_data =
            [op = std::forward<F>(op), this_site](hpx::future<hpx::id_type>&& f,
                hpx::future<T>&& local_result) mutable -> hpx::future<T> {
            using func_type = typename std::decay<F>::type;
            using action_type = typename detail::all_reduce_server<
                T>::template get_result_action<func_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            hpx::future<T> result = async(action_type(), id, this_site,
                local_result.get(), std::forward<F>(op));

            return result.then(hpx::launch::sync,
                [id = std::move(id)](hpx::future<T>&& f) -> T {
                    HPX_UNUSED(id);
                    return f.get();
                });
        };

        return dataflow(hpx::launch::sync, std::move(all_reduce_data),
            std::move(f), std::move(local_result));
    }

    template <typename T, typename F>
    hpx::future<T> all_reduce(char const* basename,
        hpx::future<T>&& local_result, F&& op,
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

        if (this_site == 0)
        {
            return all_reduce(create_all_reduce<T>(
                                  basename, num_sites, generation, root_site),
                std::move(local_result), std::forward<F>(op), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return all_reduce(hpx::find_from_basename(std::move(name), root_site),
            std::move(local_result), std::forward<F>(op), this_site);
    }

    ////////////////////////////////////////////////////////////////////////////
    // all_reduce plain values
    template <typename T, typename F>
    hpx::future<typename std::decay<T>::type> all_reduce(
        hpx::future<hpx::id_type>&& f, T&& local_result, F&& op,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using arg_type = typename std::decay<T>::type;
        auto all_reduce_data_direct =
            [op = std::forward<F>(op),
                local_result = std::forward<T>(local_result),
                this_site](hpx::future<hpx::id_type>&& f) mutable
            -> hpx::future<arg_type> {
            using func_type = typename std::decay<F>::type;
            using action_type = typename detail::all_reduce_server<
                arg_type>::template get_result_action<func_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            hpx::future<arg_type> result = async(action_type(), id, this_site,
                std::forward<T>(local_result), std::forward<F>(op));

            return result.then(hpx::launch::sync,
                [id = std::move(id)](hpx::future<arg_type>&& f) -> arg_type {
                    HPX_UNUSED(id);
                    return f.get();
                });
        };

        using result_type = typename util::decay<T>::type;
        return dataflow(
            hpx::launch::sync, all_reduce_data_direct, std::move(f));
    }

    template <typename T, typename F>
    hpx::future<typename std::decay<T>::type> all_reduce(char const* basename,
        T&& local_result, F&& op, std::size_t num_sites = std::size_t(-1),
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
            return all_reduce(create_all_reduce<T>(
                                  basename, num_sites, generation, root_site),
                std::forward<T>(local_result), std::forward<F>(op), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return all_reduce(hpx::find_from_basename(std::move(name), root_site),
            std::forward<T>(local_result), std::forward<F>(op), this_site);
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
namespace hpx {
    using lcos::all_reduce;
    using lcos::create_all_reduce;
}    // namespace hpx

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE(...)                                            \
    HPX_REGISTER_ALLREDUCE_(__VA_ARGS__)                                       \
    /**/

#define HPX_REGISTER_ALLREDUCE_(...)                                           \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_REGISTER_ALLREDUCE_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))      \
    /**/

#define HPX_REGISTER_ALLREDUCE_1(type)                                         \
    HPX_REGISTER_ALLREDUCE_2(type, HPX_PP_CAT(type, _all_reduce))              \
    /**/

#define HPX_REGISTER_ALLREDUCE_2(type, name)                                   \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::all_reduce_server<type>>                            \
        HPX_PP_CAT(all_reduce_, name);                                         \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(all_reduce_, name))                      \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
#endif
