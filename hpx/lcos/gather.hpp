//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file gather.hpp

#if !defined(HPX_LCOS_GATHER_MAY_05_2014_0418PM)
#define HPX_LCOS_GATHER_MAY_05_2014_0418PM

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
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
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(char const* basename, hpx::future<T> result,
        std::size_t num_sites = std::size_t(-1),
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
    /// \param root_site    The sequence number of the central gather point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future which will become ready once
    ///             the gather operation has been completed.
    ///
    template <typename T>
    hpx::future<void>
    gather_there(char const* basename, hpx::future<T> result,
        std::size_t generation = std::size_t(-1), std::size_t root_site = 0,
        std::size_t this_site = std::size_t(-1));

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
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<typename std::decay<T>::type> >
    gather_here(char const* basename, T && result,
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
    /// \param root_site    The sequence number of the central gather point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future which will become ready once
    ///             the gather operation has been completed.
    ///
    template <typename T>
    hpx::future<void>
    gather_there(char const* basename, T && result,
        std::size_t generation = std::size_t(-1), std::size_t root_site = 0,
        std::size_t this_site = std::size_t(-1));
}}
#else

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/and_gate.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/unmanaged.hpp>
#include <hpx/util/decay.hpp>

#include <boost/preprocessor/cat.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class gather_server
          : public hpx::components::simple_component_base<gather_server<T> >
        {
            typedef lcos::local::spinlock mutex_type;

            std::vector<T> on_ready(future<void> f)
            {
                f.get();       // propagate any exceptions

                std::vector<T> data(num_sites_);

                {
                    std::unique_lock<mutex_type> l(mtx_);
                    std::swap(data, data_);
                }

                return data;
            }

        public:
            gather_server() //-V730
            {
                HPX_ASSERT(false);  // shouldn't ever be called
            }

            gather_server(std::size_t num_sites)
              : num_sites_(num_sites), data_(num_sites), gate_(num_sites)
            {}

            hpx::future<std::vector<T> > get_result(std::size_t which, T && t)
            {
                std::unique_lock<mutex_type> l(mtx_);

                using util::placeholders::_1;
                hpx::future<std::vector<T> > f = gate_.get_future().then(
                        util::bind(&gather_server::on_ready, this, _1));

                set_result_locked(which, std::move(t), l);

                return f;
            }

            void set_result(std::size_t which, T && t)
            {
                std::unique_lock<mutex_type> l(mtx_);
                set_result_locked(which, std::move(t), l);
            }

            HPX_DEFINE_COMPONENT_ACTION(
                gather_server, get_result, get_result_action);
            HPX_DEFINE_COMPONENT_ACTION(
                gather_server, set_result, set_result_action);

        protected:
            template <typename Lock>
            void set_result_locked(std::size_t which, T && t, Lock& l)
            {
                gate_.synchronize(1, l);
                data_[which] = std::move(t);
                gate_.set(which, l);
            }

        private:
            mutex_type mtx_;
            std::size_t num_sites_;
            std::vector<T> data_;
            lcos::local::and_gate gate_;
        };

        ///////////////////////////////////////////////////////////////////////
        inline hpx::id_type register_gather_name(hpx::future<hpx::id_type> f,
            std::string const& basename, std::size_t site)
        {
            hpx::id_type target = f.get();
            hpx::register_with_basename(basename, hpx::unmanaged(target), site);
            return target;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        hpx::future<std::vector<T> >
        gather_data(hpx::future<hpx::id_type> f, std::size_t site,
            hpx::future<T> result)
        {
            typedef typename gather_server<T>::get_result_action action_type;
            return async(action_type(), f.get(), site, result.get());
        }

        template <typename T>
        hpx::future<void>
        set_data(hpx::future<hpx::id_type> f, std::size_t which,
            hpx::future<T> result)
        {
            typedef typename gather_server<T>::set_result_action action_type;
            return async(action_type(), f.get(), which, result.get());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type>
    create_gatherer(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        if (num_sites == std::size_t(-1))
            num_sites = hpx::get_num_localities_sync();
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        // create a new gather_server
        typedef typename util::decay<T>::type result_type;
        hpx::future<hpx::id_type> id =
            hpx::new_<detail::gather_server<result_type> >(
                hpx::find_here(), num_sites);

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        // register the gatherer's id using the given basename
        using util::placeholders::_1;
        return id.then(
                util::bind(&detail::register_gather_name, _1, name, this_site)
            );
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(hpx::future<hpx::id_type> f, hpx::future<T> result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using util::placeholders::_1;
        using util::placeholders::_2;
        return dataflow(
                util::bind(&detail::gather_data<T>, _1, this_site, _2),
                std::move(f), std::move(result)
            );
    }

    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(char const* basename, hpx::future<T> result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        if (num_sites == std::size_t(-1))
            num_sites = hpx::get_num_localities_sync();
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        return gather_here(
            create_gatherer<T>(basename, num_sites, generation, this_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // gather plain values
    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type> >
    gather_here(hpx::future<hpx::id_type> f, T && result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using util::placeholders::_1;
        using util::placeholders::_2;

        typedef typename util::decay<T>::type result_type;
        return dataflow(
                util::bind(&detail::gather_data<result_type>, _1, this_site, _2),
                std::move(f), std::forward<T>(result)
            );
    }

    template <typename T>
    hpx::future<std::vector<typename util::decay<T>::type> >
    gather_here(char const* basename, T && result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        if (num_sites == std::size_t(-1))
            num_sites = hpx::get_num_localities_sync();
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        return gather_here(
            create_gatherer<T>(basename, num_sites, generation, this_site),
            std::forward<T>(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<void>
    gather_there(hpx::future<hpx::id_type> id, hpx::future<T> result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using util::placeholders::_1;
        using util::placeholders::_2;
        return dataflow(
                util::bind(&detail::set_data<T>, _1, this_site, _2),
                std::move(id), std::move(result)
            );
    }

    template <typename T>
    hpx::future<void>
    gather_there(char const* basename, hpx::future<T> result,
        std::size_t generation = std::size_t(-1), std::size_t root_site = 0,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return gather_there(
            hpx::find_from_basename(name, root_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // gather plain values
    template <typename T>
    hpx::future<void>
    gather_there(hpx::future<hpx::id_type> id, T && result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        using util::placeholders::_1;
        using util::placeholders::_2;

        typedef typename util::decay<T>::type result_type;
        return dataflow(
                util::bind(&detail::set_data<result_type>, _1, this_site, _2),
                std::move(id), std::forward<T>(result)
            );
    }

    template <typename T>
    hpx::future<void>
    gather_there(char const* basename, T && result,
        std::size_t generation = std::size_t(-1), std::size_t root_site = 0,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
            this_site = static_cast<std::size_t>(hpx::get_locality_id());

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return gather_there(
            hpx::find_from_basename(name, root_site),
            std::forward<T>(result), this_site);
    }
}}

#define HPX_REGISTER_GATHER_DECLARATION(type, name)                           \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::detail::gather_server<type>::get_result_action,            \
        BOOST_PP_CAT(gather_get_result_action_, name));                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::detail::gather_server<type>::set_result_action,            \
        BOOST_PP_CAT(set_result_action_, name))                               \
    /**/

#define HPX_REGISTER_GATHER(type, name)                                       \
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::detail::gather_server<type>::get_result_action,            \
        BOOST_PP_CAT(gather_get_result_action_, name));                       \
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::detail::gather_server<type>::set_result_action,            \
        BOOST_PP_CAT(set_result_action_, name));                              \
    typedef hpx::components::simple_component<                                \
        hpx::lcos::detail::gather_server<type>                                \
    > BOOST_PP_CAT(gather_, name);                                            \
    HPX_REGISTER_COMPONENT(BOOST_PP_CAT(gather_, name))                       \
    /**/

#endif // DOXYGEN
#endif
