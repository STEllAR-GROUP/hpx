//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file gather.hpp

#if !defined(HPX_LCOS_GATHER_MAY_05_2014_0418PM)
#define HPX_LCOS_GATHER_MAY_05_2014_0418PM

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(char const* basename, hpx::future<T> result,
        std::size_t num_sites = ~0U, std::size_t this_site = ~0U);

    template <typename T>
    hpx::future<void>
    gather_there(char const* basename, hpx::future<T> result,
        std::size_t root_site = 0, std::size_t this_site = ~0U);
}}
#else

#include <hpx/config.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/local/and_gate.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/thread/locks.hpp>

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
                return std::move(data_);
            }

        public:
            gather_server()
            {
                HPX_ASSERT(false);  // shouldn't ever be called
            }

            gather_server(std::size_t num_sites)
              : generation_(0), data_(num_sites), gate_(num_sites)
            {
            }

            hpx::future<std::vector<T> > get_result(std::size_t which, T && t)
            {
                boost::unique_lock<mutex_type> l(mtx_);

                using util::placeholders::_1;
                hpx::future<std::vector<T> > f = gate_.get_future().then(
                        util::bind(&gather_server::on_ready, this, _1));

                set_result_locked(which, std::move(t), l);

                return f;
            }

            void set_result(std::size_t which, T && t)
            {
                boost::unique_lock<mutex_type> l(mtx_);
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
            std::size_t generation_;
            std::vector<T> data_;
            lcos::local::and_gate gate_;
        };

        ///////////////////////////////////////////////////////////////////////
        hpx::id_type register_name(hpx::future<hpx::id_type> id,
            char const* basename, std::size_t site)
        {
            hpx::id_type target = id.get();
            hpx::register_with_basename(basename, target, site);
            return target;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        hpx::future<std::vector<T> >
        gather_data(hpx::future<hpx::id_type> id, std::size_t site,
            hpx::future<T> result)
        {
            typedef typename gather_server<T>::get_result_action action_type;
            return async(action_type(), id.get(), site, result.get());
        }

        template <typename T>
        hpx::future<void>
        set_data(hpx::future<hpx::id_type> id, std::size_t which,
            hpx::future<T> result)
        {
            typedef typename gather_server<T>::set_result_action action_type;
            return async(action_type(), id.get(), which, result.get());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type>
    create_gatherer(char const* basename, std::size_t num_sites = ~0U,
        std::size_t this_site = ~0U)
    {
        if (num_sites == ~0U)
            num_sites = hpx::get_num_localities_sync();
        if (this_site == ~0U)
            this_site = hpx::get_locality_id();

        // create a new gather_server
        hpx::future<hpx::id_type> id = hpx::new_<detail::gather_server<T> >(
            hpx::find_here(), num_sites);

        // register the gatherer's id using the given basename
        using util::placeholders::_1;
        return id.then(
                util::bind(&detail::register_name, _1, basename, this_site)
            );
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(hpx::future<hpx::id_type> id, hpx::future<T> result,
        std::size_t this_site = ~0U)
    {
        if (this_site == ~0U)
            this_site = hpx::get_locality_id();

        using util::placeholders::_1;
        using util::placeholders::_2;

        return dataflow(
                util::bind(&detail::gather_data<T>, _1, this_site, _2),
                std::move(id), std::move(result)
            );
    }

    template <typename T>
    hpx::future<std::vector<T> >
    gather_here(char const* basename, hpx::future<T> result,
        std::size_t num_sites = ~0U, std::size_t this_site = ~0U)
    {
        if (num_sites == ~0U)
            num_sites = hpx::get_num_localities_sync();
        if (this_site == ~0U)
            this_site = hpx::get_locality_id();

        return gather_here(
            create_gatherer<T>(basename, num_sites, this_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<void>
    gather_there(hpx::future<hpx::id_type> id, hpx::future<T> result,
        std::size_t this_site = ~0U)
    {
        if (this_site == ~0U)
            this_site = hpx::get_locality_id();

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
        std::size_t root_site = 0, std::size_t this_site = ~0U)
    {
        if (this_site == ~0U)
            this_site = hpx::get_locality_id();

        return gather_there(
            hpx::find_from_basename(basename, root_site),
            std::move(result), this_site);
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
