//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_IMPL_HPP
#define HPX_LCOS_DATAFLOW_BASE_IMPL_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos { namespace detail
{
    struct dataflow_base_impl
    {
        dataflow_base_impl()
        {}

        virtual ~dataflow_base_impl()
        {}

        dataflow_base_impl(lcos::future<naming::id_type> const & promise)
            : gid_promise(promise)
        {}

        void add_target(naming::id_type const & id) const
        {
            typedef
                typename hpx::lcos::base_lco::connect_action
                action_type;

            BOOST_ASSERT(gid_promise.is_set());

            //hpx::applier::apply<action_type>(gid_promise.get(), id);
            hpx::lcos::async<action_type>(gid_promise.get(), id).get();
        }

        naming::id_type get_gid() const
        {
            return gid_promise.get();
        }

    protected:
        lcos::future<naming::id_type> gid_promise;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            naming::id_type id;
            ar & id;

            lcos::promise<naming::id_type> p;
            p.set_local_data(id);
            gid_promise = p.get_future();
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            naming::id_type id = this->get_gid();
            ar & id;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER();
    };
}}}

#endif
