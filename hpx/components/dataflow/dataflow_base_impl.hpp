//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_IMPL_HPP
#define HPX_LCOS_DATAFLOW_BASE_IMPL_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/include/async.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos { namespace detail
{
    struct dataflow_base_impl
    {
        dataflow_base_impl()
            : count_(0)
        {}

        virtual ~dataflow_base_impl()
        {}

        dataflow_base_impl(lcos::future<naming::id_type> const & promise)
            : gid_promise(promise)
            , count_(0)
        {}

        void connect(naming::id_type const & id) const
        {
            typedef
                hpx::lcos::base_lco::connect_action
                action_type;

            BOOST_ASSERT(gid_promise.get_state() != hpx::lcos::future_status::uninitialized);

            hpx::apply<action_type>(gid_promise.get(), id);
        }

        naming::id_type get_gid() const
        {
            return gid_promise.get();
        }

    protected:
        lcos::future<naming::id_type> gid_promise;

    private:
        friend class boost::serialization::access;
        
        boost::detail::atomic_count count_;

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            naming::id_type id;
            ar & id;

            lcos::promise<naming::id_type, naming::gid_type> p;
            p.set_local_data(id);
            gid_promise = p.get_future();
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            naming::id_type id = this->get_gid();
            ar & id;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()
    
    friend void intrusive_ptr_add_ref(dataflow_base_impl * p)
    {
        ++p->count_;
    }
    friend void intrusive_ptr_release(dataflow_base_impl * p)
    {
        if (0 == --p->count_)
            delete p;
    }
    };
}}}

#endif
