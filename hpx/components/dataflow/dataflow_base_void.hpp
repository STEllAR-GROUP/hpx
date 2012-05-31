//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_VOID_HPP
#define HPX_LCOS_DATAFLOW_BASE_VOID_HPP

#include <hpx/components/dataflow/dataflow_base_fwd.hpp>
#include <hpx/components/dataflow/dataflow_base_impl.hpp>
#include <hpx/components/dataflow/stubs/dataflow.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace hpx { namespace lcos
{
    template <>
    struct dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;

        dataflow_base()
        {}

        dataflow_base(future<naming::id_type> const & promise)
            : impl(new detail::dataflow_base_impl(promise))
        {}

        void connect(naming::id_type const & id) const
        {
            impl->connect(id);
        }

        future<void> get_future() const
        {
            promise<void> p;
            impl->connect(p.get_gid());
            return p.get_future();
        }

        bool valid()
        {
            return impl && impl->get_gid();
        }

    private:
        friend class boost::serialization::access;

        boost::shared_ptr<detail::dataflow_base_impl> impl;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & impl;
        }
    };
}}

#endif
