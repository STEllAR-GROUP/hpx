//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_IMPL_HPP
#define HPX_LCOS_DATAFLOW_BASE_IMPL_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos { namespace detail
{
    struct dataflow_base_impl
    {
        dataflow_base_impl()
        {}

        ~dataflow_base_impl()
        {}

        dataflow_base_impl(
            lcos::promise<naming::id_type, naming::gid_type> const & promise
        )
            : gid_promise(promise)
        {}

        void invalidate()
        {
            /*
            gid_promise.reset();
            */
        }

        naming::id_type get_gid() const
        {
            return gid_promise.get();
        }

    protected:
        lcos::promise<naming::id_type, naming::gid_type> gid_promise;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            naming::id_type id;
            ar & id;
            gid_promise.set_local_data(0, id);
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
