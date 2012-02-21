//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_DATAFLOW_OBJECT_HPP
#define HPX_COMPONENTS_DATAFLOW_OBJECT_HPP

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/remote_object/object.hpp>

namespace hpx { namespace components
{
    template <typename T>
    struct dataflow_object
    {
        dataflow_object() {}
        dataflow_object(dataflow_object const & o) : gid_(o.gid_) {}
        explicit dataflow_object(object<T> const & o) : gid_(o.gid_) {}
        explicit dataflow_object(naming::id_type const & gid) : gid_(gid) {}

        dataflow_object & operator=(dataflow_object const & o)
        {
            gid_ = o.gid_;
            return *this;
        }

        dataflow_object & operator=(object<T> const & o)
        {
            gid_ = o.gid_;
            return *this;
        }

        dataflow_object & operator=(naming::id_type const & gid)
        {
            gid_ = gid;
            return *this;
        }

        naming::id_type gid_;

        template <typename F>
        lcos::dataflow_base<
            typename boost::result_of<F(T &)>::type
        >
        apply(F const & f) const
        {
            typedef
                typename boost::result_of<F(T &)>::type
                result_type;

            typedef
                server::remote_object_apply_action<result_type>
                apply_action;

            return lcos::dataflow<apply_action>(gid_
                  , remote_object::invoke_apply_fun<T, F>(f)
                  , 0
                );
        }

        template <typename F, typename D>
        lcos::dataflow_base<
            typename boost::result_of<F(T &)>::type
        >
        apply(F const & f, D const & d) const
        {
            typedef
                typename boost::result_of<F(T &)>::type
                result_type;

            typedef
                server::remote_object_apply_action<result_type>
                apply_action;

            return lcos::dataflow<apply_action>(gid_
                  , remote_object::invoke_apply_fun<T, F>(f)
                  , 0
                  , d
                );
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & gid_;
        }
    };
}}

#endif
