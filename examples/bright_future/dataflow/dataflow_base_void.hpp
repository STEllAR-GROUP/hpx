
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_VOID_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_VOID_HPP

#include <examples/bright_future/dataflow/dataflow_base_fwd.hpp>
#include <examples/bright_future/dataflow/stubs/dataflow.hpp>

namespace hpx { namespace lcos {
    template <>
    struct dataflow_base<void>
        : components::client_base<
            dataflow_base<void>
          , stubs::dataflow
        >
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;
        typedef 
            components::client_base<
                dataflow_base<void>
              , stubs::dataflow
            >
            base_type;
        
        typedef stubs::dataflow stub_type;

        dataflow_base() {}
        
        dataflow_base(naming::id_type const & id)
            : base_type(id)
        {}

        dataflow_base(dataflow_base const & other)
            : base_type(other.get_gid())
        {}
        
        operator promise<void, remote_result_type>() const
        {
            promise<void> p;
            connect(p.get_gid());
            return p;
        }

        void get()
        {
            promise<void> p;
            connect(p.get_gid());
            p.get();
        }

        void connect(naming::id_type const & target) const
        {
            stub_type::connect(this->get_gid(), target);
        }

    private:

        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & this->gid_;
            BOOST_ASSERT(this->get_gid());
        }
    };
}}

#endif
