
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_HPP

#include <examples/bright_future/dataflow/dataflow_base_fwd.hpp>
#include <examples/bright_future/dataflow/dataflow_base_void.hpp>
#include <examples/bright_future/dataflow/stubs/dataflow.hpp>

namespace hpx { namespace lcos {
    template <typename Result, typename RemoteResult>
    struct dataflow_base
        : components::client_base<
            dataflow_base<Result, RemoteResult>
          , stubs::dataflow
        >
    {
        typedef RemoteResult remote_result_type;
        typedef Result       result_type;
        typedef 
            components::client_base<
                dataflow_base<Result, RemoteResult>
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

        operator promise<Result, remote_result_type>() const
        {
            promise<Result, remote_result_type> p;
            connect(p.get_gid());
            return p;
        }

        Result get()
        {
            promise<Result, remote_result_type> p;
            connect(p.get_gid());
            return p.get();
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
