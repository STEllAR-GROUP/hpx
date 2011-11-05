
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP

#include <examples/bright_future/dataflow/dataflow_base.hpp>
#include <examples/bright_future/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos {
    template <
        typename Action
      , typename Result
      , typename DirectExecute
    >
    struct dataflow
        : dataflow_base<Result, typename Action::result_type>
    {
        typedef typename Action::result_type remote_result_type;
        typedef Result result_type;
        typedef 
            dataflow_base<Result, typename Action::result_type>
            base_type;
            
        typedef stubs::dataflow stub_type;

        dataflow() {}
        
        ~dataflow()
        {
            LLCO_(info)
            //cout
                << "~dataflow::dataflow() ";
                //<< "\n" << flush;
        }

        dataflow(naming::id_type const & target_id)
            : base_type(stub_type::create_sync(target_id))
        {
            BOOST_ASSERT(this->get_gid());
            stubs::dataflow::init<Action>(this->get_gid(), target_id);
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow(                                                               \
            naming::id_type const & target_id                                   \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : base_type(stub_type::create_sync(target_id))                      \
        {                                                                       \
            BOOST_ASSERT(this->get_gid());                                      \
            stubs::dataflow::init<Action>(                                      \
                this->get_gid()                                                 \
              , target_id                                                       \
              , BOOST_PP_ENUM_PARAMS(N, a)                                      \
            );                                                                  \
        }                                                                       \
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_LCOS_DATAFLOW_M0
          , _
        )
#undef HPX_LCOS_DATAFLOW_M0
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
