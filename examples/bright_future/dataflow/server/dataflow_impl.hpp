//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_IMPL_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_IMPL_HPP

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, BOOST_PP_DEC(HPX_ACTION_ARGUMENT_LIMIT),                          \
    <examples/bright_future/dataflow/server/dataflow_impl.hpp>))              \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !#if !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// init_action is the action that can be used to call the variadic
    /// function from a client
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct init_action<Action, BOOST_PP_ENUM_PARAMS(N, A)>
      : BOOST_PP_CAT(hpx::actions::direct_action, BOOST_PP_INC(N))<
            dataflow, 0, hpx::naming::id_type const &
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & BOOST_PP_INTERCEPT)
          , &dataflow::init<Action, BOOST_PP_ENUM_PARAMS(N, A)>
          , init_action<Action, BOOST_PP_ENUM_PARAMS(N, A)>
        >
    {
    private:
        typedef BOOST_PP_CAT(hpx::actions::direct_action, BOOST_PP_INC(N))<
            dataflow, 0, hpx::naming::id_type const &
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & BOOST_PP_INTERCEPT)
          , &dataflow::init<Action, BOOST_PP_ENUM_PARAMS(N, A)>
          , init_action<Action, BOOST_PP_ENUM_PARAMS(N, A)>
        > base_type;

    public:
        init_action() {}

        // construct an action from its arguments
        init_action(naming::id_type const & target
              , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const& a))
          : base_type(target, BOOST_PP_ENUM_PARAMS(N, a))
        {}

        init_action(threads::thread_priority p, naming::id_type const & target
              , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const& a))
          : base_type(p, target, BOOST_PP_ENUM_PARAMS(N, a))
        {}

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<init_action, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
}}}

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>)
  , (hpx::lcos::server::init_action<Action, BOOST_PP_ENUM_PARAMS(N, A)>)
)

#undef N

#endif

