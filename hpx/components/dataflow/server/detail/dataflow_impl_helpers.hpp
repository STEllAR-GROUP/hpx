//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_IMPL_HELPERS_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_IMPL_HELPERS_HPP

#include <hpx/components/dataflow/dataflow_base_fwd.hpp>
#include <hpx/components/dataflow/dataflow_fwd.hpp>
#include <hpx/components/dataflow/dataflow_trigger_fwd.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>
#include <hpx/util/decay.hpp>

#include <boost/mpl/at.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    template <typename T>
    struct dataflow_result
    {
        typedef typename util::decay<typename T::result_type>::type type;
    };

    template <typename Vector>
    struct map_type
    {
        typedef typename boost::mpl::at_c<Vector, 0>::type type;
    };

    template <typename Vector>
    struct slot_type
    {
        typedef typename boost::mpl::at_c<Vector, 1>::type type;
    };

    template <typename Vector>
    struct arg_type
    {
        typedef typename boost::mpl::at_c<Vector, 2>::type type;
    };

    template <typename T, typename Dummy = void>
    struct dataflow_is_void
        : boost::mpl::false_
    {};

    template <typename Action>
    struct dataflow_is_void<dataflow<Action, void> >
        : boost::mpl::true_
    {};

    template <typename RemoteResult>
    struct dataflow_is_void<hpx::lcos::dataflow_base<void, RemoteResult> >
        : boost::mpl::true_
    {};

    template <typename Dummy>
    struct dataflow_is_void<dataflow_trigger, Dummy>
        : boost::mpl::true_
    {};

    template <typename Vector>
    struct dataflow_non_void
    {
        typedef boost::mpl::at_c<Vector, 0> map_type;
        typedef boost::mpl::at_c<Vector, 1> slot_type;
        typedef boost::mpl::at_c<Vector, 1> arg_type;

        typedef
            typename boost::mpl::insert<
                typename map_type::type
              , boost::mpl::pair<
                    typename slot_type::type
                  , typename arg_type::type
                >
            >
            map_insert;

        typedef
            boost::mpl::vector<
                typename map_insert::type
              , typename boost::mpl::next<
                    typename slot_type::type
                >::type
              , typename boost::mpl::next<
                    typename arg_type::type
                >::type
            >
            type;
    };

    template <typename Vector>
    struct advance_slot
    {
        typedef boost::mpl::at_c<Vector, 0> map_type;
        typedef boost::mpl::at_c<Vector, 1> slot_type;
        typedef boost::mpl::at_c<Vector, 1> arg_type;

        typedef
            boost::mpl::vector<
                typename map_type::type
              , typename boost::mpl::next<
                    typename slot_type::type
                >::type
              , typename arg_type::type
            >
            type;
    };

    template <typename Vector>
    struct passed_args_transforms
    {
        typedef
            typename boost::fusion::result_of::as_vector<
                typename boost::mpl::fold<
                    Vector
                  , boost::fusion::vector<>
                  , boost::mpl::if_<
                        hpx::traits::is_dataflow<boost::mpl::_2>
                      , boost::mpl::if_<
                            dataflow_is_void<boost::mpl::_2>
                          , boost::mpl::_1
                          , boost::fusion::result_of::push_back<
                                boost::mpl::_1
                              , dataflow_result<
                                    boost::mpl::_2
                                >
                            >
                        >
                      , boost::fusion::result_of::push_back<
                            boost::mpl::_1
                          , util::decay<boost::mpl::_2>
                        >
                    >
                >::type
            >::type
            results_type;

        typedef
            typename boost::mpl::at_c<
                typename boost::mpl::fold<
                    Vector
                  , boost::mpl::vector<
                        boost::mpl::map<>   // Mapping
                      , boost::mpl::int_<0> // SlotIndex
                      , boost::mpl::int_<0> // ArgIndex
                    >
                  , boost::mpl::if_<
                        hpx::traits::is_dataflow<boost::mpl::_2>
                      , boost::mpl::if_<
                            dataflow_is_void<boost::mpl::_2>
                            // the result type of the passed dataflow is void
                          , advance_slot<
                                boost::mpl::_1
                            >
                          , dataflow_non_void<boost::mpl::_1>
                        >
                      , dataflow_non_void<boost::mpl::_1>
                    >
                >::type
              , 0
            >::type
            slot_to_args_map;
    };
}}}}
#endif
