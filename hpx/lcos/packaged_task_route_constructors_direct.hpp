//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_PACKAGED_TASK_ROUTE_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM)
#define HPX_LCOS_PACKAGED_TASK_ROUTE_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/packaged_task_route_constructors_direct.hpp"))                  \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply(naming::id_type const& gid,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            // local, direct execution
            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<typename Action::component_type>()));
            (*this->impl_)->set_data(Action::execute_function(
                addr.address_, BOOST_PP_ENUM_PARAMS(N, arg)));
        }
        else {
            // remote execution
            hpx::applier::apply_c_route<Action>(addr, this->get_gid(), gid,
                BOOST_PP_ENUM_PARAMS(N, arg));
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    packaged_task_route(naming::id_type const& gid,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
      : apply_logger_("packaged_task_direct::apply")
    {
        LLCO_(info) << "packaged_task::packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
        apply(gid, BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    packaged_task_route(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
      : base_type(data_sink),
        apply_logger_("packaged_task_direct::apply")
    {
        LLCO_(info) << "packaged_task::packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
        apply(gid, BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif
