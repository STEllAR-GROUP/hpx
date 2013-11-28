//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_DEFERRED_PACKAGED_TASK_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM)
#define HPX_LCOS_DEFERRED_PACKAGED_TASK_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM

#include <hpx/util/move.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/deferred_packaged_task_constructors_direct.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/deferred_packaged_task_constructors_direct_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/deferred_packaged_task_constructors_direct.hpp"))               \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply(naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);

        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            // local, direct execution
            HPX_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<typename Action::component_type>()));
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg))));
        }
        else {
            // remote execution
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

private:
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static void BOOST_PP_CAT(invoke,N)(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

public:
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template BOOST_PP_CAT(invoke,N)<BOOST_PP_ENUM_PARAMS(N, Arg)>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg)))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template BOOST_PP_CAT(invoke,N)<BOOST_PP_ENUM_PARAMS(N, Arg)>,
            this, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg)))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
    }

#undef N

#endif
