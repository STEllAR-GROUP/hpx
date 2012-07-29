//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_DEFERRED_PACKAGED_TASK_CONSTRUCTORS_JUN_27_2008_0440PM)
#define HPX_LCOS_DEFERRED_PACKAGED_TASK_CONSTRUCTORS_JUN_27_2008_0440PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/lcos/preprocessed/deferred_packaged_task_constructors.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/deferred_packaged_task_constructors_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/deferred_packaged_task_constructors.hpp"))                      \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

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
        util::block_profiler_wrapper<deferred_packaged_task_tag> bp(apply_logger_);
        hpx::apply_c<Action>(
            this->get_gid(), gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

private:
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static void BOOST_PP_CAT(invoke,N)(
        hpx::lcos::deferred_packaged_task<Action,Result> *th,
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
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template BOOST_PP_CAT(invoke,N)<BOOST_PP_ENUM_PARAMS(N,Arg)>,
            this_(), naming::id_type(gid, naming::id_type::unmanaged),
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
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template BOOST_PP_CAT(invoke,N)<BOOST_PP_ENUM_PARAMS(N,Arg)>,
            this_(), gid,
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
