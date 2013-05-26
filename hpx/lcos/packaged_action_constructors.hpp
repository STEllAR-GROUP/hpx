//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_PACKAGED_ACTION_CONSTRUCTORS_JUN_27_2008_0440PM)
#define HPX_LCOS_PACKAGED_ACTION_CONSTRUCTORS_JUN_27_2008_0440PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/packaged_action_constructors.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/packaged_action_constructors_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/packaged_action_constructors.hpp"))                             \
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
    void apply(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

        naming::address addr;
        if (policy == launch::sync && agas::is_local_address(gid, addr)) {
            // local, direct execution
            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            (*this->impl_)->set_data(
                boost::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)))));
        }
        else {
            using HPX_STD_PLACEHOLDERS::_1;
            using HPX_STD_PLACEHOLDERS::_2;

            hpx::apply_c_cb<action_type>(this->get_gid(), gid,
                HPX_STD_BIND(&packaged_action::parcel_write_handler, this, _1, _2),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply_p(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

        naming::address addr;
        if (policy == launch::sync && agas::is_local_address(gid, addr)) {
            // local, direct execution
            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            (*this->impl_)->set_data(
                boost::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)))));
        }
        else {
            using HPX_STD_PLACEHOLDERS::_1;
            using HPX_STD_PLACEHOLDERS::_2;

            hpx::apply_c_p_cb<action_type>(this->get_gid(), gid, priority,
                HPX_STD_BIND(&packaged_action::parcel_write_handler, this, _1, _2),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    packaged_action(naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
        apply(launch::all, gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
        apply_p(launch::all, naming::id_type(gid, naming::id_type::unmanaged),
            priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

#undef N

#endif
