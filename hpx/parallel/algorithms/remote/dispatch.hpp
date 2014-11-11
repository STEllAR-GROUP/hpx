//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM)
#define HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define HPX_DISPATCH_DECAY_ARG(Z, N, D)                                       \
    typename hpx::util::decay<BOOST_PP_CAT(D, N)>::type                       \
    /**/
#define HPX_DISPATCH_ARG(Z, N, D) BOOST_PP_CAT(D, N) const&                   \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, 5, "hpx/parallel/algorithms/remote/dispatch.hpp"))                \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_DISPATCH_ARG
#undef HPX_DISPATCH_DECAY_ARG

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx { namespace parallel { namespace util { namespace remote
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(dispatcher, N)
    {
        static typename parallel::v1::detail::algorithm_result<
            ExPolicy, typename Algo::result_type
        >::type
        sequential(Algo const& algo, ExPolicy const& policy,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            return algo.call(policy, BOOST_PP_ENUM_PARAMS(N, arg),
                boost::mpl::true_());
        }

        static typename parallel::v1::detail::algorithm_result<
            ExPolicy, typename Algo::result_type
        >::type
        parallel(Algo const& algo, ExPolicy const& policy,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            return algo.call(policy, BOOST_PP_ENUM_PARAMS(N, arg),
                boost::mpl::false_());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Algo, typename ExPolicy, typename IsSeq, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(algorithm_invoker_action, N);

    // sequential
    template <typename Algo, typename ExPolicy, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(algorithm_invoker_action, N)<
                Algo, ExPolicy, boost::mpl::true_, R,
                BOOST_PP_ENUM_PARAMS(N, Arg)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&,
                BOOST_PP_ENUM(N, HPX_DISPATCH_ARG, Arg)),
            &BOOST_PP_CAT(dispatcher, N)<
                    Algo, ExPolicy, BOOST_PP_ENUM_PARAMS(N, Arg)
                >::sequential,
            BOOST_PP_CAT(algorithm_invoker_action, N)<
                Algo, ExPolicy, boost::mpl::true_, R,
                BOOST_PP_ENUM_PARAMS(N, Arg)>
        >
    {};

    // parallel
    template <typename Algo, typename ExPolicy, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(algorithm_invoker_action, N)<
                Algo, ExPolicy, boost::mpl::false_, R,
                BOOST_PP_ENUM_PARAMS(N, Arg)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&,
                BOOST_PP_ENUM(N, HPX_DISPATCH_ARG, Arg)),
            &BOOST_PP_CAT(dispatcher, N)<
                    Algo, ExPolicy, BOOST_PP_ENUM_PARAMS(N, Arg)
                >::parallel,
            BOOST_PP_CAT(algorithm_invoker_action, N)<
                Algo, ExPolicy, boost::mpl::false_, R,
                BOOST_PP_ENUM_PARAMS(N, Arg)>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    future<typename Algo::result_type>
    dispatch_async(id_type const& id, Algo && algo,
        ExPolicy const& policy, IsSeq, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        BOOST_PP_CAT(algorithm_invoker_action, N)<
            typename hpx::util::decay<Algo>::type, ExPolicy, IsSeq,
            typename parallel::v1::detail::algorithm_result<
                ExPolicy, typename Algo::result_type
            >::type,
            BOOST_PP_ENUM(N, HPX_DISPATCH_DECAY_ARG, Arg)
        > act;

        return hpx::async_colocated(act, id, std::forward<Algo>(algo), policy,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Algo, typename ExPolicy, typename IsSeq,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename Algo::result_type
    dispatch(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq is_seq, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return dispatch_async(id, std::forward<Algo>(algo), policy, is_seq,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg)).get();
    }
}}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Algo, typename ExPolicy, typename IsSeq, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>),
    (hpx::parallel::util::remote::BOOST_PP_CAT(algorithm_invoker_action, N)<
        Algo, ExPolicy, IsSeq, R, BOOST_PP_ENUM_PARAMS(N, Arg)>))

#undef N

#endif
