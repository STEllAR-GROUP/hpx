//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM)
#define HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/colocating_distribution_policy.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/datastructures/tuple.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <exception>
#include <list>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct algorithm_result_helper
    {
        template <typename T_>
        static HPX_FORCEINLINE T_ call(T_&& val)
        {
            return std::forward<T_>(val);
        }
    };

    template <>
    struct algorithm_result_helper<future<void> >
    {
        static HPX_FORCEINLINE future<void> call(future<void>&& f)
        {
            return std::move(f);
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<
        Iterator,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator> traits;

        static HPX_FORCEINLINE Iterator
        call(typename traits::local_raw_iterator && it)
        {
            return traits::remote(std::move(it));
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<
        std::pair<Iterator1, Iterator2>,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator2>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1> traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2> traits2;

        static HPX_FORCEINLINE
        std::pair<
            typename traits1::local_iterator, typename traits2::local_iterator
        >
        call(std::pair<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator
            > && p)
        {
            return std::make_pair(
                traits1::remote(std::move(p.first)),
                traits2::remote(std::move(p.second)));
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<
        hpx::util::tuple<Iterator1, Iterator2, Iterator3>,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator3>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1> traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2> traits2;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator3> traits3;

        static HPX_FORCEINLINE
        hpx::util::tuple<
            typename traits1::local_iterator, typename traits2::local_iterator,
            typename traits3::local_iterator
        >
        call(hpx::util::tuple<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator,
                typename traits3::local_raw_iterator
            > && p)
        {
            return hpx::util::make_tuple(
                traits1::remote(std::move(hpx::util::get<0>(p))),
                traits2::remote(std::move(hpx::util::get<1>(p))),
                traits3::remote(std::move(hpx::util::get<2>(p))));
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<
        future<Iterator>,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator> traits;

        static HPX_FORCEINLINE future<Iterator>
        call(future<typename traits::local_raw_iterator>&& f)
        {
            typedef future<typename traits::local_raw_iterator> argtype;
            return f.then(
                hpx::launch::sync,
                [](argtype&& f) -> Iterator
                {
                    return traits::remote(f.get());
                });
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<
        future<std::pair<Iterator1, Iterator2> >,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator2>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1> traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2> traits2;

        typedef std::pair<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator
            > arg_type;

        static HPX_FORCEINLINE
        future<std::pair<
            typename traits1::local_iterator, typename traits2::local_iterator
        > >
        call(future<arg_type> && f)
        {
            return f.then(
                hpx::launch::sync,
                [](future<arg_type> && f)
                ->  std::pair<
                        typename traits1::local_iterator,
                        typename traits2::local_iterator
                    >
                {
                    auto && p = f.get();
                    return std::make_pair(
                        traits1::remote(p.first),
                        traits2::remote(p.second));
                });
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<
        future<hpx::util::tuple<Iterator1, Iterator2, Iterator3> >,
        typename std::enable_if<
                hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
                hpx::traits::is_segmented_local_iterator<Iterator3>::value
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1> traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2> traits2;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator3> traits3;

        typedef hpx::util::tuple<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator,
                typename traits3::local_raw_iterator
            > arg_type;

        static HPX_FORCEINLINE
        future<hpx::util::tuple<
            typename traits1::local_iterator, typename traits2::local_iterator,
            typename traits3::local_iterator
        > >
        call(future<arg_type> && f)
        {
            return f.then(
                hpx::launch::sync,
                [](future<arg_type> && f)
                ->  hpx::util::tuple<
                        typename traits1::local_iterator,
                        typename traits2::local_iterator,
                        typename traits3::local_iterator
                    >
                {
                    auto && p = f.get();
                    return hpx::util::make_tuple(
                        traits1::remote(std::move(hpx::util::get<0>(p))),
                        traits2::remote(std::move(hpx::util::get<1>(p))),
                        traits3::remote(std::move(hpx::util::get<2>(p))));
                });
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename Algo>
    struct dispatcher_helper
    {
        template <typename ExPolicy, typename ... Args>
        static HPX_FORCEINLINE R sequential(Algo const& algo,
            ExPolicy && policy, Args &&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result_helper<R>::call(
                    algo.call(std::forward<ExPolicy>(policy), std::true_type(),
                        segmented_local_iterator_traits<
                            typename hpx::util::decay<Args>::type
                        >::local(std::forward<Args>(args))...
                    )
                );
        }

        template <typename ExPolicy, typename ... Args>
        static HPX_FORCEINLINE R parallel(Algo const& algo,
            ExPolicy && policy, Args &&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result_helper<R>::call(
                    algo.call(std::forward<ExPolicy>(policy), std::false_type(),
                        segmented_local_iterator_traits<
                            typename hpx::util::decay<Args>::type
                        >::local(std::forward<Args>(args))...
                    )
                );
        }
    };

    template <typename Algo>
    struct dispatcher_helper<void, Algo>
    {
        template <typename ExPolicy, typename ... Args>
        static HPX_FORCEINLINE
        typename parallel::util::detail::algorithm_result<ExPolicy>::type
        sequential(Algo const& algo, ExPolicy && policy, Args &&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return algo.call(std::forward<ExPolicy>(policy), std::true_type(),
                        segmented_local_iterator_traits<
                            typename hpx::util::decay<Args>::type
                        >::local(std::forward<Args>(args))...
                );
        }

        template <typename ExPolicy, typename ... Args>
        static HPX_FORCEINLINE
        typename parallel::util::detail::algorithm_result<ExPolicy>::type
        parallel(Algo const& algo, ExPolicy && policy, Args &&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return algo.call(std::forward<ExPolicy>(policy), std::false_type(),
                        segmented_local_iterator_traits<
                            typename hpx::util::decay<Args>::type
                        >::local(std::forward<Args>(args))...
                );
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher
    {
        typedef typename parallel::util::detail::algorithm_result<
                ExPolicy, typename hpx::util::decay<Algo>::type::result_type
            >::type result_type;

        typedef dispatcher_helper<result_type, Algo> base_dispatcher;

        static HPX_FORCEINLINE result_type sequential(Algo const& algo,
            ExPolicy policy, Args... args)
        {
            return base_dispatcher::sequential(algo, std::move(policy),
                std::move(args)...);
        }

        static HPX_FORCEINLINE result_type parallel(Algo const& algo,
            ExPolicy policy, Args ... args)
        {
            return base_dispatcher::parallel(algo, std::move(policy),
                std::move(args)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename F>
    struct algorithm_invoker_action;

    // sequential
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<Algo, ExPolicy, std::true_type, R(Args...)>
      : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy, Args...),
            &dispatcher<Algo, ExPolicy, Args...>::sequential,
            algorithm_invoker_action<Algo, ExPolicy, std::true_type, R(Args...)>
        >::type
    {};

    // parallel
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<Algo, ExPolicy, std::false_type, R(Args...)>
      : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy, Args...),
            &dispatcher<Algo, ExPolicy, Args...>::parallel,
            algorithm_invoker_action<Algo, ExPolicy, std::false_type, R(Args...)>
        >::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    HPX_FORCEINLINE
    future<typename hpx::util::decay<Algo>::type::result_type>
    dispatch_async(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq, Args&&... args)
    {
        typedef typename hpx::util::decay<Algo>::type algo_type;
        typedef typename parallel::util::detail::algorithm_result<
                ExPolicy, typename algo_type::result_type
            >::type result_type;

        algorithm_invoker_action<
            algo_type, ExPolicy, typename IsSeq::type,
            result_type(typename hpx::util::decay<Args>::type...)
        > act;

        return hpx::async(act, hpx::colocated(id), std::forward<Algo>(algo),
            policy, std::forward<Args>(args)...);
    }

    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    HPX_FORCEINLINE
    typename hpx::util::decay<Algo>::type::result_type
    dispatch(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq is_seq, Args&&... args)
    {
        // synchronously invoke remote operation
        future<typename hpx::util::decay<Algo>::type::result_type> f =
            dispatch_async(id, std::forward<Algo>(algo), policy, is_seq,
                std::forward<Args>(args)...);
        f.wait();

        // handle any remote exceptions
        if (f.has_exception())
        {
            std::list<std::exception_ptr> errors;
            parallel::util::detail::handle_remote_exceptions<
                    ExPolicy
                >::call(f.get_exception_ptr(), errors); // NOLINT(bugprone-use-after-move)

            // NOLINTNEXTLINE(bugprone-use-after-move)
            HPX_ASSERT(errors.empty());
            throw exception_list(std::move(errors));
        }
        return f.get();
    }
}}}}

#endif
