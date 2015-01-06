//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM)
#define HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace parallel { namespace util { namespace remote
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct algorithm_result
        {
            template <typename T_>
            static T_ const& call(T_&& val)
            {
                return std::forward<T_>(val);
            }
        };

        template <typename Iterator>
        struct algorithm_result<
            Iterator,
            typename boost::enable_if<
                    typename hpx::traits::segmented_local_iterator_traits<
                            Iterator
                        >::is_segmented_local_iterator
                >::type>
        {
            typedef hpx::traits::segmented_local_iterator_traits<Iterator> traits;

            static Iterator call(typename traits::local_raw_iterator&& it)
            {
                return traits::remote(std::move(it));
            }
        };

        template <typename Iterator>
        struct algorithm_result<
            future<Iterator>,
            typename boost::enable_if<
                    typename hpx::traits::segmented_local_iterator_traits<
                            Iterator
                        >::is_segmented_local_iterator
                >::type>
        {
            typedef hpx::traits::segmented_local_iterator_traits<Iterator> traits;

            static future<Iterator>
            call(future<typename traits::local_raw_iterator>&& f)
            {
                typedef future<typename traits::local_raw_iterator> argtype;
                return f.then([](argtype&& f) { return traits::remote(f.get()); });
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher
    {
        static typename parallel::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<Algo>::type::result_type
        >::type
        sequential(Algo const& algo, ExPolicy const& policy, Args const&... args)
        {
            typedef typename parallel::detail::algorithm_result<
                    ExPolicy, typename hpx::util::decay<Algo>::type::result_type
                >::type result_type;

            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result<result_type>::call(
                    algo.call(policy, boost::mpl::true_(),
                        segmented_local_iterator_traits<Args>::local(args)...
                    )
                );
        }

        static typename parallel::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<Algo>::type::result_type
        >::type
        parallel(Algo const& algo, ExPolicy const& policy, Args const&... args)
        {
            typedef typename parallel::detail::algorithm_result<
                    ExPolicy, typename hpx::util::decay<Algo>::type::result_type
                >::type result_type;

            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result<result_type>::call(
                    algo.call(policy, boost::mpl::false_(),
                        segmented_local_iterator_traits<Args>::local(args)...
                    )
                );
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename F>
    struct algorithm_invoker_action;

    // sequential
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::true_, R(Args const& ...)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&, Args const&...),
            &dispatcher<Algo, ExPolicy, Args...>::sequential,
            algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::true_, R(Args const& ...)>
        >
    {};

    // parallel
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::false_, R(Args const& ...)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&, Args const&...),
            &dispatcher<Algo, ExPolicy, Args...>::parallel,
            algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::false_, R(Args const& ...)>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    BOOST_FORCEINLINE
    future<typename hpx::util::decay<Algo>::type::result_type>
    dispatch_async(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq, Args&&... args)
    {
        typedef typename hpx::util::decay<Algo>::type algo_type;
        typedef typename parallel::detail::algorithm_result<
                ExPolicy, typename algo_type::result_type
            >::type result_type;

        algorithm_invoker_action<
            algo_type, ExPolicy, IsSeq,
            result_type(typename hpx::util::decay<Args>::type const&...)
        > act;

        return hpx::async_colocated(act, id, std::forward<Algo>(algo), policy,
            std::forward<Args>(args)...);
    }

    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    BOOST_FORCEINLINE
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
            std::list<boost::exception_ptr> errors;
            parallel::util::detail::handle_remote_exceptions<
                    ExPolicy
                >::call(f.get_exception_ptr(), errors);

            HPX_ASSERT(errors.empty());
            boost::throw_exception(exception_list(std::move(errors)));
        }
        return f.get();
    }
}}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Algo, typename ExPolicy, typename IsSeq, typename R,
        typename... Args>),
    (hpx::parallel::util::remote::algorithm_invoker_action<
        Algo, ExPolicy, IsSeq, R(Args const&...)>))

#endif
