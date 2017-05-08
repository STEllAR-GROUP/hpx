//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM)
#define HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>

#include <string>
#include <type_traits>
#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
#include <typeinfo>
#endif
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct local_algorithm_result
    {
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result
            >::local_raw_iterator type;
    };

    template <typename Result1, typename Result2>
    struct local_algorithm_result<std::pair<Result1, Result2> >
    {
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result1
            >::local_raw_iterator type1;
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result2
            >::local_raw_iterator type2;

        typedef std::pair<type1, type2> type;
    };

    template <>
    struct local_algorithm_result<void>
    {
        typedef void type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
    private:
        Derived const& derived() const
        {
            return static_cast<Derived const&>(*this);
        }

    public:
        typedef Result result_type;
        typedef typename local_algorithm_result<result_type>::type
            local_result_type;

        explicit algorithm(char const* const name)
          : name_(name)
        {}

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        HPX_HOST_DEVICE
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        operator()(ExPolicy && policy, Args&&... args) const
        {
#if !defined(__CUDA_ARCH__)
            try {
#endif
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;

                parallel::util::detail::scoped_executor_parameters<
                        parameters_type
                    > scoped_param(policy.parameters());

                return parallel::util::detail::algorithm_result<
                        ExPolicy, local_result_type
                    >::get(
                        Derived::sequential(std::forward<ExPolicy>(policy),
                            std::forward<Args>(args)...)
                    );
#if !defined(__CUDA_ARCH__)
            }
            catch(...) {
                // this does not return
                return detail::handle_exception<ExPolicy, local_result_type>::call();
            }
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call_execute(ExPolicy && policy, std::false_type, Args&&... args) const
        {
            return parallel::util::detail::algorithm_result<
                    ExPolicy, local_result_type
                >::get(execution::sync_execute(policy.executor(), derived(),
                    std::forward<ExPolicy>(policy), std::forward<Args>(args)...));
        }

        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<ExPolicy>::type
        call_execute(ExPolicy && policy, std::true_type, Args&&... args) const
        {
            execution::sync_execute(policy.executor(), derived(),
                std::forward<ExPolicy>(policy), std::forward<Args>(args)...);

            return parallel::util::detail::algorithm_result<ExPolicy>::get();
        }

        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call(ExPolicy && policy, std::true_type, Args&&... args) const
        {
            try {
                typedef std::is_void<local_result_type> is_void;
                return call_execute(std::forward<ExPolicy>(policy),
                    is_void(), std::forward<Args>(args)...);
            }
            catch (...) {
                return detail::handle_exception<ExPolicy, local_result_type>::
                    call();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call_sequential(ExPolicy && policy, Args&&... args) const
        {
            try {
                // run the launched task on the requested executor
                hpx::future<local_result_type> result =
                    execution::async_execute(policy.executor(), derived(),
                        std::forward<ExPolicy>(policy),
                        std::forward<Args>(args)...);

                return parallel::util::detail::algorithm_result<
                        ExPolicy, local_result_type
                    >::get(std::move(result));
            }
            catch (...) {
                return detail::handle_exception<ExPolicy, local_result_type>::
                    call();
            }
        }

        template <typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::sequenced_task_policy, local_result_type
        >::type
        call(execution::sequenced_task_policy policy, std::true_type,
            Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::sequenced_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::sequenced_task_policy_shim<
                Executor, Parameters>& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::sequenced_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::sequenced_task_policy_shim<
                Executor, Parameters> && policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::sequenced_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::sequenced_task_policy_shim<
                Executor, Parameters> const& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

#if defined(HPX_HAVE_DATAPAR)
        template <typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::dataseq_task_policy, local_result_type
        >::type
        call(execution::dataseq_task_policy policy, std::true_type,
            Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::dataseq_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::dataseq_task_policy_shim<Executor, Parameters>& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::dataseq_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::dataseq_task_policy_shim<
                Executor, Parameters> && policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::dataseq_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::dataseq_task_policy_shim<
                Executor, Parameters> const& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::parallel_task_policy, local_result_type
        >::type
        call(execution::parallel_task_policy policy, std::true_type,
            Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::parallel_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::parallel_task_policy_shim<Executor, Parameters>& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::parallel_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::parallel_task_policy_shim<Executor, Parameters> && policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::parallel_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::parallel_task_policy_shim<Executor, Parameters> const& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

#if defined(HPX_HAVE_DATAPAR)
        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::datapar_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::datapar_task_policy_shim<Executor, Parameters>& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::datapar_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::datapar_task_policy_shim<
                Executor, Parameters> && policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            execution::datapar_task_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(execution::datapar_task_policy_shim<
                Executor, Parameters> const& policy,
            std::true_type, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }
#endif

        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call(ExPolicy && policy, std::false_type, Args&&... args) const
        {
            return Derived::parallel(std::forward<ExPolicy>(policy),
                std::forward<Args>(args)...);
        }

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
        ///////////////////////////////////////////////////////////////////////////
        template <typename... Args>
        local_result_type
        call(parallel::execution_policy policy, std::false_type,
            Args&&... args) const
        {
            // this implementation is not nice, however we don't have variadic
            // virtual functions accepting template arguments and supporting
            // perfect forwarding
            std::type_info const& t = policy.type();

            if (t == typeid(execution::sequenced_policy))
            {
                return call(
                    *policy.get<execution::sequenced_policy>(),
                    std::true_type(), std::forward<Args>(args)...);
            }

            if (t == typeid(execution::sequenced_task_policy))
            {
                return call(execution::seq, std::true_type(),
                    std::forward<Args>(args)...);
            }

            if (t == typeid(execution::parallel_policy))
            {
                return call(
                    *policy.get<execution::parallel_policy>(),
                    std::false_type(), std::forward<Args>(args)...);
            }

            if (t == typeid(execution::parallel_task_policy))
            {
                execution::parallel_task_policy const& t =
                    *policy.get<execution::parallel_task_policy>();

                return call(execution::par.with(t.parameters()),
                    std::false_type(), std::forward<Args>(args)...);
            }

            if (t == typeid(execution::parallel_unsequenced_policy))
            {
                return call(
                    *policy.get<execution::parallel_unsequenced_policy>(),
                    std::false_type(), std::forward<Args>(args)...);
            }

            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                std::string("hpx::parallel::") + name_,
                "The given execution policy is not supported");
        }

        template <typename... Args>
        local_result_type
        call(parallel::execution_policy, std::true_type, Args&&... args) const
        {
            return call(execution::seq, std::true_type(), std::forward<Args>(args)...);
        }
#endif

    private:
        char const* const name_;

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive&, unsigned int)
        {
            // no need to serialize 'name_' as it is always initialized by the
            // constructor
        }
    };
}}}}

#endif
