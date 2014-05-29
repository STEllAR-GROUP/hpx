//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM)
#define HPX_STL_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/decay.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <memory>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct parallel_execution_policy {};

    /// Default parallel execution policy object.
    parallel_execution_policy const par;

    /// The class sequential_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct sequential_execution_policy {};

    /// Default sequential execution policy object.
    sequential_execution_policy const seq;

    /// The class vector_execution_policy is an execution policy type used as
    /// a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be vectorized.
    struct vector_execution_policy {};

    /// Default vector execution policy object.
    vector_execution_policy const vec;

    /// extension:
    ///
    /// The class task_execution_policy is an execution policy type used as
    /// a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized. The
    /// algorithm returns a future representing the result of the corresponding
    /// algorithm when invoked with the parallel_execution_policy.
    struct task_execution_policy {};

    /// Default vector execution policy object.
    task_execution_policy const task;


    ///////////////////////////////////////////////////////////////////////////
    ///
    /// 1. The type is_execution_policy can be used to detect parallel
    ///    execution policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_execution_policy<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_execution_policy is undefined.
    ///
    template <typename T>
    struct is_execution_policy
      : boost::mpl::false_
    {};

    template <>
    struct is_execution_policy<parallel_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_execution_policy<vector_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_execution_policy<sequential_execution_policy>
      : boost::mpl::true_
    {};

    // extension
    template <>
    struct is_execution_policy<task_execution_policy>
      : boost::mpl::true_
    {};

    class execution_policy;

    template <>
    struct is_execution_policy<execution_policy>
      : boost::mpl::true_
    {};

    // extension: detect whether give execution policy enables parallelization
    template <typename T>
    struct is_parallel_execution_policy
      : boost::mpl::false_
    {};

    template <>
    struct is_parallel_execution_policy<parallel_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_parallel_execution_policy<vector_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_parallel_execution_policy<task_execution_policy>
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    /// 1. The class execution_policy is a dynamic container for execution
    ///    policy objects. execution_policy allows dynamic control over
    ///    standard algorithm execution.
    /// 2. Objects of type execution_policy shall be constructible and
    ///    assignable from objects of type T for which
    ///    is_execution_policy<T>::value is true.
    class execution_policy
    {
    private:
        boost::shared_ptr<void> inner_;
        std::type_info const* type_;

    public:
        /// Effects: Constructs an execution_policy object with a copy of
        ///          exec's state
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        execution_policy(ExPolicy const& policy,
                typename boost::enable_if<
                    is_execution_policy<ExPolicy>, ExPolicy
                >::type* = 0)
          : inner_(boost::make_shared<ExPolicy>(policy)),
            type_(&typeid(ExPolicy))
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Cannot assign dynamic execution policy.");
        }

        /// Move constructs a new execution_policy object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy(execution_policy && policy)
          : inner_(std::move(policy.inner_)),
            type_(policy.type_)
        {
            policy.type_ = 0;
        }

        /// Effects: Assigns a copy of exec’s state to *this
        /// Returns: *this
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        typename boost::enable_if<
            is_execution_policy<ExPolicy>, execution_policy
        >::type&
        operator=(ExPolicy const& policy)
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Cannot assign dynamic execution policy.");

            inner_ = boost::make_shared<ExPolicy>(policy);
            type_ = &typeid(ExPolicy);
            return *this;
        }

        /// Move assigns a new execution policy to the object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy& operator=(execution_policy && policy)
        {
            if (this != &policy)
            {
                inner_ = std::move(policy.inner_);
                type_ = policy.type_;
                policy.type_ = 0;
            }
            return *this;
        }

        /// Returns: typeid(T), such that T is the type of the execution policy
        ///          object contained by *this
        std::type_info const& type() const BOOST_NOEXCEPT
        {
            return *type_;
        }

        /// Returns: If target_type() == typeid(T), a pointer to the stored
        ///          execution policy object; otherwise a null pointer
        /// Requires: is_execution_policy<T>::value is true
        template <typename ExPolicy>
        ExPolicy* get() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Incorrect execution policy parameter.");
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (*type_ != typeid(_ExPolicy))
                return 0;

            return static_cast<ExPolicy*>(inner_.get());
        }
    };
}}

#endif
