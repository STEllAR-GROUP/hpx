//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/execution_policy.hpp

#if !defined(HPX_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM)
#define HPX_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <memory>
#include <type_traits>

#include <boost/static_assert.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    struct task_execution_policy_tag
    {
        task_execution_policy_tag() {}
    };
    /// \endcond

    /// Rebind the type of executor used by an execution policy. The execution
    /// category of Executor shall not be weaker than that of ExecutionPolicy.
    template <typename ExecutionPolicy, typename Executor>
    struct rebind_executor
    {
        /// \cond NOINTERNAL
        typedef typename ExecutionPolicy::execution_category category1;
        typedef typename executor_traits<Executor>::execution_category category2;

        BOOST_STATIC_ASSERT(
            (parallel::v3::detail::is_not_weaker<category2, category1>::value)
        );
        /// \endcond

        /// The type of the rebound execution policy
        typedef typename ExecutionPolicy::template rebind<Executor>::type type;
    };

    /// The execution policy tag \a task can be used to create a execution
    /// policy which forces the given algorithm to be executed in an
    /// asynchronous way.
    static task_execution_policy_tag const task;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor> struct sequential_execution_policy_shim;
    template <typename Executor> struct sequential_task_execution_policy_shim;

    /// Extension: The class sequential_task_execution_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may not
    /// be parallelized (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequential_execution_policy.
    struct sequential_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::sequential_executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::sequential_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequential_task_execution_policy_shim<Executor_> type;
        };

        /// \cond NOINTERNAL
        static std::size_t get_chunk_size() { return 0; }

        sequential_task_execution_policy() {}
        /// \endcond

        /// Create a new sequential_task_execution_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        sequential_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new sequential_task_execution_policy from the given
        /// executor
        ///
        /// \tparam Executor    The type of the executor to associate with this
        ///                     execution policy.
        ///
        /// \param exec         [in] The executor to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: is_executor<Executor>::value is true
        ///
        /// \returns The new parallel_task_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            sequential_task_execution_policy, Executor
        >::type
        on(Executor& exec) const
        {
            BOOST_STATIC_ASSERT(is_executor<Executor>::value);

            typedef typename rebind_executor<
                sequential_task_execution_policy, Executor
            >::type rebound_type;
            return rebound_type(exec);
        }

        /// Return the associated executor object.
        static executor_type& executor()
        {
            static parallel::sequential_executor exec;
            return exec;
        }
    };

    /// Default sequential task execution policy object.
    static sequential_task_execution_policy const seq_task;

    /// Extension: The class sequential_task_execution_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a sequential_task_execution_policy and an executor and indicate that
    /// a parallel algorithm's execution may not be parallelized  (has to run
    /// sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequential_execution_policy.
    template <typename Executor>
    struct sequential_task_execution_policy_shim
      : sequential_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequential_task_execution_policy_shim<Executor_> type;
        };

        /// \cond NOINTERNAL
        static std::size_t get_chunk_size() { return 0; }
        /// \endcond

        /// Create a new sequential_task_execution_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        sequential_task_execution_policy_shim operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

    private:
        /// \cond NOINTERNAL
        friend struct sequential_task_execution_policy;
        template <typename> friend struct sequential_execution_policy_shim;

        sequential_task_execution_policy_shim(Executor& exec)
          : exec_(exec)
        {}

        Executor& exec_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class sequential_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct sequential_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::sequential_executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::sequential_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequential_execution_policy_shim<Executor_> type;
        };

        /// \cond NOINTERNAL
        sequential_execution_policy() {}

        static std::size_t get_chunk_size() { return 0; }
        /// \endcond

        /// Create a new sequential_task_execution_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        sequential_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return seq_task;
        }

        /// Create a new sequential_execution_policy_shim from the given
        /// executor
        ///
        /// \tparam Executor    The type of the executor to associate with this
        ///                     execution policy.
        ///
        /// \param exec         [in] The executor to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: is_executor<Executor>::value is true
        ///
        /// \returns The new parallel_task_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<sequential_execution_policy, Executor>::type
        on(Executor& exec) const
        {
            BOOST_STATIC_ASSERT(is_executor<Executor>::value);

            typedef typename rebind_executor<
                sequential_execution_policy, Executor
            >::type rebound_type;
            return rebound_type(exec);
        }

        /// Return the associated executor object.
        static executor_type& executor()
        {
            static parallel::sequential_executor exec;
            return exec;
        }
    };

    /// Default sequential execution policy object.
    static sequential_execution_policy const seq;

    /// The class sequential_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor>
    struct sequential_execution_policy_shim : sequential_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequential_execution_policy_shim<Executor_> type;
        };

        /// Create a new sequential_task_execution_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy_shim
        ///
        sequential_task_execution_policy_shim<Executor> operator()(
            task_execution_policy_tag tag) const
        {
            return sequential_task_execution_policy_shim<Executor>(exec_);
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

    private:
        /// \cond NOINTERNAL
        friend struct sequential_execution_policy;

        sequential_execution_policy_shim(Executor& exec)
          : exec_(exec)
        {}

        Executor& exec_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor> struct parallel_execution_policy_shim;
    template <typename Executor> struct parallel_task_execution_policy_shim;

    /// Extension: The class parallel_task_execution_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may be
    /// parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_execution_policy.
    struct parallel_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::parallel_executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::parallel_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_task_execution_policy_shim<Executor_> type;
        };

        /// \cond NOINTERNAL
        parallel_task_execution_policy() {}
        /// \endcond

        /// Create a new parallel_task_execution_policy from given executor
        ///
        /// \tparam Executor    The type of the executor to associate with this
        ///                     execution policy.
        ///
        /// \param exec         [in] The executor to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: is_executor<Executor>::value is true
        ///
        /// \returns The new parallel_task_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            parallel_task_execution_policy, Executor
        >::type
        on(Executor& exec) const
        {
            BOOST_STATIC_ASSERT(is_executor<Executor>::value);

            typedef typename rebind_executor<
                parallel_task_execution_policy, Executor
            >::type rebound_type;
            return rebound_type(exec, chunk_size_);
        }

        /// Create a new parallel_task_execution_policy referencing a chunk size.
        ///
        /// \param chunk_size   [in] The chunk size controlling the number of
        ///                     iterations scheduled to be executed on the same
        ///                     HPX thread
        ///
        /// \returns The new parallel_task_execution_policy
        ///
        parallel_task_execution_policy operator()(std::size_t chunk_size) const
        {
            return parallel_task_execution_policy(chunk_size);
        }

        /// Create a new parallel_task_execution_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_task_execution_policy
        ///
        parallel_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Return the associated executor object.
        static executor_type& executor()
        {
            static parallel::parallel_executor exec;
            return exec;
        }

        /// \cond NOINTERNAL
        std::size_t get_chunk_size() const { return chunk_size_; }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & chunk_size_;
        }

    protected:
        parallel_task_execution_policy(std::size_t chunk_size)
          : chunk_size_(chunk_size)
        {}

    private:
        std::size_t chunk_size_;
        /// \endcond
    };

    /// Default parallel task execution policy object.
    static parallel_task_execution_policy const par_task;

    /// Extension: The class parallel_task_execution_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a sequential_task_execution_policy and an executor and indicate that
    /// a parallel algorithm's execution may not be parallelized  (has to run
    /// sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequential_execution_policy.
    template <typename Executor>
    struct parallel_task_execution_policy_shim : parallel_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_task_execution_policy_shim<Executor_> type;
        };

        /// Create a new parallel_task_execution_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        parallel_task_execution_policy_shim operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

    private:
        /// \cond NOINTERNAL
        friend struct parallel_task_execution_policy;
        template <typename> friend struct parallel_execution_policy_shim;

        parallel_task_execution_policy_shim(Executor& exec,
                std::size_t chunk_size)
          : parallel_task_execution_policy(chunk_size), exec_(exec)
        {}

        Executor& exec_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct parallel_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::parallel_executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::parallel_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_execution_policy_shim<Executor_> type;
        };

        /// \cond NOINTERNAL
        parallel_execution_policy() {}
        /// \endcond

        /// Create a new parallel_execution_policy referencing an executor and
        /// a chunk size.
        ///
        /// \param exec         [in] The executor to use for the execution of
        ///                     the parallel algorithm the returned execution
        ///                     policy is used with
        ///
        /// \returns The new parallel_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<parallel_execution_policy, Executor>::type
        on(Executor& exec) const
        {
            BOOST_STATIC_ASSERT(is_executor<Executor>::value);

            typedef typename rebind_executor<
                parallel_execution_policy, Executor
            >::type rebound_type;
            return rebound_type(exec, this->get_chunk_size());
        }

        /// Create a new parallel_execution_policy referencing a chunk size.
        ///
        /// \param chunk_size   [in] The chunk size controlling the number of
        ///                     iterations scheduled to be executed on the same
        ///                     HPX thread
        ///
        /// \returns The new parallel_execution_policy
        ///
        parallel_execution_policy operator()(std::size_t chunk_size) const
        {
            return parallel_execution_policy(chunk_size);
        }

        /// Create a new parallel_execution_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        /// \param chunk_size   [in] The chunk size controlling the number of
        ///                     iterations scheduled to be executed on the same
        ///                     HPX thread
        ///
        /// \returns The new parallel_execution_policy
        ///
        parallel_task_execution_policy operator()(task_execution_policy_tag tag,
            std::size_t chunk_size) const
        {
            return par_task(chunk_size);
        }

        /// Create a new parallel_execution_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_execution_policy
        ///
        parallel_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return par_task(chunk_size_);
        }

        /// \cond NOINTERNAL
        std::size_t get_chunk_size() const { return chunk_size_; }
        /// \endcond

        /// Return the associated executor object.
        static executor_type& executor()
        {
            static parallel::parallel_executor exec;
            return exec;
        }

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & chunk_size_;
        }

    protected:
        parallel_execution_policy(std::size_t chunk_size)
          : chunk_size_(chunk_size)
        {}

    private:
        std::size_t chunk_size_;
        /// \endcond
    };

    /// Default parallel execution policy object.
    static parallel_execution_policy const par;

    /// The class parallel_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor>
    struct parallel_execution_policy_shim : parallel_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_execution_policy_shim<Executor_> type;
        };

        /// Create a new parallel_execution_policy_shim referencing a chunk size.
        ///
        /// \param chunk_size   [in] The chunk size controlling the number of
        ///                     iterations scheduled to be executed on the same
        ///                     HPX thread
        ///
        /// \returns The new parallel_execution_policy_shim
        ///
        parallel_execution_policy_shim operator()(std::size_t chunk_size) const
        {
            return parallel_execution_policy_shim(exec_, chunk_size);
        }

        /// Create a new parallel_execution_policy_shim referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        /// \param chunk_size   [in] The chunk size controlling the number of
        ///                     iterations scheduled to be executed on the same
        ///                     HPX thread
        ///
        /// \returns The new parallel_execution_policy_shim
        ///
        parallel_task_execution_policy_shim<Executor>
        operator()(task_execution_policy_tag tag, std::size_t chunk_size) const
        {
            return parallel_task_execution_policy_shim<Executor>(
                exec_, chunk_size);
        }

        /// Create a new parallel_execution_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_execution_policy
        ///
        parallel_task_execution_policy_shim<Executor>
        operator()(task_execution_policy_tag tag) const
        {
            return parallel_task_execution_policy_shim<Executor>(
                exec_, this->get_chunk_size());
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

    private:
        /// \cond NOINTERNAL
        friend struct parallel_execution_policy;

        parallel_execution_policy_shim(Executor& exec, std::size_t chunk_size)
          : parallel_execution_policy(chunk_size), exec_(exec)
        {}

        Executor& exec_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_vector_execution_policy is an execution policy type used as
    /// a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be vectorized.
    struct parallel_vector_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::parallel_executor executor_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::parallel_execution_tag execution_category;

        /// \cond NOINTERNAL
        parallel_vector_execution_policy() {}

        static std::size_t get_chunk_size() { return 0; }
        /// \endcond

        /// Create a new parallel_vector_execution_policy from itself
        ///
        /// \param tag [in] Specify that the corresponding asynchronous
        ///            execution policy should be used
        ///
        /// \returns The new parallel_vector_execution_policy
        ///
        parallel_vector_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Return the associated executor object.
        static executor_type& executor()
        {
            static parallel::parallel_executor exec;
            return exec;
        }
    };

    /// Default vector execution policy object.
    static parallel_vector_execution_policy const par_vec;

    ///////////////////////////////////////////////////////////////////////////
    class execution_policy;

    namespace detail
    {
        /// \cond NOINTERNAL
        struct execution_policy_base
        {
            virtual ~execution_policy_base() {}

            virtual std::type_info const& type() const BOOST_NOEXCEPT  = 0;

            virtual execution_policy make_async(
                task_execution_policy_tag tag) const = 0;
            virtual BOOST_SCOPED_ENUM(launch) launch_policy() const = 0;

            virtual void* get() BOOST_NOEXCEPT = 0;
            virtual void const* get() const BOOST_NOEXCEPT = 0;
        };

        template <typename ExPolicy>
        struct execution_policy_shim : execution_policy_base
        {
            execution_policy_shim(ExPolicy const& policy)
              : policy_(policy)
            {}
            execution_policy_shim(ExPolicy && policy)
              : policy_(std::move(policy))
            {}

            std::type_info const& type() const BOOST_NOEXCEPT
            {
                return typeid(ExPolicy);
            }

            // defined below
            execution_policy make_async(task_execution_policy_tag tag) const;
            BOOST_SCOPED_ENUM(launch) launch_policy() const;

            void* get() BOOST_NOEXCEPT
            {
                return &policy_;
            }

            void const* get() const BOOST_NOEXCEPT
            {
                return &policy_;
            }

        private:
            ExPolicy policy_;
        };
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_execution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_execution_policy<parallel_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_execution_policy<parallel_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};

        template <>
        struct is_execution_policy<parallel_vector_execution_policy>
          : boost::mpl::true_
        {};

        template <>
        struct is_execution_policy<sequential_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_execution_policy<sequential_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};

        // extension
        template <>
        struct is_execution_policy<sequential_task_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_execution_policy<
                sequential_task_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};

        template <>
        struct is_execution_policy<parallel_task_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_execution_policy<
                parallel_task_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};

        template <>
        struct is_execution_policy<execution_policy>
          : boost::mpl::true_
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// 1. The type is_execution_policy can be used to detect execution
    ///    policies for the purpose of excluding function signatures
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
      : detail::is_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_parallel_execution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_parallel_execution_policy<parallel_execution_policy>
          : boost::mpl::true_
        {};

        template <>
        struct is_parallel_execution_policy<parallel_vector_execution_policy>
          : boost::mpl::true_
        {};

        template <>
        struct is_parallel_execution_policy<parallel_task_execution_policy>
          : boost::mpl::true_
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy enables parallelization
    ///
    /// 1. The type is_parallel_execution_policy can be used to detect parallel
    ///    execution policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_parallel_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_parallel_execution_policy is undefined.
    ///
    template <typename T>
    struct is_parallel_execution_policy
      : detail::is_parallel_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_sequential_execution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_sequential_execution_policy<sequential_task_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_sequential_execution_policy<
                sequential_task_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};

        template <>
        struct is_sequential_execution_policy<sequential_execution_policy>
          : boost::mpl::true_
        {};

        template <typename Executor>
        struct is_sequential_execution_policy<
                sequential_execution_policy_shim<Executor> >
          : boost::mpl::true_
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy does not enable
    ///            parallelization
    ///
    /// 1. The type is_sequential_execution_policy can be used to detect
    ///    non-parallel execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_sequential_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_sequential_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_sequential_execution_policy
      : detail::is_sequential_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_async_execution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_async_execution_policy<sequential_task_execution_policy>
          : boost::mpl::true_
        {};

        template <>
        struct is_async_execution_policy<parallel_task_execution_policy>
          : boost::mpl::true_
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy makes algorithms
    ///            asynchronous
    ///
    /// 1. The type is_async_execution_policy can be used to detect
    ///    asynchronous execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_async_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_async_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_async_execution_policy
      : detail::is_async_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    ///
    /// An execution policy is an object that expresses the requirements on the
    /// ordering of functions invoked as a consequence of the invocation of a
    /// standard algorithm. Execution policies afford standard algorithms the
    /// discretion to execute in parallel.
    ///
    /// 1. The class execution_policy is a dynamic container for execution
    ///    policy objects. execution_policy allows dynamic control over
    ///    standard algorithm execution.
    /// 2. Objects of type execution_policy shall be constructible and
    ///    assignable from objects of type T for which
    ///    is_execution_policy<T>::value is true.
    ///
    class execution_policy
    {
    private:
        boost::shared_ptr<detail::execution_policy_base> inner_;

    public:
        /// Effects: Constructs an execution_policy object with a copy of
        ///          exec's state
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        execution_policy(ExPolicy const& policy,
                typename std::enable_if<
                    is_execution_policy<ExPolicy>::value, ExPolicy
                >::type* = 0)
          : inner_(boost::make_shared<
                    detail::execution_policy_shim<ExPolicy>
                >(policy))
        {}

        /// Move constructs a new execution_policy object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy(execution_policy && policy)
          : inner_(std::move(policy.inner_))
        {}

        execution_policy(execution_policy const& rhs)
          : inner_(rhs.inner_)
        {}

        /// Extension: Create a new execution_policy holding the current policy
        /// made asynchronous.
        ///
        /// \param tag  [in] Specify that the corresponding asynchronous
        ///             execution policy should be used
        ///
        /// \returns The new execution_policy
        ///
        execution_policy operator()(task_execution_policy_tag tag) const
        {
            return inner_->make_async(tag);
        }

        /// Extension: Retrieve default launch policy for this execution policy.
        ///
        /// \returns The associated default launch policy
        ///
        BOOST_SCOPED_ENUM(launch) launch_policy() const
        {
            return inner_->launch_policy();
        }

        /// Effects: Assigns a copy of exec's state to *this
        /// Returns: *this
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        typename std::enable_if<
            is_execution_policy<ExPolicy>::value, execution_policy
        >::type&
        operator=(ExPolicy const& policy)
        {
            if (this != &policy)
            {
                inner_ = boost::make_shared<
                        detail::execution_policy_shim<ExPolicy>
                    >(policy);
            }
            return *this;
        }

        /// Move assigns a new execution policy to the object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy& operator=(execution_policy && policy)
        {
            if (this != &policy)
                inner_ = std::move(policy.inner_);
            return *this;
        }

        /// Returns: typeid(T), such that T is the type of the execution policy
        ///          object contained by *this
        std::type_info const& type() const BOOST_NOEXCEPT
        {
            return inner_->type();
        }

        /// Returns: If target_type() == typeid(T), a pointer to the stored
        ///          execution policy object; otherwise a null pointer
        /// Requires: is_execution_policy<T>::value is true
        template <typename ExPolicy>
        ExPolicy* get() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                !(boost::is_same<ExPolicy, execution_policy>::value),
                "Incorrect execution policy parameter.");
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (inner_->type() != typeid(ExPolicy))
                return 0;

            return static_cast<ExPolicy*>(inner_->get());
        }

        /// Returns: If target_type() == typeid(T), a pointer to the stored
        ///          execution policy object; otherwise a null pointer
        /// Requires: is_execution_policy<T>::value is true
        template <typename ExPolicy>
        ExPolicy const* get() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                !(boost::is_same<ExPolicy, execution_policy>::value),
                "Incorrect execution policy parameter.");
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (inner_->type() != typeid(ExPolicy))
                return 0;

            return static_cast<ExPolicy const*>(inner_->get());
        }
    };

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy>
        execution_policy execution_policy_shim<ExPolicy>::make_async(
            task_execution_policy_tag tag) const
        {
            return policy_(tag);
        }

        template <typename ExPolicy, typename Enable = void>
        struct extract_launch_policy
        {
            static BOOST_SCOPED_ENUM(launch) call() { return launch::async; }
        };

        template <typename ExPolicy>
        struct extract_launch_policy<ExPolicy,
            typename std::enable_if<
                is_sequential_execution_policy<ExPolicy>::value
            >::type>
        {
            static BOOST_SCOPED_ENUM(launch) call() { return launch::deferred; }
        };

        template <typename ExPolicy>
        BOOST_SCOPED_ENUM(launch)
        execution_policy_shim<ExPolicy>::launch_policy() const
        {
            return extract_launch_policy<ExPolicy>::call();
        }
        /// \endcond
    }
}}}

#endif
