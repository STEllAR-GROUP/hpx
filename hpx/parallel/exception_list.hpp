//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXCEPTION_LIST_JUN_25_2014_1055PM)
#define HPX_PARALLEL_EXCEPTION_LIST_JUN_25_2014_1055PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/future.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // forward declarations, see execution_policy.hpp
    struct sequential_task_execution_policy;
    template <typename Executor> struct sequential_task_execution_policy_shim;

    struct parallel_task_execution_policy;
    template <typename Executor> struct parallel_task_execution_policy_shim;

    struct parallel_vector_execution_policy;

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename Result = void>
        struct handle_exception
        {
            typedef void type;

            HPX_ATTRIBUTE_NORETURN static void call()
            {
                try {
                    throw; //-V667
                }
                catch(std::bad_alloc const& e) {
                    boost::throw_exception(e);
                }
                catch (...) {
                    boost::throw_exception(
                        hpx::exception_list(boost::current_exception())
                    );
                }
            }
        };

        template <typename Result>
        struct handle_exception<sequential_task_execution_policy, Result>
        {
            typedef future<Result> type;

            static future<Result> call()
            {
                try {
                    try {
                        throw; //-V667
                    }
                    catch(std::bad_alloc const& e) {
                        boost::throw_exception(e);
                    }
                    catch (...) {
                        boost::throw_exception(
                            hpx::exception_list(boost::current_exception())
                        );
                    }
                }
                catch (...) {
                    return hpx::make_exceptional_future<Result>(
                        boost::current_exception());
                }
            }
        };

        template <typename Executor, typename Result>
        struct handle_exception<
                sequential_task_execution_policy_shim<Executor>, Result>
          : handle_exception<sequential_task_execution_policy, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception<parallel_task_execution_policy, Result>
        {
            typedef future<Result> type;

            static future<Result> call()
            {
                try {
                    try {
                        throw; //-V667
                    }
                    catch(std::bad_alloc const& e) {
                        boost::throw_exception(e);
                    }
                    catch (...) {
                        boost::throw_exception(
                            hpx::exception_list(boost::current_exception())
                        );
                    }
                }
                catch (...) {
                    return hpx::make_exceptional_future<Result>(
                        boost::current_exception());
                }
            }
        };

        template <typename Executor, typename Result>
        struct handle_exception<
                parallel_task_execution_policy_shim<Executor>, Result>
          : handle_exception<parallel_task_execution_policy, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception<parallel_vector_execution_policy, Result>
        {
            typedef void type;

            HPX_ATTRIBUTE_NORETURN static void call()
            {
                // any exceptions thrown by algorithms executed with the
                // parallel_vector_execution_policy are to call terminate.
                hpx::terminate();
            }
        };
        /// \endcond
    }

    // we're just reusing our existing implementation

    using hpx::exception_list;
}}}

#endif
