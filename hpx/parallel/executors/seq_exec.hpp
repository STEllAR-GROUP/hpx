//  Copyright (c) 2017 Zahra Khatami and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/seq_exec.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_SEQ_EXEC_FEB_2017)
#define HPX_PARALLEL_EXECUTOR_SEQ_EXEC_FEB_2017

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/unwrapped.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// This class wraps parallel executor and applys it as a sequantial executor
    /// This method is used while using par_if execution policy for
    /// implementing learning method
    template<typename Exec>
    struct seq_exec : executor_tag
    {
#if !defined(DOXYGEN)
        /// Create a new seq_exec
        HPX_CONSTEXPR seq_exec(Exec& e) : executor_(e) {}
#endif

        /// \cond NOINTERNAL
        typedef sequential_execution_tag execution_category;

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            executor_.execute(std::forward<F>(f), 
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<typename hpx::util::result_of<F&&(Ts&&...)>::type>
        async_execute(F && f, Ts &&... ts)
        {
            return executor_.async_execute(std::forward<F>(f), 
                std::forward<Ts>(ts)...);
        }

        
        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, Shape, Ts...>::type
        > >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {            
            return executor_.bulk_async_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
        }
        
        template <typename F, typename Shape, typename ... Ts>
        typename detail::bulk_execute_result<F, Shape, Ts...>::type
        bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            return executor_.bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
        }        

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
        }
        /// \endcond

        Exec executor_;
    };
}}}

#endif
