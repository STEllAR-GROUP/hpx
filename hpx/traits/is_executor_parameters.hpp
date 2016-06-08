//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_PARAMETERS_AUG_01_2015_0709AM)
#define HPX_TRAITS_IS_EXECUTOR_PARAMETERS_AUG_01_2015_0709AM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>
#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
#include <functional>
#endif

#include <boost/ref.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct executor_parameters_tag {};
    /// Parameters implement functions:
    /// bool variable_chunk_size(Executor &)
    /// std::size_t get_chunk_size(Executor &, F &&, std::size_t num_tasks)
    struct executor_parameters_chunk_size_tag : executor_parameters_tag {};

    /// Parameters implement functions:
    /// void mark_begin_execution();
    /// void mark_end_execution();
    struct executor_parameters_mark_begin_end_tag : executor_parameters_tag {};

    /// Parameters implement function std::size_t processing_units_count();
    struct executor_parameters_processing_units_count_tag : executor_parameters_tag {};

    /// Parameters implement function: void reset_thread_distribution(Executor &)
    struct executor_parameters_reset_thread_distr_tag : executor_parameters_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T, typename Tag>
        struct is_executor_parameters
          : std::is_base_of<Tag, T>
        {};

        template <typename Tag>
        struct is_executor_parameters<Tag, Tag>
          : std::false_type
        {};

        template <typename T, typename Tag>
        struct is_executor_parameters< ::boost::reference_wrapper<T>, Tag >
          : is_executor_parameters<typename hpx::util::decay<T>::type, Tag>
        {};

#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
        template <typename T, typename Tag>
        struct is_executor_parameters< ::std::reference_wrapper<T>, Tag >
          : is_executor_parameters<typename hpx::util::decay<T>::type, Tag>
        {};
#endif
        /// \endcond
    }

    template <typename T>
    struct is_executor_parameters
      : detail::is_executor_parameters<
            typename hpx::util::decay<T>::type, executor_parameters_tag
        >
    {};

    template <typename T>
    struct is_executor_parameters_chunk_size
      : detail::is_executor_parameters<
            typename hpx::util::decay<T>::type, executor_parameters_chunk_size_tag
        >
    {};

    template <typename T>
    struct is_executor_parameters_mark_begin_end
      : detail::is_executor_parameters<
            typename hpx::util::decay<T>::type, executor_parameters_mark_begin_end_tag
        >
    {};

    template <typename T>
    struct is_executor_parameters_reset_thread_distr
      : detail::is_executor_parameters<
            typename hpx::util::decay<T>::type, executor_parameters_reset_thread_distr_tag
        >
    {};

    template <typename T>
    struct is_executor_parameters_processing_units_count
      : detail::is_executor_parameters<
            typename hpx::util::decay<T>::type,
            executor_parameters_processing_units_count_tag
        >
    {};

    template <typename Executor, typename Enable = void>
    struct executor_parameter_traits;
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Parameters, typename Enable>
    struct is_executor_parameters
      : parallel::v3::is_executor_parameters<Parameters>
    {};

    template <typename Parameters, typename Enable>
    struct is_executor_parameters_chunk_size
      : parallel::v3::is_executor_parameters_chunk_size<Parameters>
    {};

    template <typename Parameters, typename Enable>
    struct is_executor_parameters_mark_begin_end
      : parallel::v3::is_executor_parameters_mark_begin_end<Parameters>
    {};

    template <typename Parameters, typename Enable>
    struct is_executor_parameters_reset_thread_distr
      : parallel::v3::is_executor_parameters_reset_thread_distr<Parameters>
    {};

    template <typename Parameters, typename Enable>
    struct is_executor_parameters_processing_units_count
      : parallel::v3::is_executor_parameters_processing_units_count<Parameters>
    {};
}}

#endif

