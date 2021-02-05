//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)

#include <hpx/algorithms/traits/pointer_category.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>

#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/detail/iterator.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace traits {
    // Allow for matching of iterator<T const> to iterator<T> while calculating
    // pointer category.
    template <typename T>
    struct remove_const_iterator_value_type<
        compute::detail::iterator<T const, cuda::experimental::allocator<T>>>
    {
        typedef compute::detail::iterator<T, cuda::experimental::allocator<T>>
            type;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct cuda_pointer_tag : general_pointer_tag
    {
    };

    struct cuda_copyable_pointer_tag : cuda_pointer_tag
    {
    };
    struct cuda_copyable_pointer_tag_to_host : cuda_pointer_tag
    {
    };
    struct cuda_copyable_pointer_tag_to_device : cuda_pointer_tag
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct pointer_category<
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        typename std::enable_if<!std::is_trivially_copyable<
            typename std::decay<T>::type>::value>::type>
    {
        typedef cuda_copyable_pointer_tag type;
    };

    template <typename Source, typename T>
    struct pointer_category<Source,
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        typename std::enable_if<!std::is_trivially_copyable<typename hpx::util::
                                        decay<T>::type>::value &&
            !std::is_same<Source,
                compute::detail::iterator<T,
                    cuda::experimental::allocator<T>>>::value>::type>
    {
        // FIXME: turn into proper pointer category
        static_assert(
            std::is_same<T,
                typename std::iterator_traits<Source>::value_type>::value,
            "The value types of the iterators must match");

        typedef cuda_copyable_pointer_tag_to_device type;
    };

    template <typename T, typename U, typename Dest>
    struct pointer_category<
        compute::detail::iterator<T, cuda::experimental::allocator<U>>, Dest,
        typename std::enable_if<!std::is_trivially_copyable<typename hpx::util::
                                        decay<T>::type>::value &&
            !std::is_same<Dest,
                compute::detail::iterator<T,
                    cuda::experimental::allocator<U>>>::value>::type>
    {
        // FIXME: turn into proper pointer category
        static_assert(
            std::is_same<typename std::decay<T>::type,
                typename std::iterator_traits<Dest>::value_type>::value,
            "The value types of the iterators must match");

        typedef cuda_copyable_pointer_tag_to_host type;
    };

    struct trivially_cuda_copyable_pointer_tag : cuda_copyable_pointer_tag
    {
    };
    struct trivially_cuda_copyable_pointer_tag_to_host
      : cuda_copyable_pointer_tag_to_host
    {
    };
    struct trivially_cuda_copyable_pointer_tag_to_device
      : cuda_copyable_pointer_tag_to_device
    {
    };

    template <typename T>
    struct pointer_category<
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        typename std::enable_if<std::is_trivially_copyable<
            typename std::decay<T>::type>::value>::type>
    {
        typedef trivially_cuda_copyable_pointer_tag type;
    };

    template <typename Source, typename T>
    struct pointer_category<Source,
        compute::detail::iterator<T, cuda::experimental::allocator<T>>,
        typename std::enable_if<std::is_trivially_copyable<typename hpx::util::
                                        decay<T>::type>::value &&
            !std::is_same<Source,
                compute::detail::iterator<T,
                    cuda::experimental::allocator<T>>>::value>::type>
    {
        // FIXME: turn into proper pointer category
        static_assert(
            std::is_same<T,
                typename std::iterator_traits<Source>::value_type>::value,
            "The value types of the iterators must match");

        typedef trivially_cuda_copyable_pointer_tag_to_device type;
    };

    template <typename T, typename U, typename Dest>
    struct pointer_category<
        compute::detail::iterator<T, cuda::experimental::allocator<U>>, Dest,
        typename std::enable_if<std::is_trivially_copyable<typename hpx::util::
                                        decay<T>::type>::value &&
            !std::is_same<Dest,
                compute::detail::iterator<T,
                    cuda::experimental::allocator<U>>>::value>::type>
    {
        // FIXME: turn into proper pointer category
        static_assert(
            std::is_same<typename std::decay<T>::type,
                typename std::iterator_traits<Dest>::value_type>::value,
            "The value types of the iterators must match");

        typedef trivially_cuda_copyable_pointer_tag_to_host type;
    };
}}    // namespace hpx::traits

namespace hpx { namespace parallel { namespace util { namespace detail {
    template <typename Dummy>
    struct copy_helper<hpx::traits::trivially_cuda_copyable_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_helper<hpx::traits::general_pointer_tag>::call(
                first, last, dest);
#else
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync(&(*dest), &(*first), bytes,
                cudaMemcpyDeviceToDevice,
                dest.target().native_handle().get_stream());

            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{last, dest};
#endif
        }
    };

    template <typename Dummy>
    struct copy_helper<hpx::traits::trivially_cuda_copyable_pointer_tag_to_host,
        Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_helper<hpx::traits::general_pointer_tag>::call(
                first, last, dest);
#else
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync(&(*dest), (*first).device_ptr(), bytes,
                cudaMemcpyDeviceToHost,
                first.target().native_handle().get_stream());

            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{last, dest};
#endif
        }
    };

    template <typename Dummy>
    struct copy_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_helper<hpx::traits::general_pointer_tag>::call(
                first, last, dest);
#else
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                cudaMemcpyHostToDevice,
                dest.target().native_handle().get_stream());

            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{last, dest};
#endif
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Dummy>
    struct copy_n_helper<hpx::traits::trivially_cuda_copyable_pointer_tag,
        Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_n_helper<hpx::traits::general_pointer_tag>::call(
                first, count, dest);
#else
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync((*dest).device_ptr(), (*first).device_ptr(), bytes,
                cudaMemcpyDeviceToDevice,
                dest.target().native_handle().get_stream());

            std::advance(first, count);
            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{first, dest};
#endif
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_n_helper<hpx::traits::general_pointer_tag>::call(
                first, count, dest);
#else
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync(&(*dest), (*first).device_ptr(), bytes,
                cudaMemcpyDeviceToHost,
                first.target().native_handle().get_stream());

            std::advance(first, count);
            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{first, dest};
#endif
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE
            HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            return copy_n_helper<hpx::traits::general_pointer_tag>::call(
                first, count, dest);
#else
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

            cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                cudaMemcpyHostToDevice,
                dest.target().native_handle().get_stream());

            std::advance(first, count);
            std::advance(dest, count);
            return util::in_out_result<InIter, OutIter>{first, dest};
#endif
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for copy-synchronize operations
    template <typename Dummy>
    struct copy_synchronize_helper<hpx::traits::cuda_copyable_pointer_tag,
        Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const& first, OutIter const&)
        {
            first.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const& first, OutIter const&)
        {
            first.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };
}}}}    // namespace hpx::parallel::util::detail

#endif
