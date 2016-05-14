//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARALLEL_UTIL_COMPUTE_CUDA_TRANSFER_HPP
#define HPX_PARALLEL_UTIL_COMPUTE_CUDA_TRANSFER_HPP

#include <hpx/compute/detail/iterator.hpp>
#include <hpx/compute/cuda/allocator.hpp>

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        struct trivially_cuda_copyable_pointer_tag {};
        struct trivially_cuda_copyable_pointer_tag_to_host {};
        struct trivially_cuda_copyable_pointer_tag_to_device {};

        template <>
        struct copy_helper<trivially_cuda_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                std::size_t count = std::distance(first, last);
                std::size_t bytes = count
                    * sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync((*dest).device_ptr(), (*first).device_ptr(),
                    bytes, cudaMemcpyDeviceToDevice,
                    dest.target().native_handle().stream_);
#else
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyDeviceToDevice,
                    dest.target().native_handle().stream_);
#endif

                std::advance(dest, count);
                return std::make_pair(last, dest);
            }
        };

        template <>
        struct copy_helper<trivially_cuda_copyable_pointer_tag_to_host>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                std::size_t count = std::distance(first, last);
                std::size_t bytes = count
                    * sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyDeviceToHost,
                    first.target().native_handle().stream_);
#else
                cudaMemcpyAsync(&(*dest), (*first).device_ptr(),
                    bytes, cudaMemcpyDeviceToHost,
                    first.target().native_handle().stream_);
#endif

                std::advance(dest, count);
                return std::make_pair(last, dest);
            }
        };

        template <>
        struct copy_helper<trivially_cuda_copyable_pointer_tag_to_device>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                std::size_t count = std::distance(first, last);
                std::size_t bytes = count
                    * sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyHostToDevice,
                    dest.target().native_handle().stream_);
#else
                cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                    cudaMemcpyHostToDevice,
                    dest.target().native_handle().stream_);
#endif

                std::advance(dest, count);
                return std::make_pair(last, dest);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <>
        struct copy_n_helper<trivially_cuda_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                std::size_t bytes = count
                    * sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyDeviceToDevice,
                    dest.target().native_handle().stream_);
#else
                cudaMemcpyAsync((*dest).device_ptr(), (*first).device_ptr(),
                    bytes, cudaMemcpyDeviceToDevice,
                    dest.target().native_handle().stream_);
#endif

                std::advance(first, count);
                std::advance(dest, count);
                return std::make_pair(first, dest);
            }
        };

        template <>
        struct copy_n_helper<trivially_cuda_copyable_pointer_tag_to_host>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                std::size_t bytes = count
                    * sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyDeviceToHost,
                    first.target().native_handle().stream_);
#else
                cudaMemcpyAsync(&(*dest), (*first).device_ptr(), bytes,
                    cudaMemcpyDeviceToHost,
                    first.target().native_handle().stream_);
#endif

                std::advance(first, count);
                std::advance(dest, count);
                return std::make_pair(first, dest);
            }
        };

        template <>
        struct copy_n_helper<trivially_cuda_copyable_pointer_tag_to_device>
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                std::size_t bytes = count *
                    sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
                cudaMemcpyAsync(&(*dest), &(*first), bytes,
                    cudaMemcpyHostToDevice,
                    dest.target().native_handle().stream_);
#else
                cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                    cudaMemcpyHostToDevice,
                    dest.target().native_handle().stream_);
#endif

                std::advance(first, count);
                std::advance(dest, count);
                return std::make_pair(first, dest);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Customization point for copy-synchronize operations
        template <>
        struct copy_synchronize_helper<trivially_cuda_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static void
            call(InIter const&, OutIter const& dest)
            {
                dest.target().synchronize();
            }
        };

        template <>
        struct copy_synchronize_helper<trivially_cuda_copyable_pointer_tag_to_host>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static void
            call(InIter const& first, OutIter const&)
            {
                first.target().synchronize();
            }
        };

        template <>
        struct copy_synchronize_helper<trivially_cuda_copyable_pointer_tag_to_device>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static void
            call(InIter const&, OutIter const& dest)
            {
                dest.target().synchronize();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Source, typename T>
        inline trivially_cuda_copyable_pointer_tag
        get_pointer_category(
            compute::detail::iterator<T, compute::cuda::allocator<T> > const&,
            compute::detail::iterator<T, compute::cuda::allocator<T> > const&)
        {
            // FIXME: turn into proper pointer category
#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
            static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable");
#endif
            return trivially_cuda_copyable_pointer_tag();
        }

        template <typename Source, typename T>
        inline trivially_cuda_copyable_pointer_tag_to_device
        get_pointer_category(Source const&,
            compute::detail::iterator<T, compute::cuda::allocator<T> > const&)
        {
            // FIXME: turn into proper pointer category
            static_assert(std::is_same<
                    T, typename std::iterator_traits<Source>::value_type
                >::value, "The value types of the iterators must match");

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
            static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable");
#endif
            return trivially_cuda_copyable_pointer_tag_to_device();
        }

        template <typename T, typename Dest>
        inline trivially_cuda_copyable_pointer_tag_to_host
        get_pointer_category(
            compute::detail::iterator<T, compute::cuda::allocator<T> > const&,
            Dest const&)
        {
            // FIXME: turn into proper pointer category
            static_assert(std::is_same<
                    T, typename std::iterator_traits<Dest>::value_type
                >::value, "The value types of the iterators must match");

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
            static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable");
#endif
            return trivially_cuda_copyable_pointer_tag_to_host();
        }
    }
}}}

#endif
