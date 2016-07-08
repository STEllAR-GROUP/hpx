//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARALLEL_UTIL_COMPUTE_CUDA_TRANSFER_HPP
#define HPX_PARALLEL_UTIL_COMPUTE_CUDA_TRANSFER_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)

#include <hpx/parallel/util/transfer.hpp>
#include <hpx/traits/pointer_category.hpp>

#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/detail/iterator.hpp>

namespace hpx { namespace traits
{
    struct cuda_pointer_tag : general_pointer_tag{};

    struct cuda_copyable_pointer_tag : cuda_pointer_tag{};
    struct cuda_copyable_pointer_tag_to_host : cuda_pointer_tag {};
    struct cuda_copyable_pointer_tag_to_device : cuda_pointer_tag {};

    ///////////////////////////////////////////////////////////////////////
    template <typename T>
    struct pointer_category<
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
#if defined(HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE)
        typename std::enable_if<
           !std::is_trivially_copyable<T>::value

        >::type
#else
        void
#endif
    >
    {
        typedef cuda_copyable_pointer_tag type;
    };


    template <typename Source, typename T>
    struct pointer_category<
        Source,
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        typename std::enable_if<
#if defined(HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE)
           !std::is_trivially_copyable<T>::value &&
#endif
           !std::is_same<Source, compute::detail::iterator<T, compute::cuda::allocator<T> > >::value
        >::type
    >
    {
        // FIXME: turn into proper pointer category
        static_assert(std::is_same<
                T, typename std::iterator_traits<Source>::value_type
            >::value, "The value types of the iterators must match");

        typedef cuda_copyable_pointer_tag_to_device type;
    };

    template <typename T, typename Dest>
    struct pointer_category<
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        Dest,
        typename std::enable_if<
#if defined(HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE)
           !std::is_trivially_copyable<T>::value &&
#endif
           !std::is_same<Dest, compute::detail::iterator<T, compute::cuda::allocator<T> > >::value
        >::type
    >
    {
        // FIXME: turn into proper pointer category
        static_assert(std::is_same<
                T, typename std::iterator_traits<Dest>::value_type
            >::value, "The value types of the iterators must match");

        typedef cuda_copyable_pointer_tag_to_host type;
    };

#if defined(HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE)
    struct trivially_cuda_copyable_pointer_tag : cuda_copyable_pointer_tag{};
    struct trivially_cuda_copyable_pointer_tag_to_host : cuda_copyable_pointer_tag_to_host {};
    struct trivially_cuda_copyable_pointer_tag_to_device : cuda_copyable_pointer_tag_to_device {};

    template <typename T>
    struct pointer_category<
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        typename std::enable_if<
           std::is_trivially_copyable<T>::value
        >::type
    >
    {
        typedef trivially_cuda_copyable_pointer_tag type;
    };

    template <typename Source, typename T>
    struct pointer_category<
        Source,
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        typename std::enable_if<
           std::is_trivially_copyable<T>::value &&
           !std::is_same<Source, compute::detail::iterator<T, compute::cuda::allocator<T> > >::value
        >::type
    >
    {
        // FIXME: turn into proper pointer category
        static_assert(std::is_same<
                T, typename std::iterator_traits<Source>::value_type
            >::value, "The value types of the iterators must match");

        typedef trivially_cuda_copyable_pointer_tag_to_device type;
    };

    template <typename T, typename Dest>
    struct pointer_category<
        compute::detail::iterator<T, compute::cuda::allocator<T> >,
        Dest,
        typename std::enable_if<
           std::is_trivially_copyable<T>::value &&
           !std::is_same<Dest, compute::detail::iterator<T, compute::cuda::allocator<T> > >::value
        >::type
    >
    {
        // FIXME: turn into proper pointer category
        static_assert(std::is_same<
                T, typename std::iterator_traits<Dest>::value_type
            >::value, "The value types of the iterators must match");

        typedef trivially_cuda_copyable_pointer_tag_to_host type;
    };
#endif
}}

namespace hpx { namespace parallel { namespace util { namespace detail
{
    template <typename Dummy>
    struct copy_helper<hpx::traits::trivially_cuda_copyable_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
            cudaMemcpyAsync((*dest).device_ptr(), (*first).device_ptr(),
                bytes, cudaMemcpyDeviceToDevice);
// FIXME: make get_stream() available on the device
//                 dest.target().native_handle().get_stream());
#else
            cudaMemcpyAsync(&(*dest), &(*first), bytes,
                cudaMemcpyDeviceToDevice,
                dest.target().native_handle().get_stream());
#endif

            std::advance(dest, count);
            return std::make_pair(last, dest);
        }
    };

    template <typename Dummy>
    struct copy_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
            cudaMemcpyAsync(&(*dest), &(*first), bytes,
                cudaMemcpyDeviceToHost);
// FIXME: make get_stream() available on the device
//                 first.target().native_handle().get_stream());
#else
            cudaMemcpyAsync(&(*dest), (*first).device_ptr(),
                bytes, cudaMemcpyDeviceToHost,
                first.target().native_handle().get_stream());
#endif

            std::advance(dest, count);
            return std::make_pair(last, dest);
        }
    };

    template <typename Dummy>
    struct copy_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static std::pair<InIter, OutIter>
        call(InIter first, InIter last, OutIter dest)
        {
            std::size_t count = std::distance(first, last);
            std::size_t bytes = count *
                sizeof(typename std::iterator_traits<InIter>::value_type);

#if defined(__CUDA_ARCH__)
            cudaMemcpyAsync(&(*dest), &(*first), bytes,
                cudaMemcpyHostToDevice);
// FIXME: make get_stream() available on the device
//                 dest.target().native_handle().get_stream());
#else
            cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                cudaMemcpyHostToDevice,
                dest.target().native_handle().get_stream());
#endif

            std::advance(dest, count);
            return std::make_pair(last, dest);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag, Dummy>
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
                cudaMemcpyDeviceToDevice);
// FIXME: make get_stream() available on the device
//                 dest.target().native_handle().get_stream());
#else
            cudaMemcpyAsync((*dest).device_ptr(), (*first).device_ptr(),
                bytes, cudaMemcpyDeviceToDevice,
                dest.target().native_handle().get_stream());
#endif

            std::advance(first, count);
            std::advance(dest, count);
            return std::make_pair(first, dest);
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_host, Dummy>
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
                cudaMemcpyDeviceToHost);
// FIXME: make get_stream() available on the device
//                 first.target().native_handle().get_stream());
#else
            cudaMemcpyAsync(&(*dest), (*first).device_ptr(), bytes,
                cudaMemcpyDeviceToHost,
                first.target().native_handle().get_stream());
#endif

            std::advance(first, count);
            std::advance(dest, count);
            return std::make_pair(first, dest);
        }
    };

    template <typename Dummy>
    struct copy_n_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
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
                cudaMemcpyHostToDevice);
// FIXME: make get_stream() available on the device
//                 dest.target().native_handle().get_stream());
#else
            cudaMemcpyAsync((*dest).device_ptr(), &(*first), bytes,
                cudaMemcpyHostToDevice,
                dest.target().native_handle().get_stream());
#endif

            std::advance(first, count);
            std::advance(dest, count);
            return std::make_pair(first, dest);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    // Customization point for copy-synchronize operations
    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::cuda_copyable_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const& first, OutIter const&)
        {
            first.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_host, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const& first, OutIter const&)
        {
            first.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };

    template <typename Dummy>
    struct copy_synchronize_helper<
        hpx::traits::trivially_cuda_copyable_pointer_tag_to_device, Dummy>
    {
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static void
        call(InIter const&, OutIter const& dest)
        {
            dest.target().synchronize();
        }
    };
}}}}

#endif
#endif
