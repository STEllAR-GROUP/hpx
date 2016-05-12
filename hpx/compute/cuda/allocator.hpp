//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_ALLOCATOR_HPP
#define HPX_COMPUTE_CUDA_ALLOCATOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/exception.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/target_ptr.hpp>
#include <hpx/compute/cuda/value_proxy.hpp>
#include <hpx/compute/cuda/detail/scoped_active_target.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/util/unused.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace hpx { namespace compute { namespace cuda
{
    template <typename T>
    class allocator
    {
    public:
        typedef T value_type;
        typedef target_ptr<T> pointer;
        typedef target_ptr<T const> const_pointer;
#if defined(__CUDA_ARCH__)
        typedef T& reference;
        typedef T const& const_reference;
#else
        typedef value_proxy<T> reference;
        typedef value_proxy<T const> const_reference;
#endif
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <typename U>
        struct rebind
        {
            typedef allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        typedef cuda::target target_type;

        allocator(target_type& tgt)
          : target_(tgt)
        {}

        template <typename U>
        allocator(allocator<U>& alloc)
          : target_(alloc.target_)
        {}

        // Returns the actual address of x even in presence of overloaded
        // operator&
        pointer address(reference x) const HPX_NOEXCEPT
        {
#if defined(__CUDA_ARCH__)
            return &x;
#else
            return pointer(x.device_ptr(), target_);
#endif
        }

        const_pointer address(const_reference x) const HPX_NOEXCEPT
        {
#if defined(__CUDA_ARCH__)
            return &x;
#else
            return pointer(x.device_ptr(), target_);
#endif
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage by calling
        // cudaMalloc, but it is unspecified when and how this function is
        // called. The pointer hint may be used to provide locality of
        // reference: the allocator, if supported by the implementation, will
        // attempt to allocate the new memory block as close as possible to hint.
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
        {
#if defined(__CUDA_ARCH__)
            pointer result;
#else
            value_type *p = 0;
            detail::scoped_active_target active(target_);

            cudaError_t error = cudaMalloc(&p, n*sizeof(T));

            pointer result(p, target_);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(out_of_memory,
                    "cuda::allocator<T>::allocate()",
                    std::string("cudaMalloc failed: ") +
                        cudaGetErrorString(error));
            }
#endif
            return result;
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        void deallocate(pointer p, size_type n)
        {
#if !defined(__CUDA_ARCH__)
            detail::scoped_active_target active(target_);

            cudaError_t error = cudaFree(p.device_ptr());
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::allocator<T>::deallocate()",
                    std::string("cudaFree failed: ") +
                        cudaGetErrorString(error));
            }
#endif
        }

        // Returns the maximum theoretically possible value of n, for which the
        // call allocate(n, 0) could succeed. In most implementations, this
        // returns std::numeric_limits<size_type>::max() / sizeof(value_type).
        size_type max_size() const HPX_NOEXCEPT
        {
            detail::scoped_active_target active(target_);
            std::size_t free = 0;
            std::size_t total = 0;
            cudaError_t error = cudaMemGetInfo(&free, &total);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::allocator<T>::max_size()",
                    std::string("cudaMemGetInfo failed: ") +
                        cudaGetErrorString(error));
            }

            return total / sizeof(value_type);
        }

    public:
        // Constructs count objects of type T in allocated uninitialized
        // storage pointed to by p, using placement-new
        template <typename U, typename ... Args>
        void bulk_construct(U* p, std::size_t count, Args &&... args)
        {
            int threads_per_block = (std::min)(1024, int(count));
            int num_blocks =
                int((count + threads_per_block - 1) / threads_per_block);

            detail::launch(
                target_, num_blocks, threads_per_block,
                [] __device__ (U* p, std::size_t count, Args const&... args)
                {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < count)
                    {
                        ::new (p + idx) U (std::forward<Args>(args)...);
                    }
                },
                p, count, std::forward<Args>(args)...);
            target_.synchronize();
        }

        // Constructs an object of type T in allocated uninitialized storage
        // pointed to by p, using placement-new
        template <typename U, typename ... Args>
        void construct(U* p, Args &&... args)
        {
            detail::launch(
                target_, 1, 1,
                [] __device__ (U* p, Args const&... args)
                {
                    ::new (p) U (std::forward<Args>(args)...);
                },
                p, std::forward<Args>(args)...);
            target_.synchronize();
        }

        // Calls the destructor of count objects pointed to by p
        template <typename U>
        void bulk_destroy(U* p, std::size_t count)
        {
            int threads_per_block = (std::min)(1024, int(count));
            int num_blocks =
                int((count + threads_per_block) / threads_per_block) - 1;

            detail::launch(
                target_, num_blocks, threads_per_block,
                [] __device__ (U* p, std::size_t count)
                {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < count)
                    {
                        (p + idx)->~U();
                    }
                },
                p, count);
            target_.synchronize();
        }

        // Calls the destructor of the object pointed to by p
        template <typename U>
        void destroy(U* p)
        {
            bulk_destroy(p, 1);
        }

        // Access the underlying target (device)
        target_type& target() const HPX_NOEXCEPT
        {
            return target_;
        }

    private:
        target_type& target_;
    };
}}}

#endif
#endif

