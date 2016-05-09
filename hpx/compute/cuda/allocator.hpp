//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_ALLOCATOR_HPP
#define HPX_COMPUTE_CUDA_ALLOCATOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/exception.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/detail/scoped_active_target.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/util/unused.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>
#include <memory>

namespace hpx { namespace compute { namespace cuda
{
    template <typename T>
    class allocator
    {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef T const* const_pointer;
        typedef T& reference;
        typedef T const& const_reference;
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
            return &x;
        }

        const_pointer address(const_reference x) const HPX_NOEXCEPT
        {
            return &x;
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage by calling
        // cudaMalloc, but it is unspecified when and how this function is
        // called. The pointer hint may be used to provide locality of
        // reference: the allocator, if supported by the implementation, will
        // attempt to allocate the new memory block as close as possible to hint.
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
        {
            pointer result = 0;
#if !defined(__CUDA_ARCH__)
            detail::scoped_active_target active(target_);

            cudaError_t error = cudaMalloc(&result, n*sizeof(T));
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(out_of_memory,
                    "cuda::allocator<T>::allocate()",
                    "cudaMalloc failed");
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

            cudaError_t error = cudaFree(p);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::allocator<T>::deallocate()",
                    "cudaFree failed");
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
                    "cudaMemGetInfo failed");
            }

            return total / sizeof(value_type);
        }

        template <typename U, typename ...Args>
        struct construct_helper;

        template <typename U, typename Arg0>
        struct construct_helper<U, Arg0>
        {
            typedef typename util::decay<Arg0>::type arg0_type;
            construct_helper(U* p, std::size_t count, arg0_type const& arg0)
              : p_(p)
              , count_(count)
              , arg0_(arg0)
            {}

            __device__ void operator()()
            {
                U* end = p_ + count_;
                for(U* it=p_; p_ != end; ++p_)
                {
                    ::new (it) U(arg0_);
                }
            }
            U* p_;
            std::size_t count_;
            arg0_type arg0_;
        };

        // Constructs an object of type T in allocated uninitialized storage
        // pointed to by p, using placement-new
        template <typename U, typename ... Args>
        void bulk_construct(U* p, std::size_t count, Args&&... args)
        {
            // FIXME: cuda has problems expanding parameter packs in lambdas ...
            construct_helper<U, Args...> closure(p, count, std::forward<Args>(args)...);

            detail::launch(
                target_, 1, 1,
                closure);
            target_.synchronize();
        }

        template <typename U, typename ... Args>
        void construct(U* p, Args &&... args)
        {
            build_construct(p, 1, std::forward<Args>(args)...);
        }

        // Calls the destructor of the object pointed to by p
        template <typename U>
        void bulk_destroy(U* p, std::size_t count)
        {
            detail::launch(
                target_, 1, 1,
                [p, count] __device__ () mutable
                {
                    U* end = p + count;
                    for(U* it=p; p != end; ++p)
                    {
                        it->~U();
                    }
                });
            target_.synchronize();
        }
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

