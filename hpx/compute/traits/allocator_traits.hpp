///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_ALLOCATOR_TRAITS_HPP
#define HPX_COMPUTE_ALLOCATOR_TRAITS_HPP

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/always_void.hpp>

#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/host/traits/access_target.hpp>
#include <hpx/compute/host/target.hpp>

#include <memory>

namespace hpx { namespace compute { namespace traits
{
    template <typename Allocator>
    struct allocator_traits;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Allocator, typename Enable = void>
        struct get_target_traits
        {
            typedef
                compute::traits::access_target<compute::host::target> type;
        };

        template <typename Allocator>
        struct get_target_traits<Allocator,
            typename util::always_void<
                typename Allocator::target_type
            >::type>
        {
            typedef
                compute::traits::access_target<typename Allocator::target_type>
                type;
        };

        template <typename Allocator, typename Enable = void>
        struct get_reference_type
        {
            typedef
                typename std::allocator_traits<Allocator>::value_type& type;
        };

        template <typename Allocator>
        struct get_reference_type<Allocator,
            typename util::always_void<
                typename Allocator::reference
            >::type>
        {
            typedef typename Allocator::reference type;
        };

        template <typename Allocator, typename Enable = void>
        struct get_const_reference_type
        {
            typedef
                typename std::allocator_traits<Allocator>::value_type const& type;
        };

        template <typename Allocator>
        struct get_const_reference_type<Allocator,
            typename util::always_void<
                typename Allocator::const_reference
            >::type>
        {
            typedef typename Allocator::const_reference type;
        };

        ///////////////////////////////////////////////////////////////////////
        struct target_helper
        {
            template <typename Allocator>
            HPX_HOST_DEVICE
            static compute::host::target&
            call(hpx::traits::detail::wrap_int, Allocator const& alloc)
            {
                static compute::host::target t;
                return t;
            }

            template <typename Allocator>
            HPX_HOST_DEVICE
            static auto call(int, Allocator const& alloc)
              -> decltype(alloc.target())
            {
                return alloc.target();
            }
        };

        template <typename Allocator>
        HPX_HOST_DEVICE
        auto call_target_helper(Allocator const& alloc)
        ->  decltype(target_helper::call(0, alloc))
        {
            return target_helper::call(0, alloc);
        }

        ///////////////////////////////////////////////////////////////////////
        struct bulk_construct
        {
            template <typename Allocator, typename T, typename ...Ts>
            HPX_HOST_DEVICE
            static void call(hpx::traits::detail::wrap_int,
                Allocator& alloc, T* p, typename Allocator::size_type count,
                Ts &&... vs)
            {
                T init_value(std::forward<Ts>(vs)...);
                T* end = p + count;
                typename Allocator::size_type allocated = 0;
                for(T* it = p; it != end; ++it)
                {
#if defined(__CUDA_ARCH__)
                    allocator_traits<Allocator>::construct(alloc, it, init_value);
#else
                    try
                    {
                        allocator_traits<Allocator>::construct(alloc, it, init_value);
                    } catch(...)
                    {
                        allocator_traits<Allocator>::bulk_destroy(alloc, p,
                            allocated);
                        throw;
                    }
#endif
                    ++allocated;
                }

            }

            template <typename Allocator, typename T, typename ...Ts>
            HPX_HOST_DEVICE
            static auto call(int,
                Allocator& alloc, T* p, typename Allocator::size_type count,
                    Ts &&... vs)
              -> decltype(alloc.bulk_construct(p, count, std::forward<Ts>(vs)...))
            {
                alloc.bulk_construct(p, count, std::forward<Ts>(vs)...);
            }
        };

        template <typename Allocator, typename T, typename ...Ts>
        HPX_HOST_DEVICE
        void call_bulk_construct(Allocator& alloc, T* p,
            typename Allocator::size_type count, Ts &&... vs)
        {
            bulk_construct::call(0, alloc, p, count, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        struct bulk_destroy
        {
            template <typename Allocator, typename T>
            HPX_HOST_DEVICE
            static void call(hpx::traits::detail::wrap_int,
                Allocator& alloc, T* p, typename Allocator::size_type count)
            {
                T* end = p + count;
                for(T* it = p; it != end; ++it)
                {
                    allocator_traits<Allocator>::destroy(alloc, it);
                }
            }

            template <typename Allocator, typename T>
            HPX_HOST_DEVICE
            static auto call(int,
                Allocator& alloc, T* p, typename Allocator::size_type count)
              -> decltype(alloc.bulk_destroy(p, count))
            {
                alloc.bulk_destroy(p, count);
            }
        };

        template <typename Allocator, typename T>
        HPX_HOST_DEVICE
        void call_bulk_destroy(Allocator& alloc, T* p,
            typename Allocator::size_type count)
        {
            bulk_destroy::call(0, alloc, p, count);
        }
    }

    template <typename Allocator>
    struct allocator_traits
      : std::allocator_traits<Allocator>
    {
    private:
        typedef std::allocator_traits<Allocator> base_type;

    public:
        using typename base_type::size_type;
        using typename base_type::value_type;

        typedef
            typename detail::get_reference_type<Allocator>::type reference;
        typedef
            typename detail::get_const_reference_type<Allocator>::type const_reference;

        typedef
            typename detail::get_target_traits<Allocator>::type access_target;
        typedef typename access_target::target_type target_type;

        HPX_HOST_DEVICE
        static target_type& target(Allocator const& alloc)
        {
            return detail::call_target_helper(alloc);
        }

        template <typename T, typename ...Ts>
        HPX_HOST_DEVICE
        static void bulk_construct(Allocator& alloc, T* p, size_type count,
            Ts &&... vs)
        {
            detail::call_bulk_construct(alloc, p, count, std::forward<Ts>(vs)...);
        }

        template <typename T>
        HPX_HOST_DEVICE
        static void bulk_destroy(Allocator& alloc, T* p, size_type count)
        {
            detail::call_bulk_destroy(alloc, p, count);
        }
    };
}}}

#endif
