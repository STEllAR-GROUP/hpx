///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <hpx/compute/host/target.hpp>
#include <hpx/compute/host/traits/access_target.hpp>
#include <hpx/compute/traits/access_target.hpp>

#include <memory>
#include <utility>

namespace hpx { namespace compute { namespace traits {
    template <typename Allocator>
    struct allocator_traits;

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Allocator, typename Enable = void>
        struct get_target_traits
        {
            using type = compute::traits::access_target<compute::host::target>;
        };

        template <typename Allocator>
        struct get_target_traits<Allocator,
            typename util::always_void<typename Allocator::target_type>::type>
        {
            using type =
                compute::traits::access_target<typename Allocator::target_type>;
        };

        template <typename Allocator, typename Enable = void>
        struct get_reference_type
#if defined(HPX_NATIVE_MIC)
            ;
#else
        {
            using type = typename std::allocator_traits<Allocator>::value_type&;
        };
#endif

        template <typename Allocator>
        struct get_reference_type<Allocator,
            typename util::always_void<typename Allocator::reference>::type>
        {
            typedef typename Allocator::reference type;
        };

        template <typename Allocator, typename Enable = void>
        struct get_const_reference_type
#if defined(HPX_NATIVE_MIC)
            ;
#else
        {
            using type =
                typename std::allocator_traits<Allocator>::value_type const&;
        };
#endif

        template <typename Allocator>
        struct get_const_reference_type<Allocator,
            typename util::always_void<
                typename Allocator::const_reference>::type>
        {
            typedef typename Allocator::const_reference type;
        };

        template <typename Allocator, typename Enable = void>
        struct target_helper_result
        {
            using type = compute::host::target;
        };

        template <typename Allocator>
        struct target_helper_result<Allocator,
            typename hpx::util::always_void<
                typename Allocator::target_type>::type>
        {
            using type = decltype(std::declval<Allocator const&>().target());
        };

        ///////////////////////////////////////////////////////////////////////
        struct target_helper
        {
            template <typename Allocator>
            HPX_HOST_DEVICE static compute::host::target call(
                hpx::traits::detail::wrap_int, Allocator const& /* alloc */)
            {
                return compute::host::target();
            }

            template <typename Allocator>
            HPX_HOST_DEVICE static auto call(int, Allocator const& alloc)
                -> decltype(alloc.target())
            {
                return alloc.target();
            }
        };

        template <typename Allocator>
        HPX_HOST_DEVICE typename target_helper_result<Allocator>::type
        call_target_helper(Allocator const& alloc)
        {
            return target_helper::call(0, alloc);
        }

        ///////////////////////////////////////////////////////////////////////
        struct bulk_construct
        {
            template <typename Allocator, typename... Ts>
            HPX_HOST_DEVICE static void call(hpx::traits::detail::wrap_int,
                Allocator& alloc,
                typename std::allocator_traits<Allocator>::pointer p,
                typename std::allocator_traits<Allocator>::size_type count,
                Ts&&... vs)
            {
                using pointer =
                    typename std::allocator_traits<Allocator>::pointer;
                using value_type =
                    typename std::allocator_traits<Allocator>::value_type;
                using size_type =
                    typename std::allocator_traits<Allocator>::size_type;
                value_type init_value(std::forward<Ts>(vs)...);
                pointer end = p + count;
                size_type allocated = 0;
                for (pointer it = p; it != end; ++it)
                {
#if defined(__CUDA_ARCH__)
                    allocator_traits<Allocator>::construct(
                        alloc, it, init_value);
#else
                    try
                    {
                        allocator_traits<Allocator>::construct(
                            alloc, it, init_value);
                    }
                    catch (...)
                    {
                        allocator_traits<Allocator>::bulk_destroy(
                            alloc, p, allocated);
                        throw;
                    }
#endif
                    ++allocated;
                }
            }

            template <typename Allocator, typename... Ts>
            HPX_HOST_DEVICE static auto call(int, Allocator& alloc,
                typename std::allocator_traits<Allocator>::pointer p,
                typename std::allocator_traits<Allocator>::size_type count,
                Ts&&... vs) -> decltype(alloc.bulk_construct(p, count,
                std::forward<Ts>(vs)...))
            {
                alloc.bulk_construct(p, count, std::forward<Ts>(vs)...);
            }
        };

        template <typename Allocator, typename... Ts>
        HPX_HOST_DEVICE void call_bulk_construct(Allocator& alloc,
            typename std::allocator_traits<Allocator>::pointer p,
            typename std::allocator_traits<Allocator>::size_type count,
            Ts&&... vs)
        {
            bulk_construct::call(0, alloc, p, count, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        struct bulk_destroy
        {
            template <typename Allocator>
            HPX_HOST_DEVICE static void call(hpx::traits::detail::wrap_int,
                Allocator& alloc,
                typename std::allocator_traits<Allocator>::pointer p,
                typename std::allocator_traits<Allocator>::size_type
                    count) noexcept
            {
                using pointer =
                    typename std::allocator_traits<Allocator>::pointer;
                pointer end = p + count;
                for (pointer it = p; it != end; ++it)
                {
                    allocator_traits<Allocator>::destroy(alloc, it);
                }
            }

            template <typename Allocator>
            HPX_HOST_DEVICE static auto call(int, Allocator& alloc,
                typename std::allocator_traits<Allocator>::pointer p,
                typename std::allocator_traits<Allocator>::size_type
                    count) noexcept -> decltype(alloc.bulk_destroy(p, count))
            {
                alloc.bulk_destroy(p, count);
            }
        };

        template <typename Allocator>
        HPX_HOST_DEVICE void call_bulk_destroy(Allocator& alloc,
            typename std::allocator_traits<Allocator>::pointer p,
            typename std::allocator_traits<Allocator>::size_type count) noexcept
        {
            bulk_destroy::call(0, alloc, p, count);
        }
    }    // namespace detail

    template <typename Allocator>
    struct allocator_traits
#if !defined(HPX_NATIVE_MIC)
      : std::allocator_traits<Allocator>
#endif
    {
#if defined(HPX_NATIVE_MIC)
    public:
        using value_type = typename Allocator::value_type;
        using pointer = typename Allocator::pointer;
        using const_pointer = typename Allocator::const_pointer;
        using size_type = typename Allocator::size_type;
        using difference_type = typename Allocator::difference_type;

        static pointer allocate(
            Allocator& alloc, size_type n, const_pointer hint = nullptr)
        {
            return alloc.allocate(n, hint);
        }

        static void deallocate(Allocator& alloc, pointer p, size_type n)
        {
            alloc.deallocate(p, n);
        }

        template <class... Args>
        static void construct(Allocator& alloc, pointer p, Args&&... args)
        {
            alloc.construct(p, std::forward<Args>(args)...);
        }

        template <class... Args>
        static void destroy(Allocator& alloc, pointer p)
        {
            alloc.destroy(p);
        }

        size_type max_size(Allocator const& alloc)
        {
            return alloc.max_size();
        }
#else
    private:
        using base_type = std::allocator_traits<Allocator>;

    public:
        using typename base_type::pointer;
        using typename base_type::size_type;
        using typename base_type::value_type;
#endif

        using reference = typename detail::get_reference_type<Allocator>::type;
        using const_reference =
            typename detail::get_const_reference_type<Allocator>::type;

        using access_target =
            typename detail::get_target_traits<Allocator>::type;
        using target_type = typename access_target::target_type;

        HPX_HOST_DEVICE
        static auto target(Allocator const& alloc)
            -> decltype(detail::call_target_helper(alloc))
        {
            return detail::call_target_helper(alloc);
        }

        template <typename... Ts>
        HPX_HOST_DEVICE static void bulk_construct(
            Allocator& alloc, pointer p, size_type count, Ts&&... vs)
        {
            detail::call_bulk_construct(
                alloc, p, count, std::forward<Ts>(vs)...);
        }

        HPX_HOST_DEVICE
        static void bulk_destroy(
            Allocator& alloc, pointer p, size_type count) noexcept
        {
            if (p != nullptr)
                detail::call_bulk_destroy(alloc, p, count);
        }
    };
}}}    // namespace hpx::compute::traits
