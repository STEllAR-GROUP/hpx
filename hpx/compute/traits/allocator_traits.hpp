///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_ALLOCATOR_TRAITS_HPP
#define HPX_COMPUTE_ALLOCATOR_TRAITS_HPP

#include <hpx/config.hpp>
#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/host/traits/access_target.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/always_void.hpp>

#include <memory>

namespace hpx { namespace compute { namespace traits
{
    template <typename Allocator>
    struct allocator_traits;

    namespace detail
    {
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

        struct target_helper
        {
            template <typename Allocator>
            static typename get_target_traits<Allocator>::type::target_type&
            call(hpx::traits::detail::wrap_int, Allocator& alloc)
            {
                static compute::host::target t;
                return t;
            }

            template <typename Allocator>
            static auto call(int, Allocator& alloc)
              -> decltype(alloc.target())
            {
                return alloc.target();
            }
        };

        struct bulk_construct
        {
            template <typename Allocator, typename T, typename ...Ts>
            static void call(hpx::traits::detail::wrap_int,
                Allocator& alloc, T* p, typename Allocator::size_type count,
                Ts const&... vs)
            {
                T* end = p + count;
                typename Allocator::size_type allocated = 0;
                for(T* it = p; it != end; ++it)
                {
                    try
                    {
                        allocator_traits<Allocator>::construct(alloc, it, vs...);
                    } catch(...)
                    {
                        allocator_traits<Allocator>::bulk_destroy(alloc, p,
                            allocated);
                        throw;
                    }
                    ++allocated;
                }

            }

            template <typename Allocator, typename T, typename ...Ts>
            static auto call(int,
                Allocator& alloc, T* p, typename Allocator::size_type count,
                    Ts const&... vs)
              -> decltype(alloc.bulk_construct(p, count, vs...))
            {
                alloc.bulk_construct(p, count, vs...);
            }
        };

        template <typename Allocator, typename T, typename ...Ts>
        void call_bulk_construct(Allocator& alloc, T* p,
            typename Allocator::size_type count, Ts const&... vs)
        {
            bulk_construct::call(0, alloc, p, count, vs...);
        }

        struct bulk_destroy
        {
            template <typename Allocator, typename T>
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
            static auto call(int,
                Allocator& alloc, T* p, typename Allocator::size_type count)
              -> decltype(alloc.destroy(p, count))
            {
                alloc.bulk_destroy(p, count);
            }
        };

        template <typename Allocator, typename T>
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

        typedef value_type& reference;
        typedef value_type const& const_reference;

        typedef
            typename detail::get_target_traits<Allocator>::type access_target;
        typedef typename access_target::target_type target_type;

        static target_type& target(Allocator& alloc)
        {
            return detail::target_helper::call(0, alloc);
        }

        template <typename T, typename ...Ts>
        static void bulk_construct(Allocator& alloc, T* p, size_type count,
            Ts const&... vs)
        {
            detail::call_bulk_construct(alloc, p, count, vs...);
        }

        template <typename T>
        static void bulk_destroy(Allocator& alloc, T* p, size_type count)
        {
            detail::call_bulk_destroy(alloc, p, count);
        }
    };
}}}

#endif
