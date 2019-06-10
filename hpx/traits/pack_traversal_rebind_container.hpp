//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_PACK_TRAVERSAL_REBIND_CONTAINER_JUN_10_2019_1020AM)
#define HPX_TRAITS_PACK_TRAVERSAL_REBIND_CONTAINER_JUN_10_2019_1020AM

#include <memory>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename NewType, typename Base, typename Enable = void>
    struct pack_traversal_rebind_container;

    /// Specialization for a container with a single type T (no allocator support)
    template <typename NewType, template <class> class Base, typename OldType>
    struct pack_traversal_rebind_container<NewType, Base<OldType>>
    {
        static Base<NewType> call(Base<OldType> const& /*container*/)
        {
            return Base<NewType>();
        }
    };

    /// Specialization for a container with a single type T and
    /// a particular allocator, which is preserved across the remap.
    /// -> We remap the allocator through std::allocator_traits.
    template <typename NewType, template <class, class> class Base,
        typename OldType,typename OldAllocator>
    struct pack_traversal_rebind_container<NewType, Base<OldType, OldAllocator>,
        typename std::enable_if<std::uses_allocator<
                Base<OldType, OldAllocator>, OldAllocator>::value
            >::type>
    {
        using NewAllocator = typename std::allocator_traits<OldAllocator>::
            template rebind_alloc<NewType>;

        // Check whether the second argument of the container was
        // the used allocator.
        static Base<NewType, NewAllocator> call(
            Base<OldType, OldAllocator> const& container)
        {
            // Create a new version of the allocator, that is capable of
            // allocating the mapped type.
            return Base<NewType, NewAllocator>(
                NewAllocator(container.get_allocator()));
        }
    };
}}

#endif

