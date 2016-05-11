//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_TRANSFER_MAY_06_2016_0140PM)
#define HPX_PARALLEL_UTIL_TRANSFER_MAY_06_2016_0140PM

#include <hpx/config.hpp>
#include <hpx/traits/pointer_category.hpp>

#include <algorithm>
#include <cstring> // for std::memmove
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
        ///////////////////////////////////////////////////////////////////////
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static std::pair<InIter, OutIter>
        copy_memmove(InIter first, std::size_t count, OutIter dest)
        {
            static_assert(std::is_pointer<InIter>::value &&
                    std::is_pointer<OutIter>::value,
                "optimized copy is possible for pointer-iterators only");

            typedef typename std::iterator_traits<InIter>::value_type data_type;

            const char* const first_ch = reinterpret_cast<const char*>(first);
            char* const dest_ch = reinterpret_cast<char*>(dest);

            std::memmove(dest_ch, first_ch, count * sizeof(data_type));

            std::advance(first, count);
            std::advance(dest, count);
            return std::make_pair(first, dest);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // Customization point for optimizing copy operations
        template <typename Category, typename Enable = void>
        struct copy_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                while (first != last)
                    *dest++ = *first++;
                return std::make_pair(first, dest);
            }
        };

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
        template <>
        struct copy_helper<hpx::traits::trivially_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                return copy_memmove(first, std::distance(first, last), dest);
            }
        };
#endif
    }

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE std::pair<InIter, OutIter>
    copy_helper(InIter first, InIter last, OutIter dest)
    {
        typedef decltype(hpx::traits::get_pointer_category(first, dest)) category;
        return detail::copy_helper<category>::call(first, last, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable = void>
        struct copy_n_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                for (std::size_t i = 0; i != count; ++i)
                    *dest++ = *first++;
                return std::make_pair(first, dest);
            }
        };

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
        template <>
        struct copy_n_helper<hpx::traits::trivially_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                return copy_memmove(first, count, dest);
            }
        };
#endif
    }

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE std::pair<InIter, OutIter>
    copy_n_helper(InIter first, std::size_t count, OutIter dest)
    {
        typedef decltype(hpx::traits::get_pointer_category(first, dest)) category;
        return detail::copy_n_helper<category>::call(first, count, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable = void>
        struct move_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                while (first != last)
                    *dest++ = std::move(*first++);
                return std::make_pair(first, dest);
            }
        };

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
        template <>
        struct move_helper<hpx::traits::trivially_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, InIter last, OutIter dest)
            {
                return copy_memmove(first, std::distance(first, last), dest);
            }
        };
#endif
    }

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE std::pair<InIter, OutIter>
    move_helper(InIter first, InIter last, OutIter dest)
    {
        typedef decltype(hpx::traits::get_pointer_category(first, dest)) category;
        return detail::move_helper<category>::call(first, last, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable = void>
        struct move_n_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                for (std::size_t i = 0; i != count; ++i)
                    *dest++ = std::move(*first++);
                return std::make_pair(first, dest);
            }
        };

#if defined(HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)
        template <>
        struct move_n_helper<hpx::traits::trivially_copyable_pointer_tag>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static std::pair<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest)
            {
                return copy_memmove(first, count, dest);
            }
        };
#endif
    }

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE std::pair<InIter, OutIter>
    move_n_helper(InIter first, std::size_t count, OutIter dest)
    {
        typedef decltype(hpx::traits::get_pointer_category(first, dest)) category;
        return detail::move_n_helper<category>::call(first, count, dest);
    }
}}}

#endif
