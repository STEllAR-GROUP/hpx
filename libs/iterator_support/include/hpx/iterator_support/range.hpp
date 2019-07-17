//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_RANGE_HPP
#define HPX_UTIL_RANGE_HPP

#include <hpx/config.hpp>

#include <cstddef>
#include <iterator>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t N>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        T* begin_impl(T (&array)[N], long) noexcept
        {
            return &array[0];
        }

        template <typename T, std::size_t N>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        T* end_impl(T (&array)[N], long) noexcept
        {
            return &array[N];
        }

        template <typename T, std::size_t N>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        std::size_t size_impl(T const (&array)[N], long) noexcept
        {
            return N;
        }

        template <typename T, std::size_t N>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        bool empty_impl(T const (&array)[N], long) noexcept
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename C, typename R = decltype(std::declval<C&>().begin())>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        R begin_impl(C& c, long)
            noexcept(noexcept(c.begin()))
        {
            return c.begin();
        }

        template <typename C, typename R = decltype(std::declval<C&>().end())>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        R end_impl(C& c, long)
            noexcept(noexcept(c.begin()))
        {
            return c.end();
        }

        template <typename C, typename R = decltype(std::declval<C const&>().size())>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        R size_impl(C const& c, long)
            noexcept(noexcept(c.size()))
        {
            return c.size();
        }

        template <typename C, typename R = decltype(std::declval<C const&>().empty())>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        R empty_impl(C const& c, long)
            noexcept(noexcept(c.empty()))
        {
            return c.empty();
        }

        ///////////////////////////////////////////////////////////////////////
        namespace range_impl
        {
            struct fallback
            {
                template <typename T>
                fallback(T const&){}
            };

            fallback begin(fallback);

            template <typename C, typename R = decltype(begin(std::declval<C&>()))>
            HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
            R begin_impl(C& c, int)
                noexcept(noexcept(begin(c)))
            {
                return begin(c);
            }

            fallback end(fallback);

            template <typename C, typename R = decltype(end(std::declval<C&>()))>
            HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
            R end_impl(C& c, int)
                noexcept(noexcept(end(c)))
            {
                return end(c);
            }
        }
        using range_impl::begin_impl;
        using range_impl::end_impl;

        template <typename C>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        std::size_t size_impl(C const& c, int)
        {
            return std::distance(begin_impl(c, 0L), end_impl(c, 0L));
        }

        template <typename C>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        bool empty_impl(C const& c, int)
        {
            return begin_impl(c, 0L) == end_impl(c, 0L);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct result_of_begin
        {
            typedef decltype(detail::begin_impl(std::declval<T&>(), 0L)) type;
        };

        template <typename T, typename Iter = typename result_of_begin<T>::type>
        struct iterator
        {
            typedef Iter type;
        };

        template <typename T>
        struct iterator<T, range_impl::fallback>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct result_of_end
        {
            typedef decltype(detail::end_impl(std::declval<T&>(), 0L)) type;
        };

        template <typename T, typename Iter = typename result_of_end<T>::type>
        struct sentinel
        {
            typedef Iter type;
        };

        template <typename T>
        struct sentinel<T, range_impl::fallback>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace range_adl
    {
        template <
            typename C,
            typename Iterator = typename detail::iterator<C>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        Iterator begin(C& c)
            noexcept(noexcept(detail::begin_impl(c, 0L)))
        {
            return detail::begin_impl(c, 0L);
        }

        template <
            typename C,
            typename Iterator = typename detail::iterator<C const>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        Iterator begin(C const& c)
            noexcept(noexcept(detail::begin_impl(c, 0L)))
        {
            return detail::begin_impl(c, 0L);
        }

        template <
            typename C,
            typename Sentinel = typename detail::sentinel<C>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        Sentinel end(C& c)
            noexcept(noexcept(detail::end_impl(c, 0L)))
        {
            return detail::end_impl(c, 0L);
        }

        template <
            typename C,
            typename Sentinel = typename detail::sentinel<C const>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        Sentinel end(C const& c)
            noexcept(noexcept(detail::end_impl(c, 0L)))
        {
            return detail::end_impl(c, 0L);
        }

        template <
            typename C,
            typename Iterator = typename detail::iterator<C const>::type,
            typename Sentinel = typename detail::sentinel<C const>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        std::size_t size(C const& c)
            noexcept(noexcept(detail::size_impl(c, 0L)))
        {
            return detail::size_impl(c, 0L);
        }

        template <
            typename C,
            typename Iterator = typename detail::iterator<C const>::type,
            typename Sentinel = typename detail::sentinel<C const>::type>
        HPX_HOST_DEVICE HPX_CONSTEXPR HPX_FORCEINLINE
        bool empty(C const& c)
            noexcept(noexcept(detail::empty_impl(c, 0L)))
        {
            return detail::empty_impl(c, 0L);
        }
    }
    using namespace range_adl;
}}

#endif /*HPX_UTIL_RANGE_HPP*/
