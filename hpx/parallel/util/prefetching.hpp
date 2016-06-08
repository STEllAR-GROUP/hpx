//  Copyright (c) 2016 Zahra Khatami
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREFETCHING_LOOP)
#define HPX_PREFETCHING_LOOP

#include <hpx/config.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/util/tuple.hpp>

#include <iterator>
#include <algorithm>
#include <vector>

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {


        template<typename Itr, typename ...TS >
        class prefetching_iterator
        : public std::iterator<std::random_access_iterator_tag, std::size_t>
        {
            public:
            typedef Itr base_iterator;
            hpx::util::tuple<TS*...> M_;
            base_iterator base;
            std::size_t chunk_size;
            std::size_t range_size;
            std::size_t idx;

            explicit prefetching_iterator(std::size_t idx_,
                base_iterator base_ , std::size_t chunk_size_,
                std::size_t range_size_,
                const hpx::util::tuple<TS*...> & A_)
            : M_(A_), base(base_), chunk_size(chunk_size_),
            range_size(range_size_), idx(idx_){}

            typedef  typename std::iterator<std::random_access_iterator_tag,
            std::size_t>::difference_type difference_type;

            inline prefetching_iterator& operator+=(difference_type rhs)
            {
                idx = idx + (rhs*chunk_size);
                base = base + (rhs*chunk_size);
                return *this;
            }
            inline prefetching_iterator& operator-=(difference_type rhs)
            {
                idx = idx - (rhs*chunk_size);
                base = base - (rhs*chunk_size);
                return *this;
            }
            inline prefetching_iterator& operator++()
            {
                idx = idx + chunk_size;
                base = base + chunk_size;
                return *this;
            }
            inline prefetching_iterator& operator--()
            {
                idx = idx - chunk_size;
                base = base - chunk_size;
                return *this;
            }
            inline prefetching_iterator operator++(int)
            {
                prefetching_iterator tmp(*this);
                operator++();
                return tmp;
            }
            inline prefetching_iterator operator--(int)
            {
                prefetching_iterator tmp(*this);
                operator--();
                return tmp;
            }
            inline difference_type
            operator-(const prefetching_iterator& rhs) const
            {
                return (idx-rhs.idx)/chunk_size;
            }
            inline prefetching_iterator
            operator+(difference_type rhs) const
            {
                return prefetching_iterator((idx+(rhs*chunk_size)),
                    (base+(rhs*chunk_size)),chunk_size,range_size,M_);
            }
            inline prefetching_iterator
            operator-(difference_type rhs) const
            {
                return prefetching_iterator((idx-(rhs*chunk_size)),
                    (base-(rhs*chunk_size)),chunk_size,range_size,M_);
            }
            friend inline prefetching_iterator
            operator+(difference_type lhs,
                const prefetching_iterator& rhs)
            {
                return rhs + lhs;
            }
            friend inline prefetching_iterator
            operator-(difference_type lhs,
                const prefetching_iterator& rhs)
            {
                return lhs - rhs;
            }
            inline bool operator==(const prefetching_iterator& rhs) const
            {
                return idx == rhs.idx;
            }
            inline bool operator!=(const prefetching_iterator& rhs) const
            {
                return idx != rhs.idx;
            }
            inline bool operator>(const prefetching_iterator& rhs) const
            {
                return idx > rhs.idx;
            }
            inline bool operator<(const prefetching_iterator& rhs) const
            {
                return idx < rhs.idx;
            }
            inline bool operator>=(const prefetching_iterator& rhs) const
            {
                return idx >= rhs.idx;
            }
            inline bool operator<=(const prefetching_iterator& rhs) const
            {
                return idx <= rhs.idx;
            }

            inline std::size_t & operator[](std::size_t rhs)
            {
                return idx;
            }

            inline std::size_t operator*() const {return idx;}
        };

        ////////////////////////////////////////////////////////////////////

        template <typename T>
        struct identity { typedef T type; };

        //Helper class to initialize prefetching_iterator
        template<typename Itr, typename ...TS >
        struct prefetcher_context
        {
            Itr it_begin;
            Itr it_end;
            std::size_t chunk_size;
            hpx::util::tuple<TS*...> m;
            std::size_t range_size;

            explicit prefetcher_context (Itr begin, Itr end,
                std::size_t p_factor,
                hpx::util::tuple<TS*&&...> l)
            {
                it_begin = begin;
                it_end = end;
                m = l;
                std::size_t size_of_types = sizeof(typename identity<
                    decltype(hpx::util::get<0>(l)[0])>::type);
                chunk_size = p_factor * 64ul / size_of_types;
                range_size = std::distance(begin, end);
            }

            explicit prefetcher_context (Itr begin, Itr end,
                hpx::util::tuple<TS*&&...> l)
            {
                it_begin = begin;
                it_end = end;
                std::size_t size_of_types = sizeof(typename identity<
                    decltype(hpx::util::get<0>(l)[0])>::type);
                chunk_size = 64ul / size_of_types;
                m = l;
                range_size = std::distance(begin, end);
            }

            prefetching_iterator<Itr, TS...> begin()
            {
                return prefetching_iterator<Itr, TS...>(0ul, it_begin,
                    chunk_size, range_size, m);
            }

            prefetching_iterator<Itr, TS...> end()
            {
                return prefetching_iterator<Itr, TS...>(range_size, it_end,
                    chunk_size, range_size, m);
            }

        };

        //function which initialize prefetcher_context
        template<typename Itr, typename ...TS >
        prefetcher_context<Itr, TS...> make_prefetcher_context(Itr idx_begin,
            Itr idx_end, std::size_t p_factor, TS*&&... ts)
        {
            auto && t = hpx::util::forward_as_tuple(std::forward<TS*>(ts)...);
            return prefetcher_context<Itr, TS...>(idx_begin,
                idx_end, p_factor, t);
        }


        template<std::size_t I = 0, typename... TS>
        inline typename std::enable_if<I == sizeof...(TS), void>::type
        prefetch_(hpx::util::tuple<TS...>& t, std::size_t idx)
        {}

        template<std::size_t I = 0, typename... TS>
        inline typename std::enable_if<I < sizeof...(TS), void>::type
        prefetch_(hpx::util::tuple<TS...>& t, std::size_t idx)
        {
            _mm_prefetch(((char*)(&hpx::util::get<I>(t)[idx])), _MM_HINT_T0);
            prefetch_<I + 1, TS...>(t, idx);
        }

    }

}}}

#endif

