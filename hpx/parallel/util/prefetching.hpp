//  Copyright (c) 2016 Zahra Khatami
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREFETCHING_LOOP)
#define HPX_PREFETCHING_LOOP

#include <hpx/config.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/is_range.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include <boost/ref.hpp>

#if defined(HPX_HAVE_MM_PREFETCH)
#if defined(HPX_MSVC)
#include <intrin.h>
#endif
#if defined(HPX_GCC_VERSION)
#include <emmintrin.h>
#endif
#endif

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        template <typename Itr, typename ...Ts>
        class prefetching_iterator
          : public std::iterator<
                std::random_access_iterator_tag,
                typename std::iterator_traits<Itr>::value_type
            >
        {
        private:
            typedef std::iterator<
                std::random_access_iterator_tag,
                typename std::iterator_traits<Itr>::value_type
            > base_type;

        public:
            typedef Itr base_iterator;

            typedef typename base_type::value_type value_type;
            typedef typename base_type::difference_type difference_type;
            typedef typename base_type::pointer pointer;
            typedef typename base_type::reference reference;

        private:
            typedef hpx::util::tuple<boost::reference_wrapper<Ts>...> ranges_type;

            ranges_type rngs_;
            base_iterator base_;
            std::size_t chunk_size_;
            std::size_t range_size_;
            std::size_t idx_;

        public:
            explicit prefetching_iterator(std::size_t idx,
                    base_iterator base, std::size_t chunk_size,
                    std::size_t range_size, ranges_type const& rngs)
              : rngs_(rngs), base_(base), chunk_size_(chunk_size),
                range_size_(range_size),
                idx_((std::min)(idx, range_size))
            {}

            ranges_type const& ranges() const
            {
                return rngs_;
            }

            Itr base() const { return base_; }
            std::size_t chunk_size() const { return chunk_size_; }
            std::size_t range_size() const { return range_size_; }
            std::size_t index() const { return idx_; }

            inline prefetching_iterator& operator+=(difference_type rhs)
            {
                std::size_t last =
                    (std::min)(idx_ + rhs * chunk_size_, range_size_);

                std::advance(base_, last - idx_);
                idx_ = last;
                return *this;
            }
            inline prefetching_iterator& operator-=(difference_type rhs)
            {
                std::size_t first = 0;
                if (idx_ > rhs * chunk_size_)
                    first = idx_ - rhs * chunk_size_;

                std::advance(base_, idx_ - first);
                idx_ = first;
                return *this;
            }
            inline prefetching_iterator& operator++()
            {
                *this += 1;
                return *this;
            }
            inline prefetching_iterator& operator--()
            {
                *this -= 1;
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
                // round up distance to cover all of underlying range
                return (idx_ - rhs.idx_ + chunk_size_ - 1) / chunk_size_;
            }
            inline prefetching_iterator
            operator+(difference_type rhs) const
            {
                prefetching_iterator tmp(*this);
                tmp += rhs;
                return tmp;
            }

            // FIXME: What happens if the distance is not divisible by the
            //        chunk_size? Should enforce this?
            inline prefetching_iterator
            operator-(difference_type rhs) const
            {
                prefetching_iterator tmp(*this);
                tmp -= rhs;
                return tmp;
            }

            friend inline prefetching_iterator
            operator+(const prefetching_iterator& lhs, difference_type rhs)
            {
                return rhs + lhs;
            }
            friend inline prefetching_iterator
            operator-(const prefetching_iterator& lhs, difference_type rhs)
            {
                return lhs - rhs;
            }

            // FIXME: should other members be compared too?
            inline bool operator==(const prefetching_iterator& rhs) const
            {
                return idx_ == rhs.idx_ && base_ == rhs.base_;
            }

            // FIXME: should the base iterators be compared too?
            inline bool operator!=(const prefetching_iterator& rhs) const
            {
                return idx_ != rhs.idx_;
            }
            inline bool operator>(const prefetching_iterator& rhs) const
            {
                return idx_ > rhs.idx_;
            }
            inline bool operator<(const prefetching_iterator& rhs) const
            {
                return idx_ < rhs.idx_;
            }
            inline bool operator>=(const prefetching_iterator& rhs) const
            {
                return idx_ >= rhs.idx_;
            }
            inline bool operator<=(const prefetching_iterator& rhs) const
            {
                return idx_ <= rhs.idx_;
            }

            // FIXME: This looks wrong, it should dispatch to the base iterator
            //        instead.
            inline std::size_t & operator[](std::size_t rhs)
            {
                return idx_;
            }

            // FIXME: This looks wrong, it should dispatch to the base iterator
            //        instead.
            inline std::size_t operator*() const
            {
                return idx_;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Helper class to initialize prefetching_iterator
        template <typename Itr, typename ... Ts>
        struct prefetcher_context
        {
        private:
            typedef hpx::util::tuple<boost::reference_wrapper<Ts>...> ranges_type;

            Itr it_begin_;
            Itr it_end_;
            ranges_type rngs_;
            std::size_t chunk_size_;
            std::size_t range_size_;

            static HPX_CONSTEXPR_OR_CONST
                std::size_t sizeof_first_value_type =
                    sizeof(typename hpx::util::tuple_element<
                            0, ranges_type
                        >::type::type);

        public:
            // FIXME: cache line size is probably platform dependent
            static HPX_CONSTEXPR_OR_CONST std::size_t cache_line_size = 64ull;

            prefetcher_context (Itr begin, Itr end,
                    ranges_type const& rngs, std::size_t p_factor = 1)
              : it_begin_(begin), it_end_(end), rngs_(rngs),
                chunk_size_((p_factor * cache_line_size) /
                    sizeof_first_value_type),
                range_size_(std::distance(begin, end))
            {}

            prefetching_iterator<Itr, Ts...> begin()
            {
                return prefetching_iterator<Itr, Ts...>(
                    0ull, it_begin_, chunk_size_, range_size_, rngs_);
            }

            prefetching_iterator<Itr, Ts...> end()
            {
                return prefetching_iterator<Itr, Ts...>(
                    range_size_, it_end_, chunk_size_, range_size_, rngs_);
            }
        };

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_MM_PREFETCH)
        template <typename ... T>
        HPX_FORCEINLINE void prefetch_addresses(T const& ... ts)
        {
            int const sequencer[] = {
                (_mm_prefetch((char*)&ts, _MM_HINT_T0), 0)..., 0
            };
            (void)sequencer;
        }

        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void
        prefetch_containers(hpx::util::tuple<Ts...> const& t,
            hpx::util::detail::pack_c<std::size_t, Is...>, std::size_t idx)
        {
            prefetch_addresses((hpx::util::get<Is>(t).get())[idx]...);
        }
#else
        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void
        prefetch_containers(hpx::util::tuple<Ts...> const& t,
            hpx::util::detail::pack_c<std::size_t, Is...>, std::size_t idx)
        {
            int const sequencer[] = {
                (hpx::util::get<Is>(t).get()[idx], 0)..., 0
            };
            (void)sequencer;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename ... Ts>
        struct loop<detail::prefetching_iterator<Itr, Ts...> >
        {
            typedef detail::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::detail::make_index_pack<
                    sizeof...(Ts)
                >::type index_pack_type;

            template <typename End, typename F>
            static iterator_type
            call(iterator_type it, End end, F && f)
            {
                for (/**/; it != end; ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    std::size_t last = (std::min)(it.index() + it.chunk_size(),
                        it.range_size());

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }

            template <typename End, typename CancelToken, typename F>
            static iterator_type
            call(iterator_type it, End end, CancelToken& tok, F && f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    std::size_t last = (std::min)(it.index() + it.chunk_size(),
                        it.range_size());

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Itr, typename ... Ts>
        struct loop_n<detail::prefetching_iterator<Itr, Ts...> >
        {
            typedef detail::prefetching_iterator<Itr, Ts...> iterator_type;
            typedef typename iterator_type::base_iterator type;
            typedef typename hpx::util::detail::make_index_pack<
                    sizeof...(Ts)
                >::type index_pack_type;

            template <typename F>
            static iterator_type
            call(iterator_type it, std::size_t count, F && f)
            {
                for (/**/; count != 0; (void) --count, ++it)
                {
                    Itr base = it.base();
                    std::size_t j = it.index();

                    std::size_t last = (std::min)(it.index() + it.chunk_size(),
                        it.range_size());

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }

            template <typename CancelToken, typename F>
            static iterator_type
            call(iterator_type it, std::size_t count, CancelToken& tok, F && f)
            {
                for (/**/; count != 0; (void) --count, ++it)
                {
                    if (tok.was_cancelled())
                        break;

                    Itr base = it.base();
                    std::size_t j = it.index();

                    std::size_t last = (std::min)(it.index() + it.chunk_size(),
                        it.range_size());

                    for (/**/; j != last; (void) ++j, ++base)
                        f(base);

                    if (j != it.range_size())
                        prefetch_containers(it.ranges(), index_pack_type(), j);
                }
                return it;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // function to create a prefetcher_context
    template <typename Itr, typename ... Ts>
    detail::prefetcher_context<Itr, Ts const...>
    make_prefetcher_context(Itr base_begin, Itr base_end,
        std::size_t p_factor, Ts const& ... rngs)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Itr>::value,
            "Iterators have to be of random access iterator category");
        static_assert(
            hpx::util::detail::all_of<hpx::traits::is_range<Ts>...>::value,
            "All variadic parameters have to represent ranges");

        typedef hpx::util::tuple<boost::reference_wrapper<Ts const>...> ranges_type;

        auto && ranges = ranges_type(boost::ref(rngs)...);
        return detail::prefetcher_context<Itr, Ts const...>(
            base_begin, base_end, std::move(ranges), p_factor);
    }
}}}

#endif

