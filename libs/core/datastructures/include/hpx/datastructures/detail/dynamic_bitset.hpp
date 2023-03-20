//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// This code was adapted from boost dynamic_bitset
//
// Copyright (c) 2001-2002 Chuck Allison and Jeremy Siek
// Copyright (c) 2003-2006, 2008 Gennaro Prota
// Copyright (c) 2014 Ahmed Charles
// Copyright (c) 2014 Glen Joseph Fernandes (glenjofe@gmail.com)
// Copyright (c) 2014 Riccardo Marcangelo
// Copyright (c) 2018 Evgeny Shulgin

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <algorithm>
#include <climits>    // for CHAR_BIT
#include <cstddef>
#include <cstdint>
#include <iterator>    // used to implement append(Iter, Iter)
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <istream>
#include <ostream>

#if ((defined(HPX_MSVC) || (defined(__clang__) && defined(__c2__)) ||          \
         (defined(HPX_INTEL_VERSION) && defined(_MSC_VER))) &&                 \
    (defined(_M_IX86) || defined(_M_X64))) &&                                  \
    (defined(__POPCNT__) || defined(__AVX__))
#include <intrin.h>
#endif

namespace hpx::detail {
    namespace dynamic_bitset_impl {

        template <typename T>
        constexpr T log2(T val) noexcept
        {
            int ret = -1;
            while (val != 0)
            {
                val >>= 1;
                ++ret;
            }
            return ret;
        }

        // clear all bits except the rightmost one, then calculate the logarithm
        // base 2
        template <typename T>
        constexpr T lowest_bit(T x) noexcept
        {
            HPX_ASSERT(x >= 1);    // PRE
            return log2<T>(x - (x & (x - 1)));
        }

        template <typename T>
        inline constexpr T max_limit = static_cast<T>(-1);

        // Gives (read-)access to the object representation of an object of type
        // T (3.9p4). CANNOT be used on a base sub-object
        template <typename T>
        constexpr unsigned char const* object_representation(T* p) noexcept
        {
            return static_cast<unsigned char const*>(
                static_cast<void const*>(p));
        }

#if defined(HPX_MSVC)
// warning C4293: '>>': shift count negative or too big, undefined behavior
#pragma warning(push)
#pragma warning(disable : 4293)
#endif
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#endif
        template <int Amount, int Width, typename T>
        constexpr T left_shift(T const& v) noexcept
        {
            return Amount >= Width ? 0 : v >> Amount;
        }
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#endif
#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

        // count function implementation
        using byte_type = unsigned char;

        enum class mode
        {
            access_by_bytes,
            access_by_blocks
        };

        template <mode value>
        struct value_to_type
        {
        };

        inline constexpr unsigned int const table_width = 8;

        // Some platforms have fast popcount operation, that allow us to implement
        // counting bits much more efficiently
        template <typename T>
        constexpr std::size_t popcount(T value) noexcept
        {
            constexpr byte_type count_table[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
                2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4,
                5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4,
                3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2,
                3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
                5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4,
                2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4,
                5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
                4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4,
                5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4,
                5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

            std::size_t num = 0u;
            while (value)
            {
                num += count_table[value & ((1u << table_width) - 1)];
                // NOLINTNEXTLINE(clang-diagnostic-shift-count-overflow)
                value = T(static_cast<std::size_t>(value) >> table_width);
            }
            return num;
        }

#if ((defined(HPX_MSVC) || (defined(__clang__) && defined(__c2__)) ||          \
         (defined(HPX_INTEL_VERSION) && defined(_MSC_VER))) &&                 \
    (defined(_M_IX86) || defined(_M_X64))) &&                                  \
    (defined(__POPCNT__) || defined(__AVX__))

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned short>(
            unsigned short value) noexcept
        {
            return static_cast<std::size_t>(__popcnt16(value));
        }

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned int>(
            unsigned int value) noexcept
        {
            return static_cast<std::size_t>(__popcnt(value));
        }

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned __int64>(
            unsigned __int64 value) noexcept
        {
#if defined(_M_X64)
            return static_cast<std::size_t>(__popcnt64(value));
#else
            return static_cast<std::size_t>(
                       __popcnt(static_cast<unsigned int>(value))) +
                static_cast<std::size_t>(
                    __popcnt(static_cast<unsigned int>(value >> 32)));
#endif
        }

#elif defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION) ||                \
    (defined(HPX_INTEL_VERSION) && defined(__GNUC__))

        // Note: gcc builtins are implemented by compiler runtime when the
        // target CPU may not support the necessary instructions
        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned short>(
            unsigned short value) noexcept
        {
            return static_cast<unsigned int>(
                __builtin_popcount(static_cast<unsigned int>(value)));
        }

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned int>(
            unsigned int value) noexcept
        {
            return static_cast<unsigned int>(__builtin_popcount(value));
        }

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned long>(
            unsigned long value) noexcept
        {
            return static_cast<unsigned int>(__builtin_popcountl(value));
        }

        template <>
        HPX_FORCEINLINE std::size_t popcount<unsigned long long>(
            unsigned long long value) noexcept
        {
            return static_cast<unsigned int>(__builtin_popcountll(value));
        }
#endif

        // overload for access by blocks
        template <typename Iterator, typename ValueType>
        std::size_t do_count(Iterator first, std::size_t length, ValueType,
            value_to_type<mode::access_by_blocks>) noexcept
        {
            std::size_t num1 = 0u, num2 = 0u;
            while (length >= 2u)
            {
                num1 += popcount<ValueType>(*first);
                ++first;
                num2 += popcount<ValueType>(*first);
                ++first;
                length -= 2u;
            }

            if (length > 0u)
                num1 += popcount<ValueType>(*first);

            return num1 + num2;
        }

        // overload for access by bytes
        template <typename Iterator>
        std::size_t do_count(Iterator first, std::size_t length,
            int /* dummy */, value_to_type<mode::access_by_bytes>) noexcept
        {
            if (length > 0u)
            {
                byte_type const* p = object_representation(&*first);
                length *= sizeof(*first);

                return do_count(p, length, static_cast<byte_type>(0u),
                    value_to_type<mode::access_by_blocks>{});
            }
            return 0u;
        }
    }    // namespace dynamic_bitset_impl

    ////////////////////////////////////////////////////////////////////////////
    template <typename Block = std::uint64_t,
        typename Allocator = std::allocator<Block>>
    class dynamic_bitset
    {
        static_assert(std::is_unsigned_v<Block>,
            "base type for dynamic_bitset must be unsigned");

        using buffer_type = std::vector<Block, Allocator>;

    public:
        using block_type = Block;
        using allocator_type = Allocator;
        using size_type = std::size_t;
        using block_width_type = typename buffer_type::size_type;

        static constexpr block_width_type bits_per_block =
            static_cast<block_width_type>(std::numeric_limits<Block>::digits);
        static constexpr size_type npos = static_cast<size_type>(-1);

    public:
        // A proxy class to simulate lvalues of bit type.
        class reference    //-V690
        {
            friend class dynamic_bitset<Block, Allocator>;

            // the one and only non-copy ctor
            reference(block_type& b, block_width_type pos) noexcept
              : block_(b)
              , mask_(block_type(1) << pos)
            {
                HPX_ASSERT(pos < bits_per_block);
            }

        public:
            void operator&() = delete;

            reference(reference const& rhs) = default;

            [[nodiscard]] explicit constexpr operator bool() const noexcept
            {
                return (block_ & mask_) != 0;
            }

            [[nodiscard]] constexpr bool operator~() const noexcept
            {
                return (block_ & mask_) == 0;
            }

            reference& flip() noexcept
            {
                do_flip();
                return *this;
            }

            reference& operator=(bool x) noexcept
            {
                do_assign(x);
                return *this;
            }    // for b[i] = x

            reference& operator=(reference const& rhs) noexcept
            {
                do_assign(static_cast<bool>(rhs));
                return *this;
            }    // for b[i] = b[j]

            reference& operator|=(bool x) noexcept
            {
                if (x)
                    do_set();
                return *this;
            }

            reference& operator&=(bool x) noexcept
            {
                if (!x)
                    do_reset();
                return *this;
            }

            reference& operator^=(bool x) noexcept
            {
                if (x)
                    do_flip();
                return *this;
            }

            reference& operator-=(bool x) noexcept
            {
                if (x)
                    do_reset();
                return *this;
            }

        private:
            block_type& block_;
            block_type const mask_;

            void do_set() const noexcept
            {
                block_ |= mask_;
            }

            void do_reset() const noexcept
            {
                block_ &= ~mask_;
            }

            void do_flip() const noexcept
            {
                block_ ^= mask_;
            }

            void do_assign(bool x) const noexcept
            {
                x ? do_set() : do_reset();
            }
        };

        using const_reference = bool;

        // constructors, etc.
        dynamic_bitset() = default;

        explicit dynamic_bitset(Allocator const& alloc);

        explicit dynamic_bitset(size_type nubits, unsigned long value = 0,
            Allocator const& alloc = Allocator());

        // WARNING: you should avoid using this constructor.
        //
        //  A conversion from string is, in most cases, formatting, and should
        //  be performed by using operator>>.
        //
        // NOTE:
        //  Leave the parentheses around std::basic_string<CharT, Traits,
        //  Alloc>::npos. g++ 3.2 requires them and probably the standard will -
        //  see core issue 325
        // NOTE 2:
        //  split into two constructors because of bugs in MSVC 6.0sp5 with
        //  STLport
        template <typename CharT, typename Traits, typename Alloc>
        dynamic_bitset(std::basic_string<CharT, Traits, Alloc> const& s,
            typename std::basic_string<CharT, Traits, Alloc>::size_type pos,
            typename std::basic_string<CharT, Traits, Alloc>::size_type n,
            size_type nubits = npos, Allocator const& alloc = Allocator())
          : bits_(alloc)
        {
            init_from_string(s, pos, n, nubits);
        }

        template <typename CharT, typename Traits, typename Alloc>
        explicit dynamic_bitset(
            std::basic_string<CharT, Traits, Alloc> const& s,
            typename std::basic_string<CharT, Traits, Alloc>::size_type pos = 0)
          : bits_(Allocator())
        {
            init_from_string(
                s, pos, (std::basic_string<CharT, Traits, Alloc>::npos), npos);
        }

        // The first bit in *first is the least significant bit, and the
        // last bit in the block just before *last is the most significant bit.
        template <typename BlockInputIterator>
        dynamic_bitset(BlockInputIterator first, BlockInputIterator last,
            Allocator const& alloc = Allocator())
          : bits_(alloc)
        {
            if constexpr (std::is_arithmetic_v<BlockInputIterator>)
            {
                init_from_unsigned_long(static_cast<size_type>(first), last);
            }
            else
            {
                init_from_block_range(first, last);
            }
        }

    private:
        template <typename BlockIter>
        void init_from_block_range(BlockIter first, BlockIter last)
        {
            HPX_ASSERT(bits_.size() == 0);
            bits_.insert(bits_.end(), first, last);
            nubits_ = bits_.size() * bits_per_block;
        }

    public:
        // copy constructor/assignment
        dynamic_bitset(dynamic_bitset const& b);
        dynamic_bitset& operator=(dynamic_bitset const& b);

        ~dynamic_bitset();

        void swap(dynamic_bitset& b) noexcept;

        dynamic_bitset(dynamic_bitset&& b) noexcept;
        dynamic_bitset& operator=(dynamic_bitset&& b) noexcept;

        [[nodiscard]] allocator_type get_allocator() const noexcept;

        // size changing operations
        void resize(size_type nubits, bool value = false);
        void clear();
        void push_back(bool bit);
        void pop_back();
        void append(Block value);

        template <typename BlockInputIterator>
        void append(BlockInputIterator first, BlockInputIterator last,
            std::input_iterator_tag)
        {
            std::vector<Block, Allocator> v(first, last);
            append(v.begin(), v.end(), std::random_access_iterator_tag());
        }

        template <typename BlockInputIterator>
        void append(BlockInputIterator first, BlockInputIterator last,
            std::forward_iterator_tag)
        {
            HPX_ASSERT(first != last);

            block_width_type r = count_extra_bits();
            std::size_t d = std::distance(first, last);
            bits_.reserve(num_blocks() + d);
            if (r == 0)
            {
                for (; first != last; ++first)
                    bits_.push_back(*first);    // could use vector<>::insert()
            }
            else
            {
                highest_block() |= (*first << r);
                do
                {
                    Block b = *first >> (bits_per_block - r);
                    ++first;
                    bits_.push_back(b | (first == last ? 0 : *first << r));
                } while (first != last);
            }
            nubits_ += bits_per_block * d;
        }

        template <typename BlockInputIterator>
        void append(BlockInputIterator first,
            BlockInputIterator last)    // strong guarantee
        {
            if (first != last)
            {
                using category = typename std::iterator_traits<
                    BlockInputIterator>::iterator_category;
                append(first, last, category{});
            }
        }

        // bitset operations
        dynamic_bitset& operator&=(dynamic_bitset const& rhs) noexcept;
        dynamic_bitset& operator|=(dynamic_bitset const& rhs) noexcept;
        dynamic_bitset& operator^=(dynamic_bitset const& rhs) noexcept;
        dynamic_bitset& operator-=(dynamic_bitset const& rhs) noexcept;
        dynamic_bitset& operator<<=(size_type n);
        dynamic_bitset& operator>>=(size_type n);
        dynamic_bitset operator<<(size_type n) const;
        dynamic_bitset operator>>(size_type n) const;

        // basic bit operations
        dynamic_bitset& set(size_type pos, size_type len, bool val) noexcept;
        dynamic_bitset& set(size_type pos, bool val = true) noexcept;
        dynamic_bitset& set() noexcept;
        dynamic_bitset& reset(size_type pos, size_type len) noexcept;
        dynamic_bitset& reset(size_type pos) noexcept;
        dynamic_bitset& reset() noexcept;
        dynamic_bitset& flip(size_type pos, size_type len) noexcept;
        dynamic_bitset& flip(size_type pos) noexcept;
        dynamic_bitset& flip() noexcept;

        [[nodiscard]] bool test(size_type pos) const noexcept;
        bool test_set(size_type pos, bool val = true) noexcept;
        [[nodiscard]] bool all() const noexcept;
        [[nodiscard]] bool any() const noexcept;
        [[nodiscard]] bool none() const noexcept;

        dynamic_bitset operator~() const;
        [[nodiscard]] size_type count() const noexcept;

        // subscript
        reference operator[](size_type pos)
        {
            return reference(bits_[block_index(pos)], bit_index(pos));
        }

        bool operator[](size_type pos) const
        {
            return test(pos);
        }

        [[nodiscard]] unsigned long to_ulong() const;

        [[nodiscard]] size_type size() const noexcept;
        [[nodiscard]] size_type num_blocks() const noexcept;
        [[nodiscard]] size_type max_size() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_type capacity() const noexcept;
        void reserve(size_type nubits);
        void shrink_to_fit();

        [[nodiscard]] bool is_subset_of(dynamic_bitset const& a) const noexcept;
        [[nodiscard]] bool is_proper_subset_of(
            dynamic_bitset const& a) const noexcept;
        [[nodiscard]] bool intersects(dynamic_bitset const& b) const noexcept;

        // lookup
        [[nodiscard]] size_type find_first() const noexcept;
        [[nodiscard]] size_type find_next(size_type pos) const noexcept;

        // lexicographical comparison
        template <typename B, typename A>
        friend bool operator==(dynamic_bitset<B, A> const& a,
            dynamic_bitset<B, A> const& b) noexcept;

        template <typename B, typename A>
        friend bool operator<(dynamic_bitset<B, A> const& a,
            dynamic_bitset<B, A> const& b) noexcept;

        template <typename B, typename A>
        friend bool oplessthan(dynamic_bitset<B, A> const& a,
            dynamic_bitset<B, A> const& b) noexcept;

        template <typename B, typename A, typename BlockOutputIterator>
        friend void to_block_range(
            dynamic_bitset<B, A> const& b, BlockOutputIterator result);

        template <typename BlockIterator, typename B, typename A>
        friend void from_block_range(BlockIterator first, BlockIterator last,
            dynamic_bitset<B, A>& result);

        template <typename CharT, typename Traits, typename B, typename A>
        friend std::basic_istream<CharT, Traits>& operator>>(
            std::basic_istream<CharT, Traits>& is, dynamic_bitset<B, A>& b);

        template <typename B, typename A, typename String>
        friend void to_string_helper(
            dynamic_bitset<B, A> const& b, String& s, bool dump_all);

        template <typename B, typename A>
        friend std::size_t hash_value(dynamic_bitset<B, A> const& a);

        [[nodiscard]] bool check_invariants() const noexcept;

    private:
        template <typename B, typename A>
        friend void serialize(serialization::input_archive& ar,
            hpx::detail::dynamic_bitset<B, A>& bs, unsigned);

        template <typename B, typename A>
        friend void serialize(serialization::output_archive& ar,
            hpx::detail::dynamic_bitset<B, A> const& bs, unsigned);

        static constexpr block_width_type ulong_width =
            std::numeric_limits<unsigned long>::digits;

        dynamic_bitset& range_operation(size_type pos, size_type len,
            Block (*partial_block_operation)(
                Block, size_type, size_type) noexcept,
            Block (*full_block_operation)(Block) noexcept) noexcept;

        void zero_unused_bits() noexcept;

        static bool m_not_empty(Block x) noexcept
        {
            return x != Block(0);
        }

        [[nodiscard]] size_type do_find_from(
            size_type first_block) const noexcept;

        [[nodiscard]] block_width_type count_extra_bits() const noexcept
        {
            return bit_index(size());
        }

        [[nodiscard]] static size_type block_index(size_type pos) noexcept
        {
            return pos / bits_per_block;
        }

        [[nodiscard]] static block_width_type bit_index(size_type pos) noexcept
        {
            return static_cast<block_width_type>(pos % bits_per_block);
        }

        [[nodiscard]] static Block bit_mask(size_type pos) noexcept
        {
            return Block(1) << bit_index(pos);
        }

        [[nodiscard]] static Block bit_mask(
            size_type first, size_type last) noexcept
        {
            Block res = (last == bits_per_block - 1) ?
                dynamic_bitset_impl::max_limit<Block> :
                ((Block(1) << (last + 1)) - 1);
            res ^= (Block(1) << first) - 1;
            return res;
        }

        [[nodiscard]] static Block set_block_bits(
            Block block, size_type first, size_type last, bool val) noexcept
        {
            if (val)
                return block | bit_mask(first, last);
            return block & static_cast<Block>(~bit_mask(first, last));
        }

        // Functions for operations on ranges
        [[nodiscard]] static Block set_block_partial(
            Block block, size_type first, size_type last) noexcept
        {
            return set_block_bits(block, first, last, true);
        }

        [[nodiscard]] static constexpr Block set_block_full(Block) noexcept
        {
            return dynamic_bitset_impl::max_limit<Block>;
        }

        [[nodiscard]] static Block reset_block_partial(
            Block block, size_type first, size_type last) noexcept
        {
            return set_block_bits(block, first, last, false);
        }

        [[nodiscard]] static constexpr Block reset_block_full(Block) noexcept
        {
            return 0;
        }

        [[nodiscard]] static Block flip_block_partial(
            Block block, size_type first, size_type last) noexcept
        {
            return block ^ bit_mask(first, last);
        }

        [[nodiscard]] static Block flip_block_full(Block block) noexcept
        {
            return ~block;
        }

        template <typename CharT, typename Traits, typename Alloc>
        void init_from_string(std::basic_string<CharT, Traits, Alloc> const& s,
            typename std::basic_string<CharT, Traits, Alloc>::size_type pos,
            typename std::basic_string<CharT, Traits, Alloc>::size_type n,
            size_type nubits)
        {
            HPX_ASSERT(pos <= s.size());

            using StrT = std::basic_string<CharT, Traits, Alloc>;
            using Tr = typename StrT::traits_type;

            typename StrT::size_type const rlen = (std::min)(n, s.size() - pos);
            size_type const sz = (nubits != npos ? nubits : rlen);

            bits_.resize(calc_num_blocks(sz));
            nubits_ = sz;

            std::ctype<CharT> const& fac =
                std::use_facet<std::ctype<CharT>>(std::locale());
            CharT const one = fac.widen('1');

            size_type const m = nubits < rlen ? nubits : rlen;
            typename StrT::size_type i = 0;
            for (; i < m; ++i)
            {
                CharT const c = s[(pos + m - 1) - i];

                HPX_ASSERT(Tr::eq(c, one) || Tr::eq(c, fac.widen('0')));

                if (Tr::eq(c, one))
                    set(i);
            }
        }

        void init_from_unsigned_long(size_type nubits, unsigned long value)
        {
            HPX_ASSERT(bits_.size() == 0);

            bits_.resize(calc_num_blocks(nubits));
            nubits_ = nubits;

            using num_type = unsigned long;

            // zero out all bits at pos >= nubits, if any;
            // note that: nubits == 0 implies value == 0
            if (nubits < static_cast<size_type>(ulong_width))
            {
                num_type const mask = (static_cast<num_type>(1) << nubits) - 1;
                value &= mask;
            }

            auto it = bits_.begin();
            while (value != 0)
            {
                *it++ = static_cast<block_type>(value);
                value = dynamic_bitset_impl::left_shift<bits_per_block,
                    ulong_width>(value);
            }
        }

    private:
        [[nodiscard]] bool unchecked_test(size_type pos) const noexcept;
        static size_type calc_num_blocks(size_type nubits) noexcept;

        Block& highest_block() noexcept;
        Block const& highest_block() const noexcept;

        buffer_type bits_;
        size_type nubits_ = 0;

        class bit_appender;

        friend class bit_appender;
        class bit_appender
        {
            // helper for stream >>. Supplies to the lack of an efficient append
            // at the less significant end: bits are actually appended "at left"
            // but rearranged in the destructor. From the perspective of client
            // code everything works *as if* dynamic_bitset<> had an
            // append_at_right() function (eventually throwing the same
            // exceptions as push_back) except that the function is in fact
            // called bit_appender::do_append().
            dynamic_bitset& bs;
            size_type n = 0;
            Block mask;
            Block* current = nullptr;

        public:
            bit_appender(bit_appender const&) = delete;
            bit_appender& operator=(bit_appender const&) = delete;

            explicit bit_appender(dynamic_bitset& r) noexcept
              : bs(r)
              , mask(0)
            {
            }

            ~bit_appender()
            {
                // reverse the order of blocks, shift
                // if needed, and then resize
                std::reverse(bs.bits_.begin(), bs.bits_.end());
                block_width_type const offs = bit_index(n);
                if (offs)
                    bs >>= (bits_per_block - offs);
                bs.resize(n);    // doesn't enlarge, so can't throw
                HPX_ASSERT(bs.check_invariants());
            }

            void do_append(bool value)
            {
                if (mask == 0)
                {
                    bs.append(Block(0));
                    current = &bs.highest_block();
                    mask = Block(1) << (bits_per_block - 1);
                }

                if (value)
                    *current |= mask;

                mask /= 2;
                ++n;
            }

            [[nodiscard]] constexpr size_type get_count() const noexcept
            {
                return n;
            }
        };
    };

    // Global Functions
    //
    // comparison
    template <typename Block, typename Allocator>
    bool operator!=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept;

    template <typename Block, typename Allocator>
    bool operator<=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept;

    template <typename Block, typename Allocator>
    bool operator>(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept;

    template <typename Block, typename Allocator>
    bool operator>=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept;

    // stream operators
    template <typename CharT, typename Traits, typename Block,
        typename Allocator>
    std::basic_ostream<CharT, Traits>& operator<<(
        std::basic_ostream<CharT, Traits>& os,
        dynamic_bitset<Block, Allocator> const& b);

    template <typename CharT, typename Traits, typename Block,
        typename Allocator>
    std::basic_istream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& is,
        dynamic_bitset<Block, Allocator>& b);

    // bitset operations
    template <typename Block, typename Allocator>
    [[nodiscard]] dynamic_bitset<Block, Allocator> operator&(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y);

    template <typename Block, typename Allocator>
    [[nodiscard]] dynamic_bitset<Block, Allocator> operator|(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y);

    template <typename Block, typename Allocator>
    [[nodiscard]] dynamic_bitset<Block, Allocator> operator^(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y);

    template <typename Block, typename Allocator>
    [[nodiscard]] dynamic_bitset<Block, Allocator> operator-(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y);

    // namespace scope swap
    template <typename Block, typename Allocator>
    void swap(dynamic_bitset<Block, Allocator>& left,
        dynamic_bitset<Block, Allocator>& right) noexcept;

    template <typename Block, typename Allocator, typename String>
    void to_string(dynamic_bitset<Block, Allocator> const& b, String& s);

    template <typename Block, typename Allocator, typename BlockOutputIterator>
    void to_block_range(
        dynamic_bitset<Block, Allocator> const& b, BlockOutputIterator result);

    template <typename BlockIterator, typename B, typename A>
    inline void from_block_range(
        BlockIterator first, BlockIterator last, dynamic_bitset<B, A>& result)
    {
        // PRE: distance(first, last) <= numblocks()
        std::copy(first, last, result.bits_.begin());
    }

    // dynamic_bitset implementation

    // constructors, etc.
    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>::dynamic_bitset(Allocator const& alloc)
      : bits_(alloc)
    {
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>::dynamic_bitset(
        size_type nubits, unsigned long value, Allocator const& alloc)
      : bits_(alloc)
    {
        init_from_unsigned_long(nubits, value);
    }

    // copy constructor
    template <typename Block, typename Allocator>
    inline dynamic_bitset<Block, Allocator>::dynamic_bitset(
        dynamic_bitset const& b)
      : bits_(b.bits_)
      , nubits_(b.nubits_)
    {
    }

    template <typename Block, typename Allocator>
    inline dynamic_bitset<Block, Allocator>::~dynamic_bitset()
    {
        HPX_ASSERT(check_invariants());
    }

    template <typename Block, typename Allocator>
    inline void dynamic_bitset<Block, Allocator>::swap(
        dynamic_bitset<Block, Allocator>& b) noexcept
    {
        std::swap(bits_, b.bits_);
        std::swap(nubits_, b.nubits_);
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator=(
        dynamic_bitset<Block, Allocator> const& b)
    {
        bits_ = b.bits_;
        nubits_ = b.nubits_;
        return *this;
    }

    template <typename Block, typename Allocator>
    inline dynamic_bitset<Block, Allocator>::dynamic_bitset(
        dynamic_bitset<Block, Allocator>&& b) noexcept
      : bits_(HPX_MOVE(b.bits_))
      , nubits_(HPX_MOVE(b.nubits_))
    {
        // Required so that HPX_ASSERT(check_invariants()); works.
        HPX_ASSERT((b.bits_ = buffer_type()).empty());
        b.nubits_ = 0;
    }

    template <typename Block, typename Allocator>
    inline dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator=(
        dynamic_bitset<Block, Allocator>&& b) noexcept
    {
        if (std::addressof(b) == this)
        {
            return *this;
        }

        bits_ = HPX_MOVE(b.bits_);
        nubits_ = HPX_MOVE(b.nubits_);

        // Required so that HPX_ASSERT(check_invariants()); works.
        HPX_ASSERT((b.bits_ = buffer_type()).empty());
        b.nubits_ = 0;
        return *this;
    }

    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::allocator_type
    dynamic_bitset<Block, Allocator>::get_allocator() const noexcept
    {
        return bits_.get_allocator();
    }

    // size changing operations
    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::resize(
        size_type nubits, bool value)    // strong guarantee
    {
        size_type const old_num_blocks = num_blocks();
        size_type const required_blocks = calc_num_blocks(nubits);

        block_type const v =
            value ? dynamic_bitset_impl::max_limit<Block> : Block(0);

        if (required_blocks != old_num_blocks)
        {
            bits_.resize(required_blocks, v);    // s.g. (copy)
        }

        // At this point:
        //
        //  - if the buffer was shrunk, we have nothing more to do,
        //    except a call to zero_unused_bits()
        //
        //  - if it was enlarged, all the (used) bits in the new blocks have
        //    the correct value, but we have not yet touched those bits, if
        //    any, that were 'unused bits' before enlarging: if value == true,
        //    they must be set.

        if (value && (nubits > nubits_))
        {
            block_width_type const extra_bits = count_extra_bits();
            if (extra_bits)
            {
                HPX_ASSERT(
                    old_num_blocks >= 1 && old_num_blocks <= bits_.size());

                // Set them.
                bits_[old_num_blocks - 1] |= (v << extra_bits);
            }
        }

        nubits_ = nubits;
        zero_unused_bits();
    }

    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::clear()    // no throw
    {
        bits_.clear();
        nubits_ = 0;
    }

    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::push_back(bool bit)
    {
        size_type const sz = size();
        resize(sz + 1);
        set(sz, bit);
    }

    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::pop_back()
    {
        size_type const old_num_blocks = num_blocks();
        size_type const required_blocks = calc_num_blocks(nubits_ - 1);

        if (required_blocks != old_num_blocks)
        {
            bits_.pop_back();
        }

        --nubits_;
        zero_unused_bits();
    }

    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::append(
        Block value)    // strong guarantee
    {
        block_width_type const r = count_extra_bits();

        if (r == 0)
        {
            // the buffer is empty, or all blocks are filled
            bits_.push_back(value);
        }
        else
        {
            bits_.push_back(value >> (bits_per_block - r));
            bits_[bits_.size() - 2] |= (value << r);    // bits_.size() >= 2
        }

        nubits_ += bits_per_block;
        HPX_ASSERT(check_invariants());
    }

    //-----------------------------------------------------------------------------
    // bitset operations
    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator&=(
        dynamic_bitset const& rhs) noexcept
    {
        HPX_ASSERT(size() == rhs.size());
        for (size_type i = 0; i < num_blocks(); ++i)
            bits_[i] &= rhs.bits_[i];
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator|=(
        dynamic_bitset const& rhs) noexcept
    {
        HPX_ASSERT(size() == rhs.size());
        for (size_type i = 0; i < num_blocks(); ++i)
            bits_[i] |= rhs.bits_[i];
        //zero_unused_bits();
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator^=(
        dynamic_bitset const& rhs) noexcept
    {
        HPX_ASSERT(size() == rhs.size());
        for (size_type i = 0; i < this->num_blocks(); ++i)
            bits_[i] ^= rhs.bits_[i];
        //zero_unused_bits();
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator-=(
        dynamic_bitset const& rhs) noexcept
    {
        HPX_ASSERT(size() == rhs.size());
        for (size_type i = 0; i < num_blocks(); ++i)
            bits_[i] &= ~rhs.bits_[i];
        //zero_unused_bits();
        return *this;
    }

    // NOTE:
    //  Note that the 'if (r != 0)' is crucial to avoid undefined
    //  behavior when the left hand operand of >> isn't promoted to a
    //  wider type (because rs would be too large).
    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::operator<<=(size_type n)
    {
        if (n >= nubits_)
            return reset();

        if (n > 0)
        {
            size_type const last = num_blocks() - 1;     // num_blocks() is >= 1
            size_type const div = n / bits_per_block;    // div is <= last
            block_width_type const r = bit_index(n);
            block_type* const b = &bits_[0];

            if (r != 0)
            {
                block_width_type const rs = bits_per_block - r;

                for (size_type i = last - div; i > 0; --i)
                {
                    b[i + div] = (b[i] << r) | (b[i - 1] >> rs);
                }
                b[div] = b[0] << r;
            }
            else
            {
                for (size_type i = last - div; i > 0; --i)
                {
                    b[i + div] = b[i];
                }
                b[div] = b[0];
            }

            // zero out div blocks at the less significant end
            std::fill_n(bits_.begin(), div, static_cast<block_type>(0));

            // zero out any 1 bit that flowed into the unused part
            zero_unused_bits();    // thanks to Lester Gong
        }

        return *this;
    }

    // NOTE:
    //  see the comments to operator <<=
    template <typename B, typename A>
    dynamic_bitset<B, A>& dynamic_bitset<B, A>::operator>>=(size_type n)
    {
        if (n >= nubits_)
        {
            return reset();
        }

        if (n > 0)
        {
            size_type const last = num_blocks() - 1;     // num_blocks() is >= 1
            size_type const div = n / bits_per_block;    // div is <= last
            block_width_type const r = bit_index(n);
            block_type* const b = &bits_[0];

            if (r != 0)
            {
                block_width_type const ls = bits_per_block - r;

                for (size_type i = div; i < last; ++i)
                {
                    b[i - div] = (b[i] >> r) | (b[i + 1] << ls);
                }
                // r bits go to zero
                b[last - div] = b[last] >> r;
            }

            else
            {
                for (size_type i = div; i <= last; ++i)
                {
                    b[i - div] = b[i];
                }
                // note the '<=': the last iteration 'absorbs'
                // b[last-div] = b[last] >> 0;
            }

            // div blocks are zero filled at the most significant end
            std::fill_n(bits_.begin() + (num_blocks() - div), div,
                static_cast<block_type>(0));
        }

        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>
    dynamic_bitset<Block, Allocator>::operator<<(size_type n) const
    {
        dynamic_bitset r(*this);
        return r <<= n;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>
    dynamic_bitset<Block, Allocator>::operator>>(size_type n) const
    {
        dynamic_bitset r(*this);
        return r >>= n;
    }

    // basic bit operations
    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::set(
        size_type pos, size_type len, bool val) noexcept
    {
        if (val)
            return range_operation(pos, len, set_block_partial, set_block_full);

        return range_operation(pos, len, reset_block_partial, reset_block_full);
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::set(
        size_type pos, bool val) noexcept
    {
        HPX_ASSERT(pos < nubits_);

        if (val)
            bits_[block_index(pos)] |= bit_mask(pos);
        else
            reset(pos);

        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::set() noexcept
    {
        std::fill(
            bits_.begin(), bits_.end(), dynamic_bitset_impl::max_limit<Block>);
        zero_unused_bits();
        return *this;
    }

    template <typename Block, typename Allocator>
    inline dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::reset(
        size_type pos, size_type len) noexcept
    {
        return range_operation(pos, len, reset_block_partial, reset_block_full);
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::reset(
        size_type pos) noexcept
    {
        HPX_ASSERT(pos < nubits_);
        bits_[block_index(pos)] &= ~bit_mask(pos);
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::reset() noexcept
    {
        std::fill(bits_.begin(), bits_.end(), Block(0));
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::flip(
        size_type pos, size_type len) noexcept
    {
        return range_operation(pos, len, flip_block_partial, flip_block_full);
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::flip(
        size_type pos) noexcept
    {
        HPX_ASSERT(pos < nubits_);
        bits_[block_index(pos)] ^= bit_mask(pos);
        return *this;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::flip() noexcept
    {
        for (size_type i = 0; i < num_blocks(); ++i)
            bits_[i] = ~bits_[i];
        zero_unused_bits();
        return *this;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::unchecked_test(
        size_type pos) const noexcept
    {
        return (bits_[block_index(pos)] & bit_mask(pos)) != 0;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::test(size_type pos) const noexcept
    {
        HPX_ASSERT(pos < nubits_);
        return unchecked_test(pos);
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::test_set(
        size_type pos, bool val) noexcept
    {
        bool const b = test(pos);
        if (b != val)
        {
            set(pos, val);
        }
        return b;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::all() const noexcept
    {
        if (empty())
        {
            return true;
        }

        block_width_type const extra_bits = count_extra_bits();
        block_type const all_ones = dynamic_bitset_impl::max_limit<Block>;

        if (extra_bits == 0)
        {
            for (size_type i = 0, e = num_blocks(); i < e; ++i)
            {
                if (bits_[i] != all_ones)
                {
                    return false;
                }
            }
        }
        else
        {
            for (size_type i = 0, e = num_blocks() - 1; i < e; ++i)
            {
                if (bits_[i] != all_ones)
                {
                    return false;
                }
            }
            block_type const mask = (block_type(1) << extra_bits) - 1;
            if (highest_block() != mask)
            {
                return false;
            }
        }
        return true;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::any() const noexcept
    {
        for (size_type i = 0; i < num_blocks(); ++i)
            if (bits_[i])
                return true;
        return false;
    }

    template <typename Block, typename Allocator>
    inline bool dynamic_bitset<Block, Allocator>::none() const noexcept
    {
        return !any();
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>
    dynamic_bitset<Block, Allocator>::operator~() const
    {
        dynamic_bitset b(*this);
        b.flip();
        return b;
    }

    template <typename Block, typename Allocator>
    typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::count() const noexcept
    {
        using dynamic_bitset_impl::table_width;
        using dynamic_bitset_impl::value_to_type;

        constexpr block_width_type no_padding =
            dynamic_bitset::bits_per_block == CHAR_BIT * sizeof(Block);

        constexpr bool enough_table_width = table_width >= CHAR_BIT;

#if ((defined(HPX_MSVC) || (defined(__clang__) && defined(__c2__)) ||          \
         (defined(HPX_INTEL_VERSION) && defined(_MSC_VER))) &&                 \
    (defined(_M_IX86) || defined(_M_X64))) &&                                  \
    (defined(__POPCNT__) || defined(__AVX__))

        // Windows popcount is effective starting from the unsigned short type
        constexpr bool uneffective_popcount =
            sizeof(Block) < sizeof(unsigned short);

#elif defined(HPX_GCC_VERSION) || defined(HPX_CLANG_VERSION) ||                \
    (defined(HPX_INTEL_VERSION) && defined(__GNUC__))

        // GCC popcount is effective starting from the unsigned int type
        constexpr bool uneffective_popcount =
            sizeof(Block) < sizeof(unsigned int);

#else
        constexpr bool uneffective_popcount = true;
#endif

        constexpr dynamic_bitset_impl::mode m =
            (no_padding && enough_table_width && uneffective_popcount) ?
            dynamic_bitset_impl::mode::access_by_bytes :
            dynamic_bitset_impl::mode::access_by_blocks;

        return do_count(
            bits_.begin(), num_blocks(), Block(0), value_to_type<m>{});
    }

    // conversions
    template <typename B, typename A, typename String>
    void to_string_helper(
        dynamic_bitset<B, A> const& b, String& s, bool dump_all)
    {
        typedef typename String::traits_type Tr;
        typedef typename String::value_type Ch;

        std::ctype<Ch> const& fac =
            std::use_facet<std::ctype<Ch>>(std::locale());
        Ch const zero = fac.widen('0');
        Ch const one = fac.widen('1');

        // Note that this function may access (when
        // dump_all == true) bits beyond position size() - 1

        using size_type = typename dynamic_bitset<B, A>::size_type;

        size_type const len = dump_all ?
            dynamic_bitset<B, A>::bits_per_block * b.num_blocks() :
            b.size();
        s.assign(len, zero);

        for (size_type i = 0; i < len; ++i)
        {
            if (b.unchecked_test(i))
                Tr::assign(s[len - 1 - i], one);
        }
    }

    // A comment similar to the one about the constructor from basic_string can
    // be done here. Thanks to James Kanze for making me (Gennaro) realize this
    // important separation of concerns issue, as well as many things about
    // i18n.
    template <typename Block, typename Allocator, typename String>
    void to_string(dynamic_bitset<Block, Allocator> const& b, String& s)
    {
        to_string_helper(b, s, false);
    }

    // Differently from to_string this function dumps out every bit of the
    // internal representation (may be useful for debugging purposes)
    template <typename B, typename A, typename String>
    void dump_to_string(dynamic_bitset<B, A> const& b, String& s)
    {
        to_string_helper(b, s, true /* =dump_all*/);
    }

    template <typename Block, typename Allocator, typename BlockOutputIterator>
    void to_block_range(
        dynamic_bitset<Block, Allocator> const& b, BlockOutputIterator result)
    {
        // note how this copies *all* bits, including the
        // unused ones in the last block (which are zero)
        std::copy(b.bits_.begin(), b.bits_.end(), result);
    }

    template <typename Block, typename Allocator>
    unsigned long dynamic_bitset<Block, Allocator>::to_ulong() const
    {
        if (nubits_ == 0)
            return 0;    // convention

        // Check for overflows. This may be a performance burden on very large
        // bitsets but is required by the specification, sorry
        if (find_next(ulong_width - 1) != npos)
        {
            HPX_THROW_EXCEPTION(hpx::error::out_of_range,
                "hpx::dynamic_bitset::to_ulong", "overflow");
        }

        // Ok, from now on we can be sure there's no "on" bit
        // beyond the "allowed" positions
        using result_type = unsigned long;

        size_type const maximum_size =
            (std::min)(nubits_, static_cast<size_type>(ulong_width));

        size_type const last_block = block_index(maximum_size - 1);

        HPX_ASSERT((last_block * bits_per_block) <
            static_cast<size_type>(ulong_width));

        result_type result = 0;
        for (size_type i = 0; i <= last_block; ++i)
        {
            size_type const offset = i * bits_per_block;
            result |= (static_cast<result_type>(bits_[i]) << offset);
        }

        return result;
    }

    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::size() const noexcept
    {
        return nubits_;
    }

    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::num_blocks() const noexcept
    {
        return bits_.size();
    }

    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::max_size() const noexcept
    {
        size_type const m = bits_.max_size();
        return m <= (static_cast<size_type>(-1) / bits_per_block) ?
            m * bits_per_block :
            static_cast<size_type>(-1);
    }

    template <typename Block, typename Allocator>
    inline bool dynamic_bitset<Block, Allocator>::empty() const noexcept
    {
        return size() == 0;
    }

    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::capacity() const noexcept
    {
        return bits_.capacity() * bits_per_block;
    }

    template <typename Block, typename Allocator>
    inline void dynamic_bitset<Block, Allocator>::reserve(size_type nubits)
    {
        bits_.reserve(calc_num_blocks(nubits));
    }

    template <typename Block, typename Allocator>
    void dynamic_bitset<Block, Allocator>::shrink_to_fit()
    {
        if (bits_.size() < bits_.capacity())
        {
            buffer_type(bits_).swap(bits_);
        }
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::is_subset_of(
        dynamic_bitset<Block, Allocator> const& a) const noexcept
    {
        HPX_ASSERT(size() == a.size());
        for (size_type i = 0; i < num_blocks(); ++i)
            if (bits_[i] & ~a.bits_[i])
                return false;
        return true;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::is_proper_subset_of(
        dynamic_bitset<Block, Allocator> const& a) const noexcept
    {
        HPX_ASSERT(size() == a.size());
        HPX_ASSERT(num_blocks() == a.num_blocks());

        bool proper = false;
        for (size_type i = 0; i < num_blocks(); ++i)
        {
            Block const& bt = bits_[i];
            Block const& ba = a.bits_[i];

            if (bt & ~ba)
                return false;    // not a subset at all
            if (ba & ~bt)
                proper = true;
        }
        return proper;
    }

    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::intersects(
        dynamic_bitset const& b) const noexcept
    {
        size_type const common_blocks =
            num_blocks() < b.num_blocks() ? num_blocks() : b.num_blocks();

        for (size_type i = 0; i < common_blocks; ++i)
        {
            if (bits_[i] & b.bits_[i])
                return true;
        }
        return false;
    }

    // lookup
    //
    // look for the first bit "on", starting from the block with index
    // first_block
    template <typename Block, typename Allocator>
    typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::do_find_from(
        size_type first_block) const noexcept
    {
        size_type i = std::distance(bits_.begin(),
            std::find_if(
                bits_.begin() + first_block, bits_.end(), m_not_empty));

        if (i >= num_blocks())
            return npos;    // not found

        return i * bits_per_block +
            static_cast<size_type>(dynamic_bitset_impl::lowest_bit(bits_[i]));
    }

    template <typename Block, typename Allocator>
    typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::find_first() const noexcept
    {
        return do_find_from(0);
    }

    template <typename Block, typename Allocator>
    typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::find_next(size_type pos) const noexcept
    {
        size_type const sz = size();
        if (pos >= (sz - 1) || sz == 0)
            return npos;

        ++pos;

        size_type const blk = block_index(pos);
        block_width_type const ind = bit_index(pos);

        // shift bits up to one immediately after current
        Block const fore = bits_[blk] >> ind;

        return fore ? pos +
                static_cast<size_type>(dynamic_bitset_impl::lowest_bit(fore)) :
                      do_find_from(blk + 1);
    }

    //-----------------------------------------------------------------------------
    // comparison

    template <typename Block, typename Allocator>
    bool operator==(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        return (a.nubits_ == b.nubits_) && (a.bits_ == b.bits_);
    }

    template <typename Block, typename Allocator>
    bool operator!=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        return !(a == b);
    }

    template <typename Block, typename Allocator>
    bool operator<(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        //    HPX_ASSERT(a.size() == b.size());
        using size_type = typename dynamic_bitset<Block, Allocator>::size_type;

        size_type bsize(b.size());

        if (!bsize)
        {
            return false;
        }

        size_type asize(a.size());
        if (!asize)
        {
            return true;
        }

        if (asize == bsize)
        {
            for (size_type ii = a.num_blocks(); ii > 0; --ii)
            {
                size_type i = ii - 1;
                if (a.bits_[i] < b.bits_[i])
                    return true;
                if (a.bits_[i] > b.bits_[i])
                    return false;
            }
            return false;
        }

        size_type leqsize((std::min)(asize, bsize));

        for (size_type ii = 0; ii < leqsize; ++ii, --asize, --bsize)
        {
            size_type const i = asize - 1;
            size_type const j = bsize - 1;
            if (a[i] < b[j])
                return true;
            if (a[i] > b[j])
                return false;
        }
        return a.size() < b.size();
    }

    template <typename Block, typename Allocator>
    bool oplessthan(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        //    HPX_ASSERT(a.size() == b.size());
        using size_type = typename dynamic_bitset<Block, Allocator>::size_type;

        size_type bsize(b.num_blocks());
        HPX_ASSERT(bsize == 4);    //-V112

        if (!bsize)
        {
            return false;
        }

        size_type asize(a.num_blocks());
        HPX_ASSERT(asize == 3);

        if (!asize)
        {
            return true;
        }

        size_type leqsize((std::min)(asize, bsize));
        HPX_ASSERT(leqsize == 3);

        // Since we are storing the most significant bit at pos == size() - 1,
        // we need to do the comparisons in reverse.
        //
        for (size_type ii = 0; ii < leqsize; ++ii, --asize, --bsize)
        {
            size_type i = asize - 1;
            size_type j = bsize - 1;
            if (a.bits_[i] < b.bits_[j])
                return true;
            else if (a.bits_[i] > b.bits_[j])
                return false;
        }
        return (a.num_blocks() < b.num_blocks());
    }

    template <typename Block, typename Allocator>
    bool operator<=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        return !(a > b);
    }

    template <typename Block, typename Allocator>
    bool operator>(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        return b < a;
    }

    template <typename Block, typename Allocator>
    bool operator>=(dynamic_bitset<Block, Allocator> const& a,
        dynamic_bitset<Block, Allocator> const& b) noexcept
    {
        return !(a < b);
    }

    // hash operations
    template <typename Block, typename Allocator>
    std::size_t hash_value(dynamic_bitset<Block, Allocator> const& a)
    {
        std::size_t res = hash_value(a.nubits_);
        return res ^ std::hash<decltype(a.bits_)>()(a.bits_);
    }

    // stream operations
    template <typename Ch, typename Tr, typename Block, typename Alloc>
    std::basic_ostream<Ch, Tr>& operator<<(
        std::basic_ostream<Ch, Tr>& os, dynamic_bitset<Block, Alloc> const& b)
    {
        using namespace std;

        ios_base::iostate const ok = ios_base::goodbit;
        ios_base::iostate err = ok;

        typename basic_ostream<Ch, Tr>::sentry cerberos(os);
        if (cerberos)
        {
            std::ctype<Ch> const& fac =
                std::use_facet<std::ctype<Ch>>(os.getloc());
            Ch const zero = fac.widen('0');
            Ch const one = fac.widen('1');

            try
            {
                typedef typename dynamic_bitset<Block, Alloc>::size_type
                    bitset_size_type;
                typedef basic_streambuf<Ch, Tr> buffer_type;

                buffer_type* buf = os.rdbuf();
                // careful: os.width() is signed (and can be < 0)
                bitset_size_type const width = (os.width() <= 0) ?
                    0 :
                    static_cast<bitset_size_type>(os.width());
                streamsize npad = (width <= b.size()) ? 0 : width - b.size();

                Ch const fill_char = os.fill();
                ios_base::fmtflags const adjustfield =
                    os.flags() & ios_base::adjustfield;

                // if needed fill at left; pad is decreased along the way
                if (adjustfield != ios_base::left)
                {
                    for (; 0 < npad; --npad)
                        if (Tr::eq_int_type(Tr::eof(), buf->sputc(fill_char)))
                        {
                            err |= ios_base::failbit;
                            break;
                        }
                }

                if (err == ok)
                {
                    // output the bitset
                    for (bitset_size_type i = b.size(); 0 < i; --i)
                    {
                        typename buffer_type::int_type ret =
                            buf->sputc(b.test(i - 1) ? one : zero);
                        if (Tr::eq_int_type(Tr::eof(), ret))
                        {
                            err |= ios_base::failbit;
                            break;
                        }
                    }
                }

                if (err == ok)
                {
                    // if needed fill at right
                    for (; 0 < npad; --npad)
                    {
                        if (Tr::eq_int_type(Tr::eof(), buf->sputc(fill_char)))
                        {
                            err |= ios_base::failbit;
                            break;
                        }
                    }
                }

                os.width(0);
            }
            catch (...)
            {
                // see std 27.6.1.1/4
                bool rethrow = false;
                try
                {
                    os.setstate(ios_base::failbit);
                }
                catch (...)
                {
                    rethrow = true;
                }
                if (rethrow)
                    throw;
            }
        }

        if (err != ok)
            os.setstate(err);    // may throw exception
        return os;
    }

    template <typename Ch, typename Tr, typename Block, typename Alloc>
    std::basic_istream<Ch, Tr>& operator>>(
        std::basic_istream<Ch, Tr>& is, dynamic_bitset<Block, Alloc>& b)
    {
        using namespace std;

        typedef dynamic_bitset<Block, Alloc> bitset_type;
        typedef typename bitset_type::size_type size_type;

        streamsize const w = is.width();
        size_type const limit =
            0 < w && static_cast<size_type>(w) < b.max_size() ?
            static_cast<size_type>(w) :
            b.max_size();

        ios_base::iostate err = ios_base::goodbit;
        typename basic_istream<Ch, Tr>::sentry cerberos(
            is);    // skips whitespaces
        if (cerberos)
        {
            // in accordance with prop. resolution of lib DR 303 [last checked 4
            // Feb 2004]
            std::ctype<Ch> const& fac =
                std::use_facet<std::ctype<Ch>>(is.getloc());
            Ch const zero = fac.widen('0');
            Ch const one = fac.widen('1');

            b.clear();
            try
            {
                typename bitset_type::bit_appender appender(b);
                basic_streambuf<Ch, Tr>* buf = is.rdbuf();
                typename Tr::int_type c = buf->sgetc();
                for (; appender.get_count() < limit; c = buf->snextc())
                {
                    if (Tr::eq_int_type(Tr::eof(), c))
                    {
                        err |= ios_base::eofbit;
                        break;
                    }
                    else
                    {
                        Ch const to_c = Tr::to_char_type(c);
                        bool const is_one = Tr::eq(to_c, one);

                        if (!is_one && !Tr::eq(to_c, zero))
                            break;    // non digit character

                        appender.do_append(is_one);
                    }

                }    // for
            }
            catch (...)
            {
                // catches from stream buf, or from vector:
                //
                // bits_stored bits have been extracted and stored, and
                // either no further character is extractable or we can't
                // append to the underlying vector (out of memory)

                bool rethrow = false;    // see std 27.6.1.1/4
                try
                {
                    is.setstate(ios_base::badbit);
                }
                catch (...)
                {
                    rethrow = true;
                }
                if (rethrow)
                    throw;
            }
        }

        is.width(0);
        if (b.size() == 0 /*|| !cerberos*/)
            err |= ios_base::failbit;
        if (err != ios_base::goodbit)
            is.setstate(err);    // may throw

        return is;
    }

    // bitset operations
    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator> operator&(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y)
    {
        dynamic_bitset<Block, Allocator> b(x);
        return b &= y;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator> operator|(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y)
    {
        dynamic_bitset<Block, Allocator> b(x);
        return b |= y;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator> operator^(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y)
    {
        dynamic_bitset<Block, Allocator> b(x);
        return b ^= y;
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator> operator-(
        dynamic_bitset<Block, Allocator> const& x,
        dynamic_bitset<Block, Allocator> const& y)
    {
        dynamic_bitset<Block, Allocator> b(x);
        return b -= y;
    }

    // namespace scope swap
    template <typename Block, typename Allocator>
    void swap(dynamic_bitset<Block, Allocator>& left,
        dynamic_bitset<Block, Allocator>& right) noexcept
    {
        left.swap(right);
    }

    // private member functions
    template <typename Block, typename Allocator>
    inline typename dynamic_bitset<Block, Allocator>::size_type
    dynamic_bitset<Block, Allocator>::calc_num_blocks(size_type nubits) noexcept
    {
        return nubits / bits_per_block +
            static_cast<size_type>(nubits % bits_per_block != 0);
    }

    // gives a reference to the highest block
    template <typename Block, typename Allocator>
    inline Block& dynamic_bitset<Block, Allocator>::highest_block() noexcept
    {
        return const_cast<Block&>(
            static_cast<dynamic_bitset const*>(this)->highest_block());
    }

    // gives a const-reference to the highest block
    template <typename Block, typename Allocator>
    inline Block const& dynamic_bitset<Block, Allocator>::highest_block()
        const noexcept
    {
        HPX_ASSERT(size() > 0 && num_blocks() > 0);
        return bits_.back();
    }

    template <typename Block, typename Allocator>
    dynamic_bitset<Block, Allocator>&
    dynamic_bitset<Block, Allocator>::range_operation(size_type pos,
        size_type len,
        Block (*partial_block_operation)(Block, size_type, size_type) noexcept,
        Block (*full_block_operation)(Block) noexcept) noexcept
    {
        HPX_ASSERT(pos + len <= nubits_);

        // Do nothing in case of zero length
        if (!len)
            return *this;

        // Use an additional asserts in order to detect size_type overflow For
        // example: pos = 10, len = size_type_limit - 2, pos + len = 7 In case
        // of overflow, 'pos + len' is always smaller than 'len'
        HPX_ASSERT(pos + len >= len);

        // Start and end blocks of the [pos; pos + len - 1] sequence
        size_type const first_block = block_index(pos);
        size_type const last_block = block_index(pos + len - 1);

        size_type const first_bit_index = bit_index(pos);
        size_type const last_bit_index = bit_index(pos + len - 1);

        if (first_block == last_block)
        {
            // Filling only a sub-block of a block
            bits_[first_block] = partial_block_operation(
                bits_[first_block], first_bit_index, last_bit_index);
        }
        else
        {
            // Check if the corner blocks won't be fully filled with 'val'
            size_type const first_block_shift = bit_index(pos) ? 1 : 0;
            size_type const last_block_shift =
                (bit_index(pos + len - 1) == bits_per_block - 1) ? 0 : 1;

            // Blocks that will be filled with ~0 or 0 at once
            size_type const first_full_block = first_block + first_block_shift;
            size_type const last_full_block = last_block - last_block_shift;

            for (size_type i = first_full_block; i <= last_full_block; ++i)
            {
                bits_[i] = full_block_operation(bits_[i]);
            }

            // Fill the first block from the 'first' bit index to the end
            if (first_block_shift)
            {
                bits_[first_block] = partial_block_operation(
                    bits_[first_block], first_bit_index, bits_per_block - 1);
            }

            // Fill the last block from the start to the 'last' bit index
            if (last_block_shift)
            {
                bits_[last_block] = partial_block_operation(
                    bits_[last_block], 0, last_bit_index);
            }
        }

        return *this;
    }

    // If size() is not a multiple of bits_per_block then not all the bits in
    // the last block are used. This function resets the unused bits (convenient
    // for the implementation of many member functions)
    template <typename Block, typename Allocator>
    inline void dynamic_bitset<Block, Allocator>::zero_unused_bits() noexcept
    {
        HPX_ASSERT(num_blocks() == calc_num_blocks(nubits_));

        // if != 0 this is the number of bits used in the last block
        block_width_type const extra_bits = count_extra_bits();

        if (extra_bits != 0)
        {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
            // NOLINTNEXTLINE(stringop-overflow=)
            highest_block() &= (Block(1) << extra_bits) - 1;
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif
        }
    }

    // check class invariants
    template <typename Block, typename Allocator>
    bool dynamic_bitset<Block, Allocator>::check_invariants() const noexcept
    {
        block_width_type const extra_bits = count_extra_bits();
        if (extra_bits > 0)
        {
            block_type const mask =
                dynamic_bitset_impl::max_limit<Block> << extra_bits;
            if ((highest_block() & mask) != 0)
                return false;
        }

        if (bits_.size() > bits_.capacity() ||
            num_blocks() != calc_num_blocks(size()))
        {
            return false;
        }

        return true;
    }
}    // namespace hpx::detail

// std::hash support
#include <functional>

namespace std {

    template <typename Block, typename Allocator>
    struct hash<::hpx::detail::dynamic_bitset<Block, Allocator>>
    {
        using argument_type = hpx::detail::dynamic_bitset<Block, Allocator>;
        using result_type = std::size_t;

        result_type operator()(argument_type const& a) const noexcept
        {
            std::hash<argument_type> const hasher;
            return hasher(a);
        }
    };
}    // namespace std
