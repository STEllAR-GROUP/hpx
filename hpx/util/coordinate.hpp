//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header <coordinate> synopsis

#if !defined(HPX_UTIL_COORDINATE_NOV_03_2014_0227PM)
#define HPX_UTIL_COORDINATE_NOV_03_2014_0227PM

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <numeric>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // forward declaration
    template <int Rank> class index;
    template <int Rank> class bounds;
    template <int Rank> class bounds_iterator;

    ///////////////////////////////////////////////////////////////////////////
    // [coord.index], class template index
    template <int Rank>
    class index
    {
        static_assert(Rank > 0, "Rank shall be greater than 0");

    public:
        // constants and types
        static HPX_CONSTEXPR_OR_CONST int rank = Rank;

        typedef std::ptrdiff_t& reference;
        typedef std::ptrdiff_t const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t value_type;

        // [coord.index.cnstr], index construction
        //! Effects: Zero-initializes each component.
        index() noexcept
        {
            std::fill(vs_ + 0, vs_ + rank, 0);
        }

#if !defined(HPX_INTEL_VERSION)
        //! Requires: il.size() == Rank.
        //! Effects: For all i in the range [0, Rank), initializes the ith
        //! component of *this with *(il.begin() + i).
        index(std::initializer_list<value_type> const& il)
        {
            HPX_ASSERT(il.size() == std::size_t(rank) &&
                "il.size() must be equal to Rank");
            std::copy(il.begin(), il.end(), vs_ + 0);
        }
#else
        index(value_type const (&il)[Rank])
        {
            std::copy(il, il + Rank, vs_ + 0);
        }

        index(value_type const* il, std::size_t size)
        {
            HPX_ASSERT(size == std::size_t(rank) && "size must be equal to Rank");
            std::copy(il, il + size, vs_ + 0);
        }
#endif

        // [coord.index.eq], index equality
        //! Returns: true if (*this)[i] == rhs[i] for all i in the range
        //! [0, Rank), otherwise false.
        bool operator==(index const& rhs) const noexcept
        {
            return std::equal(vs_ + 0, vs_ + rank, rhs.vs_ + 0);
        }

        //! Returns: !(*this == rhs).
        bool operator!=(index const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        // [coord.index.cmpt], index component access
        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        reference operator[](size_type n)
        {
            HPX_ASSERT(n < std::size_t(rank) && "n must be less than Rank");
            return vs_[n];
        }

        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        const_reference operator[](size_type n) const
        {
            HPX_ASSERT(n < std::size_t(rank) && "n must be less than Rank");
            return vs_[n];
        }

        // [coord.index.arith], index arithmetic
        //! Returns: index<Rank>{*this} += rhs.
        index operator+(index const& rhs) const
        {
            return index(*this) += rhs;
        }

        //! Returns: index<Rank>{*this} -= rhs.
        index operator-(index const& rhs) const
        {
            return index(*this) -= rhs;
        }

        //! Effects: For all i in the range [0, Rank), adds the ith component
        //! of rhs to the ith component of *this and stores the sum in the ith
        //! component of *this.
        //! Returns: *this.
        index& operator+=(index const& rhs)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] += rhs.vs_[i];
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), subtracts the ith
        //! component of rhs from the ith component of *this and stores the
        //! difference in the ith component of *this.
        //! Returns: *this.
        index& operator-=(index const& rhs)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] -= rhs.vs_[i];
            return *this;
        }

        //! Returns: *this.
        index  operator+() const noexcept
        {
            return index(*this);
        }

        //! Returns: A copy of *this with each component negated.
        index  operator-() const
        {
            index r;
            for (std::size_t i = 0; i < rank; ++i)
                r.vs_[i] = -vs_[i];
            return r;
        }

        //! Returns: index<Rank>{*this} *= v.
        index  operator*(value_type v) const
        {
            return index(*this) *= v;
        }

        //! Returns: index<Rank>{*this} /= v.
        index  operator/(value_type v) const
        {
            return index(*this) /= v;
        }

        //! Effects: For all i in the range [0, Rank), multiplies the ith
        //! component of *this by v and stores the product in the ith component
        //! of *this.
        //! Returns: *this.
        index& operator*=(value_type v)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] *= v;
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), divides the ith component
        //! of *this by v and stores the quotient in the ith component of *this.
        //! Returns: *this.
        index& operator/=(value_type v)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] /= v;
            return *this;
        }

    private:
        value_type vs_[Rank];
    };

    ///////////////////////////////////////////////////////////////////////////
    // [coord.bounds], class template bounds
    template <int Rank>
    class bounds
    {
        static_assert(Rank > 0, "Rank shall be greater than 0");

    public:
        // constants and types
        static HPX_CONSTEXPR_OR_CONST int rank = Rank;

        typedef std::ptrdiff_t& reference;
        typedef std::ptrdiff_t const& const_reference;
        typedef bounds_iterator<Rank> iterator;
        typedef bounds_iterator<Rank> const_iterator;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t value_type;

        // [coord.bounds.cnstr], bounds construction

        //! Effects: Zero-initializes each component.
        bounds() noexcept
        {
            std::fill(vs_ + 0, vs_ + rank, 0);
        }

#if !defined(HPX_INTEL_VERSION)
        //! Requires: il.size() == Rank.
        //! Effects: For all i in the range [0, Rank), initializes the ith
        //! component of *this with *(il.begin() + i).
        bounds(std::initializer_list<value_type> const& il)
        {
            HPX_ASSERT(il.size() == std::size_t(rank) &&
                "il.size() must be equal to Rank");
            std::copy(il.begin(), il.end(), vs_ + 0);
        }
#else
        bounds(value_type const (&il)[Rank])
        {
            std::copy(il, il + Rank, vs_ + 0);
        }

        bounds(value_type const* il, std::size_t size)
        {
            HPX_ASSERT(size == std::size_t(rank) && "size must be equal to Rank");
            std::copy(il, il + size, vs_ + 0);
        }
#endif

        // [coord.bounds.eq], bounds equality
        //! Returns: true if (*this)[i] == rhs[i] for all i in the range
        //! [0, Rank), otherwise false.
        bool operator==(bounds const& rhs) const noexcept
        {
            return std::equal(vs_ + 0, vs_ + rank, rhs.vs_ + 0);
        }

        //! Returns: !(*this == rhs).
        bool operator!=(bounds const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        // [coord.bounds.obs], bounds observers
        //! Returns: The product of all components of *this.
        size_type size() const noexcept
        {
            return std::accumulate(vs_ + 0, vs_ + rank, 1,
                std::multiplies<value_type>());
        }

        //! Returns: true if 0 <= idx[i] and idx[i] < (*this)[i] for all i in
        //! the range [0, Rank), otherwise false.
        bool contains(index<Rank> const& idx) const noexcept
        {
            for (std::size_t i = 0; i < rank; ++i)
            {
                if (!(idx[i] >= 0 && idx[i] < vs_[i]))
                    return false;
            }
            return true;
        }

        // [coord.bounds.iter], bounds iterators
        //! Returns: A bounds_iterator referring to the first element of the
        //! space defined by *this such that *begin() == index<Rank>{} if
        //! size() != 0, begin() == end() otherwise.
        const_iterator begin() const noexcept
        {
            return const_iterator(*this, index<Rank>());
        }

        //! Returns: A bounds_iterator which is the past-the-end iterator for
        //! the space defined by *this.
        const_iterator end() const noexcept
        {
            index<Rank> idx;
            idx[0] = vs_[0];
            return const_iterator(*this, idx);
        }

        // [coord.bounds.cmpt], bounds component access
        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        reference operator[](size_type n)
        {
            HPX_ASSERT(n < std::size_t(rank) && "n must be less than Rank");
            return vs_[n];
        }

        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        const_reference operator[](size_type n) const
        {
            HPX_ASSERT(n < std::size_t(rank) && "n must be less than Rank");
            return vs_[n];
        }

        // [coord.bounds.arith], bounds arithmetic
        //! Returns: bounds<Rank>{*this} += rhs.
        bounds operator+(const index<Rank>& rhs) const
        {
            return bounds(*this) += rhs;
        }

        //! Returns: bounds<Rank>{*this} -= rhs.
        bounds operator-(const index<Rank>& rhs) const
        {
            return bounds(*this) -= rhs;
        }

        //! Effects: For all i in the range [0, Rank), adds the ith component
        //! of rhs to the ith component of *this and stores the sum in the ith
        //! component of *this.
        //! Returns: *this.
        bounds& operator+=(const index<Rank>& rhs)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] += rhs[i];
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), subtracts the ith
        //! component of rhs from the ith component of *this and stores the
        //! difference in the ith component of *this.
        //! Returns: *this.
        bounds& operator-=(const index<Rank>& rhs)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] -= rhs[i];
            return *this;
        }

        //! Returns: bounds<Rank>{*this} *= v.
        bounds  operator*(value_type v) const
        {
            return bounds(*this) /= v;
        }

        //! Returns: bounds<Rank>{*this} /= v.
        bounds  operator/(value_type v) const
        {
            return bounds(*this) /= v;
        }

        //! Effects: For all i in the range [0, Rank), multiplies the ith
        //! component of *this by v and stores the product in the ith component
        //! of *this.
        //! Returns: *this.
        bounds& operator*=(value_type v)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] += v;
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), divides the ith component
        //! of *this by v and stores the quotient in the ith component of *this.
        //! Returns: *this.
        bounds& operator/=(value_type v)
        {
            for (std::size_t i = 0; i < rank; ++i)
                vs_[i] += v;
            return *this;
        }

    private:
        value_type vs_[Rank];
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <int Rank>
        class bounds_iterator_pointer
        {
        public:
            explicit bounds_iterator_pointer(index<Rank> const& idx)
              : index_(idx)
            {}

            index<Rank> const& operator*() const
            {
                return index_;
            }

            index<Rank> const* operator->() const
            {
                return &index_;
            }

        private:
            index<Rank> index_;
        };
    }

    // [coord.bounds.iterator], class template bounds_iterator
    template <int Rank>
    class bounds_iterator
    {
    public:
        typedef std::random_access_iterator_tag iterator_category;
        typedef index<Rank> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef typename detail::bounds_iterator_pointer<Rank> pointer;  // unspecified
        typedef index<Rank> const reference;

        explicit bounds_iterator(bounds<Rank> const& bnd, index<Rank> const& idx)
          : bounds_(bnd)
          , index_(idx)
        {}

        //! Requires: *this and rhs are iterators over the same bounds object.
        //! Returns: index_ == rhs.index_.
        bool operator==(bounds_iterator const& rhs) const
        {
            return index_ == rhs.index_;
        }

        bool operator!=(bounds_iterator const& rhs) const
        {
            return !(*this == rhs);
        }

        bool operator<(bounds_iterator const& rhs) const
        {
            for (std::size_t i = 0; i < Rank; ++i)
            {
                if (index_[i] < rhs.index_[i])
                    return true;
                else if (index_[i] > rhs.index_[i])
                    return false;
            }
            return false;
        }

        bool operator<=(bounds_iterator const& rhs) const
        {
            return !(rhs < *this);
        }

        bool operator>(bounds_iterator const& rhs) const
        {
            return rhs < *this;
        }

        bool operator>=(bounds_iterator const& rhs) const
        {
            return !(*this < rhs);
        }

        //! Requires: *this is not the past-the-end iterator.
        //! Effects: Equivalent to:
        //!          for (auto i = Rank - 1; i >= 0; --i) {
        //!              if (++index_[i] < bounds_[i])
        //!                   return *this;
        //!              index_[i] = 0;
        //!          }
        //!          index_ = unspecified past-the-end value;
        //! Returns: *this.
        bounds_iterator& operator++()
        {
            for (int i = Rank - 1; i >= 0; --i)
            {
                if (++index_[i] < bounds_[i])
                    return *this;
                index_[i] = 0;
            }
            index_[0] = bounds_[0];
            return *this;
        }

        bounds_iterator operator++(int)
        {
            bounds_iterator r(*this);
            ++(*this);
            return r;
        }

        //! Requires: There exists a bounds_iterator<Rank> it such that
        //! *this == ++it.
        //! Effects: *this = it.
        //! Returns: *this.
        bounds_iterator& operator--()
        {
            for (int i = Rank - 1; i >= 0; --i)
            {
                if (--index_[i] >= 0)
                    return *this;
                index_[i] = bounds_[i] - 1;
            }
            // index_[Rank - 1] == -1;
            return *this;
        }

        bounds_iterator operator--(int)
        {
            bounds_iterator r(*this);
            --(*this);
            return r;
        }

        bounds_iterator operator+(difference_type n) const
        {
            return bounds_iterator(*this) += n;
        }

        bounds_iterator& operator+=(difference_type n)
        {
            for (int i = Rank - 1; i >= 0 && n != 0; --i)
            {
                std::ptrdiff_t nx = index_[i] + n;

                if (nx >= bounds_[i])
                {
                    n = nx / bounds_[i];
                    index_[i] = nx % bounds_[i];
                }
                else
                {
                    index_[i] = nx;
                    return *this;
                }
            }

            index_[0] = bounds_[0];
            return *this;
        }

        bounds_iterator operator-(difference_type n) const
        {
            return bounds_iterator(*this) -= n;
        }

        bounds_iterator& operator-=(difference_type n)
        {
            return (*this += (-n));
        }

        difference_type operator-(bounds_iterator const& rhs) const
        {
            difference_type r = 0;
            difference_type flat_bounds = 1;
            for (int i = Rank - 1; i >= 0; --i)
            {
                r += (index_[i] - rhs.index_[i]) * flat_bounds;
                flat_bounds *= bounds_[i];
            }
            return r;
        }

        //! Returns: index_.
        reference operator*() const
        {
            return index_;
        }

        pointer operator->() const
        {
            return pointer(index_);
        }

        reference operator[](difference_type n) const
        {
            return *(*this + n);
        }

    protected:
        bounds<Rank> bounds_;
        index<Rank> index_;
    };

    template <int Rank>
    bounds_iterator<Rank> operator+(
        typename bounds_iterator<Rank>::difference_type n,
        bounds_iterator<Rank> const& rhs)
    {
        return rhs + n;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class index<1>
    {
    public:
        // constants and types
        static HPX_CONSTEXPR_OR_CONST int rank = 1;

        typedef std::ptrdiff_t& reference;
        typedef std::ptrdiff_t const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t value_type;

        // [coord.index.cnstr], index construction
        //! Effects: Zero-initializes each component.
        index() noexcept
          : vs_(0)
        {
        }

        //! Effects: Initializes the 0th component of *this with v.
        //! Remarks: This constructor shall not participate in overload
        //! resolution unless Rank is 1.
        index(value_type v) noexcept
          : vs_(v)
        {}

        // [coord.index.eq], index equality
        //! Returns: true if (*this)[i] == rhs[i] for all i in the range
        //! [0, Rank), otherwise false.
        bool operator==(index const& rhs) const noexcept
        {
            return vs_ == rhs.vs_;
        }

        //! Returns: !(*this == rhs).
        bool operator!=(index const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        // [coord.index.cmpt], index component access
        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        reference operator[](size_type n)
        {
            HPX_ASSERT(n == 0 && "n must be less than Rank (1)");
            return vs_;
        }

        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        const_reference operator[](size_type n) const
        {
            HPX_ASSERT(n == 0 && "n must be less than Rank (1)");
            return vs_;
        }

        // [coord.index.arith], index arithmetic
        //! Returns: index<Rank>{*this} += rhs.
        index operator+(index const& rhs) const
        {
            return index(*this) += rhs;
        }

        //! Returns: index<Rank>{*this} -= rhs.
        index operator-(index const& rhs) const
        {
            return index(*this) -= rhs;
        }

        //! Effects: For all i in the range [0, Rank), adds the ith component
        //! of rhs to the ith component of *this and stores the sum in the ith
        //! component of *this.
        //! Returns: *this.
        index& operator+=(index const& rhs)
        {
            vs_ += rhs.vs_;
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), subtracts the ith
        //! component of rhs from the ith component of *this and stores the
        //! difference in the ith component of *this.
        //! Returns: *this.
        index& operator-=(index const& rhs)
        {
            vs_ -= rhs.vs_;
            return *this;
        }

        //! Requires: Rank == 1.
        //! Effects: ++(*this)[0].
        //! Returns: *this.
        index& operator++()
        {
            return ++vs_, *this;
        }

        //! Requires: Rank == 1.
        //! Returns: index<Rank>{(*this)[0]++}.
        index  operator++(int)
        {
            return index(vs_++);
        }

        //! Requires: Rank == 1.
        //! Effects: --(*this)[0].
        //! Returns: *this.
        index& operator--()
        {
            return --vs_, *this;
        }

        //! Requires: Rank == 1.
        //! Returns: index<Rank>{(*this)[0]--}.
        index  operator--(int)
        {
            return index(vs_--);
        }

        //! Returns: *this.
        index  operator+() const noexcept
        {
            return index(*this);
        }

        //! Returns: A copy of *this with each component negated.
        index  operator-() const
        {
            index r;
            r.vs_ = -vs_;
            return r;
        }

        //! Returns: index<Rank>{*this} *= v.
        index  operator*(value_type v) const
        {
            return index(*this) *= v;
        }

        //! Returns: index<Rank>{*this} /= v.
        index  operator/(value_type v) const
        {
            return index(*this) /= v;
        }

        //! Effects: For all i in the range [0, Rank), multiplies the ith
        //! component of *this by v and stores the product in the ith component
        //! of *this.
        //! Returns: *this.
        index& operator*=(value_type v)
        {
            vs_ *= v;
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), divides the ith component
        //! of *this by v and stores the quotient in the ith component of *this.
        //! Returns: *this.
        index& operator/=(value_type v)
        {
            vs_ /= v;
            return *this;
        }

    private:
        value_type vs_;
    };

    // [coord.index.arith], index arithmetic
    //! Returns: index<Rank>{rhs} *= v.
    template <int Rank>
    index<Rank> operator*(std::ptrdiff_t v, index<Rank> const& rhs)
    {
        return index<Rank>(rhs) *= v;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class bounds<1>
    {
    public:
        // constants and types
        static HPX_CONSTEXPR_OR_CONST int rank = 1;

        typedef std::ptrdiff_t& reference;
        typedef std::ptrdiff_t const& const_reference;
        typedef bounds_iterator<1> iterator;
        typedef bounds_iterator<1> const_iterator;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t value_type;

        // [coord.bounds.cnstr], bounds construction

        //! Effects: Zero-initializes each component.
        bounds() noexcept
          : vs_ (0)
        {}

        //! Effects: Initializes the 0th component of *this with v.
        //! Remarks: This constructor shall not participate in overload
        //! resolution unless Rank is 1.
        bounds(value_type v) noexcept
          : vs_(v)
        {}

        bounds(value_type il[1])
        {
            vs_ = il[0];
        }

        bounds(value_type const* il, std::size_t size)
        {
            HPX_ASSERT(size == 1 && "size must be equal to Rank (1)");
            vs_ = il[0];
        }

        // [coord.bounds.eq], bounds equality
        //! Returns: true if (*this)[i] == rhs[i] for all i in the range
        //! [0, Rank), otherwise false.
        bool operator==(bounds const& rhs) const noexcept
        {
            return vs_ == rhs.vs_;
        }

        //! Returns: !(*this == rhs).
        bool operator!=(bounds const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        // [coord.bounds.obs], bounds observers
        //! Returns: The product of all components of *this.
        size_type size() const noexcept
        {
            return vs_;
        }

        //! Returns: true if 0 <= idx[i] and idx[i] < (*this)[i] for all i in
        //! the range [0, Rank), otherwise false.
        bool contains(index<1> const& idx) const noexcept
        {
            return (idx[0] >= 0 && idx[0] < vs_);
        }

        // [coord.bounds.iter], bounds iterators
        //! Returns: A bounds_iterator referring to the first element of the
        //! space defined by *this such that *begin() == index<Rank>{} if
        //! size() != 0, begin() == end() otherwise.
        const_iterator begin() const noexcept
        {
            return const_iterator(*this, index<1>());
        }

        //! Returns: A bounds_iterator which is the past-the-end iterator for
        //! the space defined by *this.
        const_iterator end() const noexcept
        {
            return const_iterator(*this, index<1>(vs_));
        }

        // [coord.bounds.cmpt], bounds component access
        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        reference operator[](size_type n)
        {
            HPX_ASSERT(n == 0 && "n must be less than Rank (1)");
            return vs_;
        }

        //! Requires: n < Rank.
        //! Returns: A reference to the nth component of *this.
        const_reference operator[](size_type n) const
        {
            HPX_ASSERT(n == 0 && "n must be less than Rank (1)");
            return vs_;
        }

        // [coord.bounds.arith], bounds arithmetic
        //! Returns: bounds<Rank>{*this} += rhs.
        bounds operator+(const index<1>& rhs) const
        {
            return bounds(*this) += rhs;
        }

        //! Returns: bounds<Rank>{*this} -= rhs.
        bounds operator-(const index<1>& rhs) const
        {
            return bounds(*this) -= rhs;
        }

        //! Effects: For all i in the range [0, Rank), adds the ith component
        //! of rhs to the ith component of *this and stores the sum in the ith
        //! component of *this.
        //! Returns: *this.
        bounds& operator+=(const index<1>& rhs)
        {
            vs_ += rhs[0];
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), subtracts the ith
        //! component of rhs from the ith component of *this and stores the
        //! difference in the ith component of *this.
        //! Returns: *this.
        bounds& operator-=(const index<1>& rhs)
        {
            vs_ -= rhs[0];
            return *this;
        }

        //! Returns: bounds<Rank>{*this} *= v.
        bounds  operator*(value_type v) const
        {
            return bounds(*this) /= v;
        }

        //! Returns: bounds<Rank>{*this} /= v.
        bounds  operator/(value_type v) const
        {
            return bounds(*this) /= v;
        }

        //! Effects: For all i in the range [0, Rank), multiplies the ith
        //! component of *this by v and stores the product in the ith component
        //! of *this.
        //! Returns: *this.
        bounds& operator*=(value_type v)
        {
            vs_ += v;
            return *this;
        }

        //! Effects: For all i in the range [0, Rank), divides the ith component
        //! of *this by v and stores the quotient in the ith component of *this.
        //! Returns: *this.
        bounds& operator/=(value_type v)
        {
            vs_ += v;
            return *this;
        }

    private:
        value_type vs_;
    };

    // [coord.bounds.arith], bounds arithmetic
    //! Returns: bounds<Rank>{rhs} += lhs.
    template <int Rank>
    bounds<Rank> operator+(index<Rank> const& lhs, bounds<Rank> const& rhs)
    {
        return bounds<Rank>(rhs) += lhs;
    }

    //! Returns: bounds<Rank>{rhs} *= lhs.
    template <int Rank>
    bounds<Rank> operator*(std::ptrdiff_t v, bounds<Rank> const& rhs)
    {
        return bounds<Rank>(rhs) *= v;
    }
}}

#endif
