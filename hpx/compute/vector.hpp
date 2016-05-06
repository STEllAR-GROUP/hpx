///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_VECTOR_HPP
#define HPX_COMPUTE_VECTOR_HPP

#include <hpx/config.hpp>
#include <hpx/compute/traits/allocator_traits.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/assert.hpp>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <utility>

namespace hpx { namespace compute
{
    namespace detail
    {
        template <typename T, typename Allocator>
        struct iterator
          : std::iterator<
                std::random_access_iterator_tag,
                T,
                std::ptrdiff_t,
                typename traits::allocator_traits<Allocator>::pointer,
                typename traits::allocator_traits<Allocator>::reference
            >
        {
            typedef
                typename traits::allocator_traits<Allocator>::pointer
                pointer;
            typedef
                typename traits::allocator_traits<Allocator>::reference
                reference;

            iterator()
              : p_(nullptr)
            {}

            // FIXME: should be private
            iterator(pointer p, std::size_t pos)
              : p_(p + pos)
            {}

            iterator const& operator++()
            {
                HPX_ASSERT(p_);
                ++p_;
                return *this;
            }

            iterator const& operator--()
            {
                HPX_ASSERT(p_);
                --p_;
                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp(*this);
                HPX_ASSERT(p_);
                ++p_;
                return tmp;
            }

            iterator operator--(int)
            {
                iterator tmp(*this);
                HPX_ASSERT(p_);
                --p_;
                return tmp;
            }

            bool operator==(iterator const& other) const
            {
                return p_ == other.p_;
            }

            bool operator!=(iterator const& other) const
            {
                return p_ != other.p_;
            }

            bool operator<(iterator const& other) const
            {
                return p_ < other.p_;
            }

            bool operator>(iterator const& other) const
            {
                return p_ > other.p_;
            }

            bool operator<=(iterator const& other) const
            {
                return p_ <= other.p_;
            }

            bool operator>=(iterator const& other) const
            {
                return p_ >= other.p_;
            }

            iterator& operator+=(std::ptrdiff_t offset) const
            {
                HPX_ASSERT(p_);
                p_ += offset;
                return *this;
            }

            iterator& operator-=(std::ptrdiff_t offset) const
            {
                HPX_ASSERT(p_);
                p_ -= offset;
                return *this;
            }

            iterator operator+(std::ptrdiff_t offset) const
            {
                iterator tmp(*this);
                tmp += offset;
                return *this;
            }

            iterator operator-(std::ptrdiff_t offset) const
            {
                iterator tmp(*this);
                tmp -= offset;
                return *this;
            }

            reference operator*() const
            {
                HPX_ASSERT(p_);
                return *p_;
            }

            pointer operator->() const
            {
                HPX_ASSERT(p_);
                return p_;
            }

            // FIXME: operator[] is missing

        private:
            pointer p_;
        };

        template <typename T, typename Allocator>
        struct const_iterator
        {
        };

        template <typename T, typename Allocator>
        struct reverse_iterator
        {
        };

        template <typename T, typename Allocator>
        struct const_reverse_iterator
        {
        };
    }

    template <typename T, typename Allocator = std::allocator<T> >
    class vector
    {
    private:
        typedef traits::allocator_traits<Allocator> alloc_traits;

    public:
        /// Member types (FIXME: add reference to std
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef typename alloc_traits::access_target target_type;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef typename alloc_traits::reference reference;
        typedef typename alloc_traits::const_reference const_reference;
        typedef typename alloc_traits::pointer pointer;
        typedef typename alloc_traits::const_pointer const_pointer;
        typedef detail::iterator<T, Allocator> iterator;
        typedef detail::const_iterator<T, Allocator> const_iterator;
        typedef detail::reverse_iterator<T, Allocator> reverse_iterator;
        typedef detail::const_reverse_iterator<T, Allocator> const_reverse_iterator;

        // Default constructor. Constructs an empty container
        explicit vector(Allocator const& alloc = Allocator())
        // C++-14, delegating ctor version:
        // vector() : vector(Allocator()) {}
        // explicit vector(Allocator const& alloc)
          : size_(0)
          , capacity_(0)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(nullptr)
        {}

        // Constructs the container with count copies of elements with value value
        vector(size_type count, T const& value, Allocator const& alloc = Allocator())
          : size_(count)
          , capacity_(count)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(alloc_traits::allocate(alloc_, count))
        {
            alloc_traits::bulk_construct(alloc_, data_, size_, value);
        }

        // Constructs the container with count default-inserted instances of T.
        // No copies are made.
        explicit vector(size_type count, Allocator const& alloc = Allocator())
          : size_(count)
          , capacity_(count)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(alloc_traits::allocate(alloc_, count))
        {
            alloc_traits::bulk_construct(alloc_, data_, size_, value_type{});
        }

        template <typename InIter,
            typename Enable = typename std::enable_if<
                hpx::traits::is_input_iterator<InIter>::value>::type>
        vector(InIter first, InIter last, Allocator const& alloc)
          : size_(std::distance(first, last))
          , capacity_(size_)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(alloc_traits::allocate(alloc_, size_))
        {
            // FIXME: copy from iterators...
        }

        vector(vector const& other)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(other.alloc_)
          , target_(other.target_)
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            // FIXME: copy from other...
        }

        vector(vector const& other, Allocator const& alloc)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            // FIXME: copy from other...
        }

        vector(vector && other)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(std::move(other.alloc_))
          , target_(other.target_)
          , data_(other.data_)
        {
            other.size_ = 0;
            other.capacity_ = 0;
            other.data_ = nullptr;
        }

        vector(vector && other, Allocator const& alloc)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(other.data_)
        {
            other.size_ = 0;
            other.capacity_ = 0;
            other.data_ = nullptr;
        }

        vector(std::initializer_list<T> init, Allocator const& alloc)
          : size_(init.size())
          , capacity_(init.size())
          , alloc_(alloc)
          , target_(alloc_traits::target(alloc_))
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            // FIXME: copy from init...
        }

        ~vector()
        {
            if(data_ != nullptr)
            {
                alloc_traits::bulk_destroy(alloc_, data_, size_);
                alloc_traits::deallocate(alloc_, data_, capacity_);
            }
        }

        // TODO: implement operator=
        // TODO: implement assign

        /// Returns the allocator associated with the container
        allocator_type get_allocator() const
        {
            return alloc_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Element access
        // TODO: implement at()

        reference operator[](size_type pos)
        {
            HPX_ASSERT(pos < size_);
            return target_type::access(target_, data_, pos);
        }

        const_reference operator[](size_type pos) const
        {
            HPX_ASSERT(pos < size_);
            return target_type::access(target_, data_, pos);
        }

        // TODO: implement front()
        // TODO: implement back()

        /// Returns pointer to the underlying array serving as element storage.
        /// The pointer is such that range [data(); data() + size()) is always
        /// a valid range, even if the container is empty (data() is not
        /// dereferenceable in that case).
        pointer data() HPX_NOEXCEPT
        {
            return data_;
        }

        /// \copydoc data()
        const_pointer data() const HPX_NOEXCEPT
        {
            return data_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Iterators
        // TODO: implement cbegin, cend, rbegin, crbegin, rend, crend
        // TODO: debug support
        iterator begin()
        {
            return iterator(data_, 0);
        }

        iterator end()
        {
            return iterator(data_, size_);
        }


        size_type size_;
        size_type capacity_;
        allocator_type alloc_;
        target_type& target_;
        pointer data_;
    };
}}

#endif
