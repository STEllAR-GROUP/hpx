///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_VECTOR_HPP
#define HPX_COMPUTE_VECTOR_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/compute/detail/iterator.hpp>
#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/traits/allocator_traits.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/runtime/report_error.hpp>
#include <hpx/iterator_support/is_iterator.hpp>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace compute
{
    template <typename T, typename Allocator = std::allocator<T> >
    class vector
    {
    private:
        typedef traits::allocator_traits<Allocator> alloc_traits;

    public:
        /// Member types (FIXME: add reference to std
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef typename alloc_traits::access_target access_target;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef typename alloc_traits::reference reference;
        typedef typename alloc_traits::const_reference const_reference;
        typedef typename alloc_traits::pointer pointer;
        typedef typename alloc_traits::const_pointer const_pointer;
        typedef detail::iterator<T, Allocator> iterator;
        typedef detail::iterator<T const, Allocator> const_iterator;
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
          , data_()
        {}

        // Constructs the container with count copies of elements with value value
        vector(size_type count, T const& value, Allocator const& alloc = Allocator())
          : size_(count)
          , capacity_(count)
          , alloc_(alloc)
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
          , data_(alloc_traits::allocate(alloc_, count))
        {
            alloc_traits::bulk_construct(alloc_, data_, size_);
        }

        template <typename InIter,
            typename Enable = typename std::enable_if<
                hpx::traits::is_input_iterator<InIter>::value>::type>
        vector(InIter first, InIter last, Allocator const& alloc)
          : size_(std::distance(first, last))
          , capacity_(size_)
          , alloc_(alloc)
          , data_(alloc_traits::allocate(alloc_, size_))
        {
            hpx::parallel::util::copy(first, last, begin());
        }

        vector(vector const& other)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(other.alloc_)
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            hpx::parallel::util::copy(other.begin(), other.end(), begin());
        }

        vector(vector const& other, Allocator const& alloc)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(alloc)
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            hpx::parallel::util::copy(other.begin(), other.end(), begin());
        }

        vector(vector && other)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(std::move(other.alloc_))
          , data_(std::move(other.data_))
        {
            other.data_ = pointer(nullptr);
            other.size_ = 0;
            other.capacity_ = 0;
        }

        vector(vector && other, Allocator const& alloc)
          : size_(other.size_)
          , capacity_(other.capacity_)
          , alloc_(alloc)
          , data_(std::move(other.data_))
        {
            other.data_ = pointer(nullptr);
            other.size_ = 0;
            other.capacity_ = 0;
        }

        vector(std::initializer_list<T> init, Allocator const& alloc)
          : size_(init.size())
          , capacity_(init.size())
          , alloc_(alloc)
          , data_(alloc_traits::allocate(alloc_, capacity_))
        {
            hpx::parallel::util::copy(init.begin(), init.end(), begin());
        }

        ~vector()
        {
            if(data_ == nullptr)
                return;

#if !defined(__CUDA_ARCH__)
            try {
#endif
                alloc_traits::bulk_destroy(alloc_, data_, size_);
                alloc_traits::deallocate(alloc_, data_, capacity_);
#if !defined(__CUDA_ARCH__)
            }
            catch(...) {
                // make sure no exception escapes this destructor
                hpx::report_error(std::current_exception());
            }
#endif
        }

        vector& operator=(vector const& other)
        {
            if (this == &other)
                return *this;

            pointer data = alloc_traits::allocate(other.alloc_, other.capacity_);
            hpx::parallel::util::copy(other.begin(), other.end(),
                iterator(data, 0, alloc_traits::target(other.alloc_)));

            if(data_ != nullptr)
            {
                alloc_traits::bulk_destroy(alloc_, data_, size_);
                alloc_traits::deallocate(alloc_, data_, capacity_);
            }

            size_ = other.size_;
            capacity_ = other.capacity_;
            alloc_ = other.alloc_;
            data_ = std::move(data);

            return *this;
        }

        vector& operator=(vector && other)
        {
            if (this == &other)
                return *this;


            size_ = other.size_;
            capacity_ = other.capacity_;
            alloc_ = std::move(other.alloc_);
            data_ = std::move(other.data_);

            other.data_ = pointer(nullptr);
            other.size_ = 0;
            other.capacity_ = 0;

            return *this;
        }

        // TODO: implement assign

        /// Returns the allocator associated with the container
        allocator_type get_allocator() const noexcept
        {
            return alloc_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Element access
        // TODO: implement at()

        HPX_HOST_DEVICE
        reference operator[](size_type pos)
        {
#if !defined(__CUDA_ARCH__)
            HPX_ASSERT(pos < size_);
#endif
            return *(data_ + pos);
        }

        HPX_HOST_DEVICE
        const_reference operator[](size_type pos) const
        {
#if !defined(__CUDA_ARCH__)
            HPX_ASSERT(pos < size_);
#endif
            return *(data_ + pos);
        }

        // TODO: implement front()
        // TODO: implement back()

        /// Returns pointer to the underlying array serving as element storage.
        /// The pointer is such that range [data(); data() + size()) is always
        /// a valid range, even if the container is empty (data() is not
        /// dereferenceable in that case).
        pointer data() noexcept
        {
            return data_;
        }

        /// \copydoc data()
        const_pointer data() const noexcept
        {
            return data_;
        }

        /// Returns a raw pointer corresponding to the address of the data
        /// allocated on the device.
        T* device_data() const noexcept
        {
#if defined(__NVCC__) || defined(__CUDACC__)
            return data_.device_ptr();
#else
            return data_;
#endif
        }

        //
        std::size_t size() const noexcept
        {
            return size_;
        }

        std::size_t capacity() const noexcept
        {
            return capacity_;
        }

        /// Returns: size() == 0
        bool empty() const noexcept
        {
            return size_ == 0;
        }

        /// Effects: If size <= size(), equivalent to calling pop_back()
        /// size() - size times. If size() < size, appends size - size()
        /// default-inserted elements to the sequence.
        ///
        /// Requires: T shall be MoveInsertable and DefaultInsertable into *this.
        ///
        /// Remarks: If an exception is thrown other than by the move constructor
        /// of a non-CopyInsertable T there are no effects.
        ///
        void resize(size_type size)
        {
            // TODO: implement this
        }

        /// Effects: If size <= size(), equivalent to calling pop_back()
        /// size() - size times. If size() < size, appends size - size()
        /// copies of val to the sequence.
        ///
        /// Requires: T shall be CopyInsertable into *this.
        ///
        /// Remarks: If an exception is thrown there are no effects.
        ///
        void resize(size_type size, T const& val)
        {
            // TODO: implement this
        }

        ///////////////////////////////////////////////////////////////////////
        // Iterators
        // TODO: implement cbegin, cend, rbegin, crbegin, rend, crend
        // TODO: debug support
        iterator begin() noexcept
        {
            return iterator(data_, 0, alloc_traits::target(alloc_));
        }

        iterator end() noexcept
        {
            return iterator(data_, size_, alloc_traits::target(alloc_));
        }

        const_iterator cbegin() const noexcept
        {
            return const_iterator(data_, 0, alloc_traits::target(alloc_));
        }

        const_iterator cend() const noexcept
        {
            return const_iterator(data_, size_, alloc_traits::target(alloc_));
        }

        const_iterator begin() const noexcept
        {
            return const_iterator(data_, 0, alloc_traits::target(alloc_));
        }

        const_iterator end() const noexcept
        {
            return const_iterator(data_, size_, alloc_traits::target(alloc_));
        }

        /// Effects: Exchanges the contents and capacity() of *this with that
        /// of x.
        ///
        /// Complexity: Constant time.
        ///
        void swap(vector& other)
        {
            vector tmp = std::move(other);
            other = std::move(*this);
            *this = std::move(tmp);
        }

        /// Effects: Erases all elements in the range [begin(),end()).
        /// Destroys all elements in a. Invalidates all references, pointers,
        /// and iterators referring to the elements of a and may invalidate the
        /// past-the-end iterator.
        ///
        /// Post: a.empty() returns true.
        ///
        /// Complexity: Linear.
        ///
        void clear() noexcept
        {
            alloc_traits::bulk_destroy(alloc_, data_, size_);
            size_ = 0;
        }

    private:
        size_type size_;
        size_type capacity_;
        allocator_type alloc_;
        pointer data_;
    };

    /// Effects: x.swap(y);
    template <typename T, typename Allocator>
    HPX_FORCEINLINE
    void swap(vector<T, Allocator>& x, vector<T, Allocator>& y)
    {
        x.swap(y);
    }
}}

#endif
