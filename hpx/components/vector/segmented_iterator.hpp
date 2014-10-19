//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#ifndef HPX_VECTOR_SEGMENTED_ITERATOR_HPP
#define HPX_VECTOR_SEGMENTED_ITERATOR_HPP

/// \file hpx/components/vector/segmented_iterator.hpp
/// \brief This file contains the implementation of iterators for hpx::vector.

 // The idea for these iterators is taken from
 // http://afstern.org/matt/segmented.pdf.

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/vector/chunk_vector_component.hpp>

#include <cstdint>
#include <iterator>

#include <boost/integer.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T> class vector;       // forward declaration

    template <typename T> class local_vector_iterator;
    template <typename T> class const_local_vector_iterator;

    template <typename T> class vector_iterator;
    template <typename T> class const_vector_iterator;

    template <typename T, typename BaseIter = vector_iterator<T> >
    class segment_vector_iterator;
    template <typename T> class const_segment_vector_iterator;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the local iterator functionality for
    /// the chunked backend of a hpx::vector.
    template <typename T>
    class local_vector_iterator
      : public boost::iterator_facade<
            local_vector_iterator<T>, T, std::random_access_iterator_tag, T
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                local_vector_iterator<T>, T, std::random_access_iterator_tag, T
            > base_type;

    public:
        // constructors
        local_vector_iterator()
          : chunk_(), local_index_(size_type(-1))
        {}

        local_vector_iterator(chunk_vector<T> chunk, size_type local_index)
          : chunk_(chunk), local_index_(local_index)
        {}

    protected:
        bool is_at_end() const
        {
            return !chunk_ || local_index_ == size_type(-1);
        }

        friend class boost::iterator_core_access;

        bool equal(local_vector_iterator const& other) const
        {
            return chunk_ == other.chunk_ && local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return chunk_.get_value(local_index_);
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++local_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --local_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            local_index_ += n;
        }

        std::ptrdiff_t distance_to(local_vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;

                return chunk_.size() - local_index_;
            }

            if (is_at_end())
                return other_.local_index_ - other.chunk_.size();

            HPX_ASSERT(chunk_ == other.chunk_);
            return other.local_index_ - local_index_;
        }

    public:
        chunk_vector<T>& get_client() { return chunk_; }
        chunk_vector<T> const& get_client() const { return chunk_; }

        size_type get_local_index() const { return local_index_; }

    protected:
        // refer to a partition of the vector
        chunk_vector<T> chunk_;

        // local position in the referenced partition
        size_type local_index_;
    };

    template <typename T>
    class const_local_vector_iterator : public local_vector_iterator<T const>
    {
    public:
        // constructors
        const_local_vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        const_local_vector_iterator(chunk_vector<T> chunk, size_type local_index)
          : local_vector_iterator(chunk, local_index)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implement the segmented iterator for the hpx::vector
    template <typename T, typename BaseIter>
    class segment_vector_iterator
      : public boost::iterator_adaptor<
            segment_vector_iterator<T, BaseIter>, BaseIter, T,
            std::random_access_iterator_tag, T
        >
    {
    private:
        typedef boost::iterator_adaptor<
            segment_vector_iterator<T, BaseIter>, BaseIter, T,
            std::random_access_iterator_tag, T
        > base_type;

    public:
        segment_vector_iterator()
          : data_(0)
        {}

        segment_vector_iterator(vector<T>* data, BaseIter const& it)
          : base_type(it), data_(data)
        {}

        vector<T>* get_data() const { return data_; }

    private:
        vector<T>* data_;
    };

    template <typename T>
    class const_segment_vector_iterator
      : public segment_vector_iterator<T const, const_vector_iterator<T> >
    {
    public:
        const_segment_vector_iterator()
          : data_(0)
        {}

        const_segment_vector_iterator(vector<T>* data,
                const_vector_iterator<T> const& it)
          : base_type(it), data_(data)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the iterator functionality for hpx::vector.
    template <typename T>
    class vector_iterator
      : public boost::iterator_facade<
            vector_iterator<T>, T, std::random_access_iterator_tag, T
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                vector_iterator<T>, T, std::random_access_iterator_tag, T
            > base_type;

    public:
        typedef segment_vector_iterator<T> segment_iterator;
        typedef local_vector_iterator<T> local_iterator;

        // constructors
        vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        vector_iterator(vector<T>* data, size_type global_index)
          : data_(data), global_index_(global_index)
        {}

    protected:
        bool is_at_end() const
        {
            return data_ == 0 || global_index_ == size_type(-1);
        }

        friend class boost::iterator_core_access;

        bool equal(vector_iterator const& other) const
        {
            return data_ == other.data_ && global_index_ == other.global_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return data_->get_value(global_index_);
        }

        void increment()
        {
            HPX_ASSERT(!is_at_end());
            ++global_index_;
        }

        void decrement()
        {
            HPX_ASSERT(!is_at_end());
            --global_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(!is_at_end());
            global_index_ += n;
        }

        std::ptrdiff_t distance_to(vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;
                return data_->size() - global_index_;
            }

            if(is_at_end())
                return other.global_index_ - other.data_->size();

            HPX_ASSERT(data_ == other.data_);
            return other.global_index_ - global_index_;
        }

    protected:
        // refer to the vector
        vector<T>* data_;

        // global position in the referenced vector
        size_type global_index_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class const_vector_iterator : public vector_iterator<T const>
    {
    private:
        typedef std::size_t size_type;

    public:
        typedef const_segment_vector_iterator<T> segment_iterator;
        typedef const_local_vector_iterator<T> local_iterator;

        // constructors
        const_vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        const_vector_iterator(vector<T>* data, size_type global_index)
          : vector_iterator(data, global_index)
        {}
    };
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename T>
    struct segmented_iterator_traits<vector_iterator<T> >
    {
        typedef boost::mpl::true_ is_segmented_iterator;

        typedef vector_iterator<T> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator const& iter)
        {
            return iter.data_->get_segment_iterator(iter.global_index_);
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator const& iter)
        {
            return iter.data_->get_local_iterator(iter.global_index_);
        }

        //
        static iterator compose(segment_iterator const& seg_iter,
            local_iterator const& local_iter)
        {
            vector* data = seg_iter.get_data();
            std::size_t local_index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter->base_index_));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the chunk.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter->chunk_, 0);
        }

        //  This function should specify the local iterator which is at the
        //  end of the chunk.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter->chunk_, seg_iter->size_);
        }
    };

    template <typename T>
    struct segmented_iterator_traits<const_vector_iterator<T> >
    {
        typedef boost::mpl::true_ is_segmented_iterator;

        typedef const_vector_iterator<T> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator const& iter)
        {
            return iter.data_->get_segment_iterator(iter.global_index_);
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator const& iter)
        {
            return iter.data_->get_local_iterator(iter.global_index_);
        }

        //
        static iterator compose(segment_iterator const& seg_iter,
            local_iterator const& local_iter)
        {
            vector* data = seg_iter.get_data();
            std::size_t local_index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter->base_index_));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the chunk.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter->chunk_, 0);
        }

        //  This function should specify the local iterator which is at the
        //  end of the chunk.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter->chunk_, seg_iter->size_);
        }
    };
}}

#endif //  SEGMENTED_ITERATOR_HPP
