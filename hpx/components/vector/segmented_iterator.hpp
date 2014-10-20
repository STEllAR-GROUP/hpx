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

#include <hpx/components/vector/partition_vector_component.hpp>

#include <cstdint>
#include <iterator>

#include <boost/integer.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <boost/serialization/serialization.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T> class vector;       // forward declaration

    template <typename T> class local_vector_iterator;
    template <typename T> class const_local_vector_iterator;

    template <typename T> class vector_iterator;
    template <typename T> class const_vector_iterator;

    template <typename T, typename BaseIter> class segment_vector_iterator;
    template <typename T, typename BaseIter> class const_segment_vector_iterator;

    namespace server
    {
        class HPX_COMPONENT_EXPORT partition_vector;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        struct vector_value_proxy
        {
            vector_value_proxy(hpx::vector<T>& v, std::size_t index)
              : v_(v), index_(index)
            {}

            operator T() const
            {
                return v_.get_value(index_);
            }

            template <typename T_>
            vector_value_proxy& operator=(T_ && value)
            {
                v_.set_value(index_, std::forward<T_>(value));
                return *this;
            }

            vector<T>& v_;
            std::size_t index_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the local iterator functionality for
    /// the partitioned backend of a hpx::vector.
    template <typename T>
    class local_vector_iterator
      : public boost::iterator_facade<
            local_vector_iterator<T>, T, std::random_access_iterator_tag
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                local_vector_iterator<T>, T, std::random_access_iterator_tag
            > base_type;

    public:
        // constructors
        local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        local_vector_iterator(partition_vector<T> partition,
                size_type local_index)
          : partition_(partition), local_index_(local_index)
        {}

        typedef typename std::vector<T>::iterator base_iterator_type;
        typedef typename std::vector<T>::const_iterator base_const_iterator_type;

        ///////////////////////////////////////////////////////////////////////
        base_iterator_type base_iterator()
        {
            return partition_.get_ptr()->begin() + local_index_;
        }
        base_const_iterator_type base_iterator() const
        {
            return partition_.get_ptr()->begin() + local_index_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
        }

    protected:
        bool is_at_end() const
        {
            return !partition_ || local_index_ == size_type(-1);
        }

        friend class boost::iterator_core_access;

        bool equal(local_vector_iterator const& other) const
        {
            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return partition_.get_value(local_index_);
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

                return partition_.size() - local_index_;
            }

            if (is_at_end())
                return other_.local_index_ - other.partition_.size();

            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partition_vector<T>& get_client() { return partition_; }
        partition_vector<T> const& get_client() const { return partition_; }

        size_type get_local_index() const { return local_index_; }

    protected:
        // refer to a partition of the vector
        partition_vector<T> partition_;

        // local position in the referenced partition
        size_type local_index_;
    };

    template <typename T>
    class const_local_vector_iterator
      : public boost::iterator_facade<
            const_local_vector_iterator<T>, T const,
            std::random_access_iterator_tag
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                const_local_vector_iterator<T>, T const,
                std::random_access_iterator_tag
            > base_type;

    public:
        // constructors
        const_local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        const_local_vector_iterator(partition_vector<T> partition,
                size_type local_index)
          : partition_(partition), local_index_(local_index)
        {}

        typedef typename std::vector<T>::const_iterator base_iterator_type;
        typedef typename std::vector<T>::const_iterator base_const_iterator_type;

        ///////////////////////////////////////////////////////////////////////
        base_const_iterator_type base_iterator()
        {
            return partition_.get_ptr()->cbegin() + local_index_;
        }
        base_const_iterator_type base_iterator() const
        {
            return partition_.get_ptr()->cbegin() + local_index_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
        }

    protected:
        bool is_at_end() const
        {
            return !partition_ || local_index_ == size_type(-1);
        }

        friend class boost::iterator_core_access;

        bool equal(const_local_vector_iterator const& other) const
        {
            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return partition_.get_value(local_index_);
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

        std::ptrdiff_t distance_to(const_local_vector_iterator const& other) const
        {
            if (other.is_at_end())
            {
                if (is_at_end())
                    return 0;

                return partition_.size() - local_index_;
            }

            if (is_at_end())
                return other_.local_index_ - other.partition_.size();

            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partition_vector<T> const& get_client() const { return partition_; }
        size_type get_local_index() const { return local_index_; }

    protected:
        // refer to a partition of the vector
        partition_vector<T> partition_;

        // local position in the referenced partition
        size_type local_index_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implement the segmented iterator for the hpx::vector
    template <typename T, typename BaseIter>
    class segment_vector_iterator
      : public boost::iterator_adaptor<
            segment_vector_iterator<T, BaseIter>, BaseIter, T,
            std::random_access_iterator_tag
        >
    {
    private:
        typedef boost::iterator_adaptor<
            segment_vector_iterator<T, BaseIter>, BaseIter, T,
            std::random_access_iterator_tag
        > base_type;

    public:
        segment_vector_iterator(vector<T>* data, BaseIter const& it)
          : base_type(it), data_(data)
        {}

        vector<T>* get_data() { return data_; }
        vector<T> const* get_data() const { return data_; }

    private:
        vector<T>* data_;
    };

    template <typename T, typename BaseIter>
    class const_segment_vector_iterator
      : public boost::iterator_adaptor<
            const_segment_vector_iterator<T, BaseIter>, BaseIter, T const,
            std::random_access_iterator_tag
        >
    {
    private:
        typedef boost::iterator_adaptor<
            const_segment_vector_iterator<T, BaseIter>, BaseIter, T const,
            std::random_access_iterator_tag
        > base_type;

    public:
        const_segment_vector_iterator(vector<T> const* data, BaseIter const& it)
          : base_type(it), data_(data)
        {}

        vector<T> const* get_data() const { return data_; }

    private:
        vector<T> const* data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This class implements the iterator functionality for hpx::vector.
    template <typename T>
    class vector_iterator
      : public boost::iterator_facade<
            vector_iterator<T>, T, std::random_access_iterator_tag,
            detail::vector_value_proxy<T>
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                vector_iterator<T>, T, std::random_access_iterator_tag,
                detail::vector_value_proxy<T>
            > base_type;

    public:
        typedef typename vector<T>::segment_iterator segment_iterator;
        typedef typename vector<T>::local_iterator local_iterator;

        // constructors
        vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        vector_iterator(vector<T>* data, size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        vector<T>* get_data() { return data_; }
        vector<T> const* get_data() const { return data_; }

        size_type get_global_index() const { return global_index_; }

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
            return detail::vector_value_proxy<T>(*data_, global_index_);
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
    class const_vector_iterator
      : public boost::iterator_facade<
            const_vector_iterator<T>, T const,
            std::random_access_iterator_tag, T const
        >
    {
    private:
        typedef std::size_t size_type;
        typedef boost::iterator_facade<
                const_vector_iterator<T>, T const,
                std::random_access_iterator_tag, T const
            > base_type;

    public:
        typedef typename vector<T>::const_segment_iterator segment_iterator;
        typedef typename vector<T>::const_local_iterator local_iterator;

        // constructors
        const_vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        const_vector_iterator(vector<T> const* data, size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        vector<T> const* get_data() const { return data_; }
        size_type get_global_index() const { return global_index_; }

    protected:
        bool is_at_end() const
        {
            return data_ == 0 || global_index_ == size_type(-1);
        }

        friend class boost::iterator_core_access;

        bool equal(const_vector_iterator const& other) const
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

        std::ptrdiff_t distance_to(const_vector_iterator const& other) const
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
        vector<T> const* data_;

        // global position in the referenced vector
        size_type global_index_;
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
        static segment_iterator segment(iterator& iter)
        {
            return iter.get_data()->get_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator& iter)
        {
            return iter.get_data()->get_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator& seg_iter,
            local_iterator& local_iter)
        {
            vector<T>* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter.base()->partition_, 0);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter.base()->partition_,
                seg_iter.base()->size_);
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
        static segment_iterator segment(iterator& iter)
        {
            return iter.get_data()->get_const_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator const& iter)
        {
            return iter.get_data()->get_const_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator const& seg_iter,
            local_iterator const& local_iter)
        {
            vector<T> const* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter.base()->partition_, 0);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator const& seg_iter)
        {
            return local_iterator(seg_iter.base()->partition_,
                seg_iter.base()->size_);
        }
    };
}}

#endif //  SEGMENTED_ITERATOR_HPP
