//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2012 Scott Downie, Tag Games Limited
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
//  vector_reverse_iterator.h
//  Chilli Source
//  Created by Scott Downie on 19/09/2014.
//
//  The MIT License (MIT)
//
//  Copyright (c) 2014 Tag Games Limited
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#ifndef _HPX_CONCURRENT_VECTOR_REVERSE_ITERATOR_H_
#define _HPX_CONCURRENT_VECTOR_REVERSE_ITERATOR_H_

#include <hpx/config.hpp>
#include <hpx/lcos/local/recursive_mutex.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <iterator>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx { namespace concurrent
{
    //--------------------------------------------------------------------
    /// Reverse iterator for the concurrent vector class that is read only
    ///
    /// @author S Downie
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    class vector_reverse_iterator
      : public util::iterator_facade<
            vector_reverse_iterator<TType, Allocator>,
            TType, std::forward_iterator_tag>
    {
        typedef util::iterator_facade<
                vector_reverse_iterator<TType, Allocator>,
                TType, std::forward_iterator_tag
            > base_type;

        typedef hpx::lcos::local::recursive_mutex recursive_mutex;
        typedef std::unique_lock<recursive_mutex> unique_lock;

        typedef typename std::allocator_traits<Allocator>::
                template rebind_alloc<std::pair<TType, bool> >
            allocator_type;

    public:
        typedef typename base_type::difference_type difference_type;

        //--------------------------------------------------------------------
        /// Constructor
        ///
        /// @author S Downie
        ///
        /// @param Data structure to iterate over
        /// @param Mutex used to protect the underlying iterable
        //--------------------------------------------------------------------
        vector_reverse_iterator(
                std::vector<std::pair<TType, bool>, allocator_type>* in_iterable,
                recursive_mutex* in_iterable_mutex, difference_type index = 0)
          : iterable_(in_iterable)
          , iterable_mutex_(in_iterable_mutex)
        {
            iterable_index_ =
                find_previous_occupied_index(iterable_->size() - index);
        }
        //--------------------------------------------------------------------
        /// Copy constructor that creates this as a copy of the given iterator
        ///
        /// @author S Downie
        ///
        /// @param iterator to copy
        //--------------------------------------------------------------------
        vector_reverse_iterator(const vector_reverse_iterator& rhs)
          : iterable_(rhs.iterable_)
          , iterable_mutex_(rhs.iterable_mutex_)
          , iterable_index_(rhs.iterable_index_)
        {
        }
        //--------------------------------------------------------------------
        /// Copy assignment that creates this as a copy of the given iterator
        ///
        /// @author S Downie
        ///
        /// @param iterator to copy
        ///
        /// @return This as a copy
        //--------------------------------------------------------------------
        vector_reverse_iterator& operator=(const vector_reverse_iterator& rhs)
        {
            iterable_ = rhs.iterable_;
            iterable_mutex_ = rhs.iterable_mutex_;
            iterable_index_ = rhs.iterable_index_;

            return *this;
        }
        //--------------------------------------------------------------------
        /// Move constructor that transfers ownership from the given iterator
        ///
        /// @author S Downie
        ///
        /// @param iterator to move
        //--------------------------------------------------------------------
        vector_reverse_iterator(vector_reverse_iterator&& rhs)
          : iterable_(rhs.iterable_)
          , iterable_mutex_(rhs.iterable_mutex_)
          , iterable_index_(rhs.iterable_index_)
        {
            rhs.iterable_ = nullptr;
            rhs.iterable_mutex_ = nullptr;
            rhs.iterable_index_ = 0;
        }
        //--------------------------------------------------------------------
        /// Move assignment that transfers ownership from the given iterator
        ///
        /// @author S Downie
        ///
        /// @param iterator to move
        ///
        /// @return This having taken ownership of the given iterator
        //--------------------------------------------------------------------
        vector_reverse_iterator& operator=(
            vector_reverse_iterator&& rhs)
        {
            iterable_ = rhs.iterable_;
            rhs.iterable_ = nullptr;
            iterable_mutex_ = rhs.iterable_mutex_;
            rhs.iterable_mutex_ = nullptr;
            iterable_index_ = rhs.iterable_index_;
            rhs.iterable_index_ = 0;

            return *this;
        }
        //--------------------------------------------------------------------
        /// NOTE: This is an internal method used to query the element index
        /// pointed to by the iterator
        ///
        /// @author S Downie
        ///
        /// @param Element index that the iterate currently points to
        //--------------------------------------------------------------------
        difference_type get_index() const
        {
            return iterable_index_;
        }

    private:
        //--------------------------------------------------------------------
        /// Find the previous element index (exclusive of given index) that is
        /// not flagged for removal
        ///
        /// @author S Downie
        ///
        /// @param Index to begin at (exclusive)
        ///
        /// @return Index of previous element (or start if none)
        //--------------------------------------------------------------------
        difference_type
        find_previous_occupied_index(difference_type in_begin_index) const
        {
            if (in_begin_index == 0)
            {
                return -1;
            }

            in_begin_index--;

            unique_lock l(*iterable_mutex_);
            for (auto i = in_begin_index; i >= 0; --i)
            {
                if (!(*iterable_)[i].second)
                {
                    return i;
                }
            }

            return -1;
        }


        ///////////////////////////////////////////////////////////////////////
        friend class util::iterator_core_access;

        void increment()
        {
            iterable_index_ = find_previous_occupied_index(iterable_index_);
        }

        typename base_type::reference dereference() const
        {
            unique_lock l(*iterable_mutex_);
            return (*iterable_)[iterable_index_].first;
        }

        bool equal(vector_reverse_iterator const& rhs) const
        {
            return iterable_index_ != rhs.iterable_index_;
        }

    private:
        std::vector<std::pair<TType, bool>, allocator_type>* iterable_;
        recursive_mutex* iterable_mutex_;
        difference_type iterable_index_ = 0;
    };
}}

#endif
