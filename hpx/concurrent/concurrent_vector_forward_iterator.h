//
//  concurrent_vector_forward_iterator.h
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

#ifndef _CHILLISOURCE_CORE_CONTAINER_CONCURRENTVECTORFORWARDITERATOR_H_
#define _CHILLISOURCE_CORE_CONTAINER_CONCURRENTVECTORFORWARDITERATOR_H_

#include <vector>

namespace ChilliSource
{
    namespace Core
    {
        //------------------------------------------------------------------------
        /// Forward iterator for the concurrent vector class.
        ///
        /// @author S Downie
        //------------------------------------------------------------------------
        template <typename TType> class concurrent_vector_forward_iterator
        {
        public:
            
            using difference_type = typename std::iterator<std::forward_iterator_tag, TType>::difference_type;
            
            //------------------------------------------------------------------------
            /// Constructor
            ///
            /// @author S Downie
            ///
            /// @param Data structure to iterate over
            /// @param Mutex used to protect the underlying iterable
            //------------------------------------------------------------------------
            concurrent_vector_forward_iterator(std::vector<std::pair<TType, bool>>* in_iterable, std::recursive_mutex* in_iterableMutex)
            : m_iterable(in_iterable), m_iterableMutex(in_iterableMutex)
            {
                m_iterableIndex = find_next_occupied_index(0);
            }
            //--------------------------------------------------------------------
            /// Copy constructor that creates this as a copy of the given iterator
            ///
            /// @author S Downie
            ///
            /// @param iterator to copy
            //--------------------------------------------------------------------
            concurrent_vector_forward_iterator(const concurrent_vector_forward_iterator& in_toCopy)
            : m_iterableIndex(in_toCopy.m_iterableIndex), m_iterable(in_toCopy.m_iterable), m_iterableMutex(in_toCopy.m_iterableMutex)
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
            concurrent_vector_forward_iterator& operator=(const concurrent_vector_forward_iterator& in_toCopy)
            {
                m_iterableIndex = in_toCopy.m_iterableIndex;
                m_iterable = in_toCopy.m_iterable;
                m_iterableMutex = in_toCopy.m_iterableMutex;
                
                return *this;
            }
            //--------------------------------------------------------------------
            /// Move constructor that transfers ownership from the given iterator
            ///
            /// @author S Downie
            ///
            /// @param iterator to move
            //--------------------------------------------------------------------
            concurrent_vector_forward_iterator(concurrent_vector_forward_iterator&& in_toMove)
            : m_iterableIndex(in_toMove.m_iterableIndex), m_iterable(in_toMove.m_iterable), m_iterableMutex(in_toMove.m_iterableMutex)
            {
                in_toMove.m_iterableIndex = 0;
                in_toMove.m_iterable = nullptr;
                in_toMove.m_iterableMutex = nullptr;
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
            concurrent_vector_forward_iterator& operator=(concurrent_vector_forward_iterator&& in_toMove)
            {
                m_iterableIndex = in_toMove.m_iterableIndex;
                in_toMove.m_iterableIndex = 0;
                m_iterable = in_toMove.m_iterable;
                in_toMove.m_iterable = nullptr;
                m_iterableMutex = in_toMove.m_iterableMutex;
                in_toMove.m_iterableMutex = nullptr;
                
                return *this;
            }
            //------------------------------------------------------------------------
            /// Moves the iterator to point to the next element in the vector
            ///
            /// @author S Downie
            ///
            /// @return Updated iterator
            //------------------------------------------------------------------------
            concurrent_vector_forward_iterator& operator++()
            {
                m_iterableIndex = find_next_occupied_index(m_iterableIndex + 1);
                return *this;
            }
            //------------------------------------------------------------------------
            /// Moves the iterator to point to the element in the vector at the given
            /// offset from the current iterator
            ///
            /// @author S Downie
            ///
            /// @param Offset
            ///
            /// @return Updated iterator
            //------------------------------------------------------------------------
            concurrent_vector_forward_iterator& operator+=(difference_type in_stride)
            {
                m_iterableIndex = find_next_occupied_index(m_iterableIndex + in_stride);
                return *this;
            }
            //------------------------------------------------------------------------
            /// Create an iterator that points to the element at the given offset from
            /// this iterator
            ///
            /// @author S Downie
            ///
            /// @param Offset
            ///
            /// @return New iterator
            //------------------------------------------------------------------------
            concurrent_vector_forward_iterator operator+(difference_type in_stride)
            {
                auto iterableIndex = find_next_occupied_index(m_iterableIndex + in_stride);
                return concurrent_vector_forward_iterator(m_iterable, m_iterableMutex, iterableIndex);
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Pointer to the object pointed to by the iterator
            //------------------------------------------------------------------------
            TType* operator->()
            {
                std::unique_lock<std::recursive_mutex> scopedLock(*m_iterableMutex);
                return &((*m_iterable)[m_iterableIndex].first);
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Pointer to the object pointed to by the iterator
            //------------------------------------------------------------------------
            const TType* operator->() const
            {
                std::unique_lock<std::recursive_mutex> scopedLock(*m_iterableMutex);
                return &((*m_iterable)[m_iterableIndex].first);
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object pointed to by the iterator
            //------------------------------------------------------------------------
            TType& operator*()
            {
                std::unique_lock<std::recursive_mutex> scopedLock(*m_iterableMutex);
                return (*m_iterable)[m_iterableIndex].first;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object pointed to by the iterator
            //------------------------------------------------------------------------
            const TType& operator*() const
            {
                std::unique_lock<std::recursive_mutex> scopedLock(*m_iterableMutex);
                return (*m_iterable)[m_iterableIndex].first;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether the iterators are considered equal
            //------------------------------------------------------------------------
            bool operator==(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex == in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether the iterators are considered unequal
            //------------------------------------------------------------------------
            bool operator!=(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex != in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether this iterator points to an element after the given iterator
            //------------------------------------------------------------------------
            bool operator>(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex > in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether this iterator points to an element after or the same as the given iterator
            //------------------------------------------------------------------------
            bool operator>=(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex >= in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether this iterator points to an element before the given iterator
            //------------------------------------------------------------------------
            bool operator<(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex < in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Iterator to compare with
            ///
            /// @return Whether this iterator points to an element before or the same as the given iterator
            //------------------------------------------------------------------------
            bool operator<=(const concurrent_vector_forward_iterator& in_toCompare) const
            {
                return m_iterableIndex <= in_toCompare.m_iterableIndex;
            }
            //------------------------------------------------------------------------
            /// NOTE: This is an internal method used to query the element index
            /// pointed to by the iterator
            ///
            /// @author S Downie
            ///
            /// @param Element index that the iterate currently points to
            //------------------------------------------------------------------------
            difference_type get_index() const
            {
                return m_iterableIndex;
            }
            
        private:
            //------------------------------------------------------------------------
            /// Constructor
            ///
            /// @author S Downie
            ///
            /// @param Data structure to iterate over
            /// @param Mutex used to protect the underlying iterable
            /// @param Initial index
            //------------------------------------------------------------------------
            concurrent_vector_forward_iterator(std::vector<std::pair<TType, bool>>* in_iterable, std::recursive_mutex* in_iterableMutex, difference_type in_initialIndex)
            : m_iterable(in_iterable), m_iterableMutex(in_iterableMutex), m_iterableIndex(in_initialIndex)
            {
                
            }
            //------------------------------------------------------------------------
            /// Find the next element index (inclusive of given index) that is not
            /// flagged for removal
            ///
            /// @author S Downie
            ///
            /// @param Index to begin at (inclusive)
            ///
            /// @return Index of next element (or end if none)
            //------------------------------------------------------------------------
            difference_type find_next_occupied_index(difference_type in_beginIndex) const
            {
                std::unique_lock<std::recursive_mutex> scopedLock(*m_iterableMutex);
                auto size = m_iterable->size();
                
                for(std::size_t i = in_beginIndex; i < size; ++i)
                {
                    if((*m_iterable)[i].second == false)
                    {
                        return i;
                    }
                }
                
                return size;
            }
            
        private:
            
            difference_type m_iterableIndex = 0;
            std::vector<std::pair<TType, bool>>* m_iterable;
            std::recursive_mutex* m_iterableMutex;
        };
    }
}

#endif
