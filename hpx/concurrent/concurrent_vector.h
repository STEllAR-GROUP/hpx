//
//  concurrent_vector.h
//  Chilli Source
//  Created by Scott Downie on 19/09/2015.
//
//  The MIT License (MIT)
//
//  Copyright (c) 2012 Tag Games Limited
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

#ifndef _CHILLISOURCE_CORE_CONTAINER_CONCURRENTVECTOR_H_
#define _CHILLISOURCE_CORE_CONTAINER_CONCURRENTVECTOR_H_

#include <ChilliSource/Core/Container/concurrent_vector_const_forward_iterator.h>
#include <ChilliSource/Core/Container/concurrent_vector_const_reverse_iterator.h>
#include <ChilliSource/Core/Container/concurrent_vector_forward_iterator.h>
#include <ChilliSource/Core/Container/concurrent_vector_reverse_iterator.h>

#include <atomic>
#include <vector>

namespace ChilliSource
{
    namespace Core
    {
        //------------------------------------------------------------------------
        /// Concurrent vector is a thread safe dynamic array implementation.
        /// It presevers the integrity of the array when accessing from different
        /// threads and also from changes to the array while iterating
        ///
        /// NOTE: This class syntax mimics STL and therefore doe not use the CS coding
        /// standards.
        ///
        /// @author S Downie
        //------------------------------------------------------------------------
        template <typename TType> class concurrent_vector
        {
        public:
            
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using iterator = concurrent_vector_forward_iterator<TType>;
            using const_iterator = concurrent_vector_const_forward_iterator<TType>;
            using reverse_iterator = concurrent_vector_reverse_iterator<TType>;
            using const_reverse_iterator = concurrent_vector_const_reverse_iterator<TType>;
            
            //--------------------------------------------------------------------
            /// Constructor default
            ///
            /// @author S Downie
            //--------------------------------------------------------------------
            concurrent_vector();
            //--------------------------------------------------------------------
            /// Construct from initialiser list
            ///
            /// @author S Downie
            ///
            /// @param Initialiser list
            //--------------------------------------------------------------------
            concurrent_vector(std::initializer_list<TType>&& in_initialObjects);
            //--------------------------------------------------------------------
            /// Copy constructor that creates this as a copy of the given vector
            ///
            /// @author S Downie
            ///
            /// @param Vector to copy
            //--------------------------------------------------------------------
            concurrent_vector(const concurrent_vector& in_toCopy);
            //--------------------------------------------------------------------
            /// Copy assignment that creates this as a copy of the given vector
            ///
            /// @author S Downie
            ///
            /// @param Vector to copy
            ///
            /// @return This as a copy
            //--------------------------------------------------------------------
            concurrent_vector& operator=(const concurrent_vector& in_toCopy);
            //--------------------------------------------------------------------
            /// Move constructor that transfers ownership from the given vector
            ///
            /// @author S Downie
            ///
            /// @param Vector to move
            //--------------------------------------------------------------------
            concurrent_vector(concurrent_vector&& in_toMove);
            //--------------------------------------------------------------------
            /// Move assignment that transfers ownership from the given vector
            ///
            /// @author S Downie
            ///
            /// @param Vector to move
            ///
            /// @return This having taken ownership of the given vector
            //--------------------------------------------------------------------
            concurrent_vector& operator=(concurrent_vector&& in_toMove);
            //--------------------------------------------------------------------
            /// Push the object onto the back of the array. Unlike STL this does
            /// not invalidate any iterators
            ///
            /// @author S Downie
            ///
            /// @param Object to add
            //--------------------------------------------------------------------
            void push_back(TType&& in_object);
            //--------------------------------------------------------------------
            /// Push the object onto the back of the array. Unlike STL this does
            /// not invalidate any iterators
            ///
            /// @author S Downie
            ///
            /// @param Object to add
            //--------------------------------------------------------------------
            void push_back(const TType& in_object);
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object in the first element of the array (undefined if empty)
            //--------------------------------------------------------------------
            TType& front();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object in the first element of the array (undefined if empty)
            //--------------------------------------------------------------------
            const TType& front() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object in the last element of the array (undefined if empty)
            //--------------------------------------------------------------------
            TType& back();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The object in the last element of the array (undefined if empty)
            //--------------------------------------------------------------------
            const TType& back() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Index
            ///
            /// @return The object at the given index of the array (undefined if out of bounds)
            //--------------------------------------------------------------------
            TType& at(size_type in_index);
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Index
            ///
            /// @return The object at the given index of the array (undefined if out of bounds)
            //--------------------------------------------------------------------
            const TType& at(size_type in_index) const;
            //--------------------------------------------------------------------
            /// Lock the array which prevents its contents being modified unsafely.
            ///
            /// @author S Downie
            //--------------------------------------------------------------------
            void lock();
            //--------------------------------------------------------------------
            /// Unlock the array.
            ///
            /// @author S Downie
            //--------------------------------------------------------------------
            void unlock();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return The number of items currently in the vector
            //--------------------------------------------------------------------
            size_type size() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Whether the vector is empty or not
            //--------------------------------------------------------------------
            bool empty() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector
            //--------------------------------------------------------------------
            iterator begin();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the end of the vector (the end being)
            /// after the last element
            //--------------------------------------------------------------------
            iterator end();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector
            //--------------------------------------------------------------------
            const_iterator begin() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the end of the vector (the end being)
            /// after the last element
            //--------------------------------------------------------------------
            const_iterator end() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector
            //--------------------------------------------------------------------
            const_iterator cbegin() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the end of the vector (the end being)
            /// after the last element
            //--------------------------------------------------------------------
            const_iterator cend() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the last element of the vector
            //--------------------------------------------------------------------
            reverse_iterator rbegin();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector (the beginning
            /// in this case being before the first element)
            //--------------------------------------------------------------------
            reverse_iterator rend();
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the last element of the vector
            //--------------------------------------------------------------------
            const_reverse_iterator rbegin() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector (the beginning
            /// in this case being before the first element)
            //--------------------------------------------------------------------
            const_reverse_iterator rend() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the last element of the vector
            //--------------------------------------------------------------------
            const_reverse_iterator crbegin() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @return Iterator pointing to the beginning of the vector (the beginning
            /// in this case being before the first element)
            //--------------------------------------------------------------------
            const_reverse_iterator crend() const;
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Index
            ///
            /// @return The object at the given index of the array (undefined if out of bounds)
            //--------------------------------------------------------------------
            TType& operator[](size_type in_index);
            //--------------------------------------------------------------------
            /// @author S Downie
            ///
            /// @param Index
            ///
            /// @return The object at the given index of the array (undefined if out of bounds)
            //--------------------------------------------------------------------
            const TType& operator[](size_type in_index) const;
            //--------------------------------------------------------------------
            /// Remove the object from the vector that is pointed to by the given
            /// iterator. Unlike STL this does not invalidate any iterators
            ///
            /// @author S Downie
            ///
            /// @param Iterator
            ///
            /// @return The next iterator.
            //--------------------------------------------------------------------
            iterator erase(const iterator& in_itErase);
            //--------------------------------------------------------------------
            /// Remove the object from the vector that is pointed to by the given
            /// iterator. Unlike STL this does not invalidate any iterators
            ///
            /// @author S Downie
            ///
            /// @param Iterator
            ///
            /// @return The next iterator.
            //--------------------------------------------------------------------
            reverse_iterator erase(const reverse_iterator& in_itErase);
            //--------------------------------------------------------------------
            /// Clears the vector. Unlike STL this does not invalidate any iterators
            ///
            /// @author S Downie
            //--------------------------------------------------------------------
            void clear();
            
        private:
            
            //--------------------------------------------------------------------
            /// Cleanup any elements that are marked for removal
            ///
            /// @author S Downie
            //--------------------------------------------------------------------
            void garbage_collect();
            
        private:
            
            std::vector<std::pair<TType, bool>> m_container;
            
            std::atomic<size_type> m_size;
            bool m_isLocked = false;
            bool m_requiresGC = false;
            
            std::recursive_mutex m_mutex;
            std::unique_lock<std::recursive_mutex> m_lock;
            s32 m_lockCount = 0;
        };
        
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>::concurrent_vector()
        : m_lock(m_mutex, std::defer_lock), m_size(0)
        {
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>::concurrent_vector(std::initializer_list<TType>&& in_initialObjects)
        : m_lock(m_mutex, std::defer_lock), m_size(0)
        {
            m_size = in_initialObjects.size();
            m_container.reserve(m_size);
            
            for(const auto& object : in_initialObjects)
            {
                m_container.push_back(std::make_pair(object, false));
            }
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>::concurrent_vector(const concurrent_vector& in_toCopy)
        {
            m_size = in_toCopy.m_size.load();
            m_container = in_toCopy.m_container;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>& concurrent_vector<TType>::operator=(const concurrent_vector<TType>& in_toCopy)
        {
            m_size = in_toCopy.m_size.load();
            m_container = in_toCopy.m_container;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>::concurrent_vector(concurrent_vector&& in_toMove)
        {
            m_size = in_toMove.m_size.load();
            in_toMove.m_size = 0;
            m_container = std::move(in_toMove.m_container);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> concurrent_vector<TType>& concurrent_vector<TType>::operator=(concurrent_vector<TType>&& in_toMove)
        {
            m_size = in_toMove.m_size.load();
            in_toMove.m_size = 0;
            m_container = std::move(in_toMove.m_container);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::push_back(TType&& in_object)
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            m_container.push_back(std::make_pair(std::forward<TType>(in_object), false));
            m_size++;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::push_back(const TType& in_object)
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            m_container.push_back(std::make_pair(in_object, false));
            m_size++;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> TType& concurrent_vector<TType>::front()
        {
            return at(0);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> const TType& concurrent_vector<TType>::front() const
        {
            return at(0);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> TType& concurrent_vector<TType>::back()
        {
            return at(size()-1);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> const TType& concurrent_vector<TType>::back() const
        {
            return at(size()-1);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> TType& concurrent_vector<TType>::at(size_type in_index)
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            if(m_requiresGC == false)
            {
                return m_container.at(in_index).first;
            }
            else
            {
                size_type size = m_container.size();
                difference_type count = -1;
                size_type index = 0;
                for(size_type i=0; i<size; ++i)
                {
                    if(m_container[i].second == false)
                    {
                        if(++count == in_index)
                        {
                            index = i;
                            break;
                        }
                    }
                }
                
                return m_container[index].first;
            }
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> const TType& concurrent_vector<TType>::at(size_type in_index) const
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            if(m_requiresGC == false)
            {
                return m_container.at(in_index).first;
            }
            else
            {
                size_type size = m_container.size();
                difference_type count = -1;
                size_type index = 0;
                for(size_type i=0; i<size; ++i)
                {
                    if(m_container[i].second == false)
                    {
                        if(++count == in_index)
                        {
                            index = i;
                            break;
                        }
                    }
                }
                
                return m_container[index].first;
            }
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::lock()
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            
            if (m_lockCount == 0)
            {
                m_lock.lock();
            }
            
            m_lockCount++;
            m_isLocked = true;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::unlock()
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            
            m_isLocked = false;
            m_lockCount--;
            
            if (m_lockCount == 0)
            {
                garbage_collect();
                m_lock.unlock();
            }
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::size_type concurrent_vector<TType>::size() const
        {
            return m_size;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> bool concurrent_vector<TType>::empty() const
        {
            return m_size == 0;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::iterator concurrent_vector<TType>::begin()
        {
            return iterator(&m_container, &m_mutex);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::iterator concurrent_vector<TType>::end()
        {
            return iterator(&m_container, &m_mutex) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_iterator concurrent_vector<TType>::begin() const
        {
            return const_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex));
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_iterator concurrent_vector<TType>::end() const
        {
            return const_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex)) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_iterator concurrent_vector<TType>::cbegin() const
        {
            return const_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex));
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_iterator concurrent_vector<TType>::cend() const
        {
            return const_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex)) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::reverse_iterator concurrent_vector<TType>::rbegin()
        {
            return reverse_iterator(&m_container, &m_mutex);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::reverse_iterator concurrent_vector<TType>::rend()
        {
            return reverse_iterator(&m_container, &m_mutex) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_reverse_iterator concurrent_vector<TType>::rbegin() const
        {
            return const_reverse_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex));
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_reverse_iterator concurrent_vector<TType>::rend() const
        {
            return const_reverse_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex)) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_reverse_iterator concurrent_vector<TType>::crbegin() const
        {
            return const_reverse_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex));
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> typename concurrent_vector<TType>::const_reverse_iterator concurrent_vector<TType>::crend() const
        {
            return const_reverse_iterator(&m_container, const_cast<std::recursive_mutex*>(&m_mutex)) + size();
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> TType& concurrent_vector<TType>::operator[](size_type in_index)
        {
            return at(in_index);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> const TType& concurrent_vector<TType>::operator[](size_type in_index) const
        {
            return at(in_index);
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
		template <typename TType> concurrent_vector_forward_iterator<TType>  concurrent_vector<TType>::erase(const concurrent_vector_forward_iterator<TType>& in_itErase)
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            if(m_isLocked == false)
            {
                m_container.erase(m_container.begin() + in_itErase.get_index());
            }
            else
            {
                m_container[in_itErase.get_index()].second = true;
                m_requiresGC = true;
            }
            
            m_size--;
            
            auto itCopy = in_itErase;
            ++itCopy;
            return itCopy;
        }
        //--------------------------------------------------------------------
		/// This uses the 
        //--------------------------------------------------------------------
		template <typename TType> concurrent_vector_reverse_iterator<TType> concurrent_vector<TType>::erase(const concurrent_vector_reverse_iterator<TType>& in_itErase)
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            if(m_isLocked == false)
            {
                m_container.erase(m_container.begin() + in_itErase.get_index());
            }
            else
            {
                m_container[in_itErase.get_index()].second = true;
                m_requiresGC = true;
            }
            
            m_size--;
            
            auto itCopy = in_itErase;
            ++itCopy;
            return itCopy;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::clear()
        {
            std::unique_lock<std::recursive_mutex> scopedLock(m_mutex);
            if(m_isLocked == false)
            {
                m_container.clear();
                
            }
            else
            {
                for(auto& object : m_container)
                {
                    object.second = true;
                }
                
                m_requiresGC = true;
            }
            
            m_size = 0;
        }
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        template <typename TType> void concurrent_vector<TType>::garbage_collect()
        {
            for(auto it = m_container.begin(); it != m_container.end(); )
            {
                if(it->second == true)
                {
                    it = m_container.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            
            m_requiresGC = false;
        }
    }
}

#endif
