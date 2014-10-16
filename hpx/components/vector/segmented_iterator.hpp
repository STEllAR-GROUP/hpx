//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#ifndef SEGMENTED_ITERATOR_HPP
#define SEGMENTED_ITERATOR_HPP

/** @file hpx/components/vector/segmented_iterator.hpp
 *
 *  @brief This file contain the implementation of segmented vector iterator for
 *          the hpx::vector.
 *
 */

 //     The idea of these iterator is taken from
 //     http://afstern.org/matt/segmented.pdf with some modification
 //

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>

#include <cstdint>
#include <boost/integer.hpp>

#include <hpx/components/vector/chunk_vector_component.hpp>

namespace hpx
{
    /** @brief This class implement const iterator functionality for hpx::vector.
    *
    *   This contain the implementation of the all random access iterator API
    *    need. This Class also contain some additional API which is needed to
    *    iterate over segmented data structure.
    */
    template <typename T>
    class const_segmented_vector_iterator
    {
    public:
        typedef std::size_t size_type;

    private:
        // This typedef helps to call object of same class.
        typedef const_segmented_vector_iterator<T> self_type;
        typedef chunk_vector chunk_vector_client;

    public:
        //  This represent the return type of the local(), begin() and end() API
        //   which are important for segmented_vector iterator.
        typedef std::pair<chunk_vector_client, size_type> local_return_type;

    protected:
        //  For the following two typedefs refer to hpx::vector class
        typedef std::pair<chunk_vector_client, size_type> chunk_description_type;
        typedef std::vector<chunk_description_type> chunks_vector_type;

        typedef typename chunks_vector_type::const_iterator segment_iterator;

    public:
        // constructors
        const_segmented_vector_iterator()
          : local_index_(size_type(-1))
        {}

        const_segmented_vector_iterator(segment_iterator curr_chunk,
                size_type local_index)
          : curr_chunk_(curr_chunk),
            local_index_(local_index)
        {}

        /** @brief Copy Constructor
         *
         *  @param other   The const_segmented_vector_iterator object which
         *                  is to be copied
         */
        const_segmented_vector_iterator(self_type const& other)
          : curr_chunk_(other.curr_chunk_),
            local_index_(other.local_index_)
        {}

        /** @brief Copy one const_segmented_vector_iterator into other.
         *
         *  @param other The const_segmented_vector_iterator objects which
         *                is to be copied
         *
         *  @return This return the reference to the newly created
         *           const_segmented_vector_iterator
         */
        self_type& operator= (self_type const& other)
        {
            if (this != &other)
            {
                curr_chunk_ = other.curr_chunk_;
                local_index_ = other.local_index_;
                state_ = other.state_;
            }
            return *this;
        }

        /** @brief Compare the two iterators for equality.
         *
         *  @param other The iterator objects which is to be compared
         *
         *  @return Return true if both are equal, false otherwise
         */
        bool operator== (self_type const& other) const
        {
            return curr_chunk_ == other.curr_chunk_ &&
                   local_index_ == other.local_index_;
        }

        /** @brief Compare the two iterators for inequality.
         *
         *  @param other The iterator objects which is to be compared
         *
         *  @return Return false if both are equal, false otherwise
         */
        bool operator!= (self_type const& other) const
        {
            return !(*this == other);
        }

        /** @brief Dereferences the iterator and returns the value of the
         *          element.
         *
         *   If iterator is out of range of container then it cause undefined
         *      behavior.
         *
         *  @return Value in the element pointed by the iterator [Note like
         *           standard iterator it does not return reference, it just
         *           returns value]
         */
        T operator* () const
        {
            return curr_chunk_->first.get_value(local_index_);
        }

        /** @brief Increment the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the incremented const_segmented_vector_iterator object
         */
        self_type operator++ ()  // prefix behavior
        {
            if (++local_index_ == curr_chunk_->second)
            {
                // end of current chunk is reached
                ++curr_chunk_;
                local_index_ = 0;
            }

            // this condition does not cause function call hence it must
            // be first
            if(((curr_chunk_ + 1)->second).get() != invalid_id &&
                local_index_ >= chunk_vector_stubs::size_async(
                                            (curr_chunk_->second).get()
                                                                ).get()
                )
            {
                ++curr_chunk_;
                local_index_ = 0;
            }
            return *this;
        }

        /** @brief Increment the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original const_segmented_vector_iterator object
         */
        self_type operator ++ (int) // postfix behavior
        {
            // temp object should be return to simulate the postfix behavior
            self_type temp = *this;
            ++(*this);
            return temp;
        }

        /** @brief Decrement the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the decremented const_segmented_vector_iterator object
         */
        self_type operator -- () // prefix behavior
        {
            // if it is just first gid just decrement the local index
            if( local_index_ == 0)
            {
                if(curr_chunk_->first != 0)
                {
                    --curr_chunk_;
                    local_index_ =
                        ( chunk_vector_stubs::size_async(
                                        (curr_chunk_->second).get()
                                                         ).get() - 1
                         );
                }
            }
            else
            {
                --local_index_;
            }
            return *this;
        }

        /** @brief Decrement the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original const_segmented_vector_iterator object
         */
        self_type operator -- (int) // postfix behavior
        {
             // temp object should be return to simulate the postfix behavior
            self_type temp = *this;
            --(*this);
            return temp;
        }

        /** @brief Return the const_segmented_vector_iterator pointing to the
         *  position which is \a n units ahead of the current position.
         */
        self_type operator + (size_type n) const
        {
            // copying the current states of the iterator
            segment_iterator temp_curr_chunk_description_type = this->curr_chunk_;
            size_type temp_local_index = this->local_   index_;
            hpx::iter_state temp_state = this->state_;

            // temp variables
            id_type invalid_id;
            bool same_chunk = true;
            size_type size = 0;

            if(temp_state == invalid)
            {
                // calculate the length through which it is invalid
                size_type diff = (std::numeric_limits<size_type>::max() -
                                    temp_local_index + 1);

                if(n < diff )
                {
                    return self_type(temp_curr_chunk_description_type,
                                    (temp_local_index + n),
                                     temp_state);
                }
                else
                {
                    n = n - diff;
                    temp_local_index = 0;
                    temp_state = valid;
                }
            }

            // calculating the size of the first chunk
            size = chunk_vector_stubs::size_async(
                                    (temp_curr_chunk_description_type->second).get()
                                                  ).get()
                                     - temp_local_index;

            while( n >= size)
            {
                 // break this loop if this is previous to LAST gid
                 // i.e. last valid gid in the list
                if(((temp_curr_chunk_description_type + 1)->second).get() == invalid_id )
                    break;

                same_chunk = false;
                n = n - size;
                ++temp_curr_chunk_description_type;
                // calculate the size of current chunk
                size = chunk_vector_stubs::size_async(
                                    (temp_curr_chunk_description_type->second).get()
                                                            ).get();
            }
            if(same_chunk)
            {
                temp_local_index += n;
            }
            else
            {
                temp_local_index = n;
            }
            return self_type(temp_curr_chunk_description_type, temp_local_index, temp_state);
        } // end of a + n

        /** @brief Return the const_segmented_vector_iterator pointing to the
         *          position which is \a n units behind the current position.
         */
        self_type operator - (size_type n) const
        {
            // copying the current states of the iterator
            segment_iterator temp_curr_chunk_description_type = this->curr_chunk_;
            size_type temp_local_index = this->local_index_;
            hpx::iter_state temp_state = this->state_;

            // temp variables
            bool same_chunk = true;
            size_type size = 0;
            if(temp_state == invalid)
            {
                return self_type(temp_curr_chunk_description_type,
                                 (temp_local_index - n),
                                 temp_state );
            }
            else
            {
                //
                // this calculates remaining elements in current chunk
                //

                // this size tells how many need to go out side of current gid
                size = temp_local_index + 1;
                while (n >= size)
                {
                    // this condition is only met when iterator is going invalid
                    if(temp_curr_chunk_description_type->first == 0)
                    {
                        temp_state = invalid;
                        break;
                    }
                    same_chunk = false;
                    n = n - size;
                    --temp_curr_chunk_description_type;
                    size = chunk_vector_stubs::size_async(
                                    (temp_curr_chunk_description_type->second).get()
                                                          ).get();
                } // end of while
            } // end of else

            if (same_chunk)
            {
                temp_local_index -= n;
            }
            else
            {
                temp_local_index = size - (n + 1);
            }

            return self_type(temp_curr_chunk_description_type, temp_local_index, temp_state);
        } // end of a - n

//        // TODO this returning int64_t which has half range with size_t
//        boost::int64_t operator - (self_type const& other) const
//        {
//            if(this->curr_chunk_description_type_ == other.curr_chunk_description_type_)
//            {
//                return static_cast<boost::int64_t>(this->local_index_
//                                                   - other.local_index_);
//            }
//            else if(this->curr_chunk_description_type_ > other.curr_chunk_description_type_) // nswer is positive
//            {
//                std::size_t diff = diff_helper(other.curr_chunk_description_type_,
//                                                this->curr_chunk_description_type_);
//                // adding the part from (*this) chunk
//                diff = diff + (this->local_index_ + 1);
//                // subtracting extra part from from the other chunk
//                diff = diff - (other.local_index_ + 1);
//                // TODO this should be the exception not the assert
//                HPX_ASSERT( diff <= std::numeric_limits<boost::int64_t>::max());
//                return static_cast<boost::int64_t>(diff);
//            }
//            else if(this->curr_chunk_description_type_ < other.curr_chunk_description_type_) // nswer is negative
//            {
//                std::size_t diff = diff_helper(this->curr_chunk_description_type_,
//                                                other.curr_chunk_description_type_);
//                 // subtracting extra part from (*this) chunk
//                diff = diff - (this->local_index_ + 1);
//                 // adding the part from from the other chunk
//                diff = diff + (other.local_index_ + 1);
//                // TODO this should be the exception not the assert
//                HPX_ASSERT( diff <= std::numeric_limits<boost::int64_t>::max());
//                return static_cast<boost::int64_t>(diff);
//            }
//            else{HPX_ASSERT(0);}
//        } // end of a - b

        /** @brief Compare the two iterator for less than relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is less than
         *           other, false otherwise
         */
        bool operator < (self_type const& other) const
        {
             // if both are from diff gid
            if (curr_chunk_ < other.curr_chunk_description_type_)
            {
                return true;
            }

            // now if bot are from same gid
            if (curr_chunk_ == other.curr_chunk_description_type_)
            {
                // if both are same then check local index
                if (local_index_ < other.local_index_)
                {
                    return true;
                }
            }

            return false;
        } // End of <

        /** @brief Compare the two iterator for greater than relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is greater than
         *           other, false otherwise
         */
        bool operator > (self_type const& other) const
        {
            if (curr_chunk_ > other.curr_chunk_description_type_)
            {
                return true;
            }

            if (curr_chunk_ == other.curr_chunk_description_type_)
            {
                if (local_index_ > other.local_index_)
                {
                    return true;
                }
            }

            return false;
        } // End of >

        /** @brief Compare the two iterator for less than or equal to relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is less than or
         *           equal to the other, false otherwise
         */
        bool operator <= (self_type const& other) const
        {
            return (*this) < other || (*this) == other;
        } // End of <=

        /** @brief Compare the two iterator for greater than or equal to
         *          relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is greater than
         *           or equal to the other, false otherwise
         */
        bool operator >= (self_type const& other) const
        {
            return (*this) > other || (*this) == other;
        } // End of >=

        // OMPOUND ASSIGNMENT
        /** @brief Increment the const_segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the incremented object
         */
        self_type & operator+= (size_type n)
        {
            *this = *this + n;
            return *this;       // return self_type to make (a = (b += n)) work
        } // end of +=

        /** @brief Decrement the const_segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the decremented object
         */
        self_type & operator-= (size_type n)
        {
            *this = *this - n;
            return *this;       // return self_type to make (a = (b -= n)) work
        } // end of +=

        // FFSET DEREFERENCE
        /** @brief Dereferences the iterator which is at \a n position ahead of the
         *          current iterator position and returns the value of the element.
         *
         *  @return Value in the element which is at n position ahead of the
         *           current iterator
         */
        T operator[](size_type n) const
        {
            self_type temp = *this;
            temp = temp + n;
            return *temp;
        }

    protected:
        //  This is the iterator pointing the the vector which stores the
        //   (base_index, gid) pair. (For base_index and gid refer to
        //   hpx::vector class in hpx/components/vector/vector.hpp). This
        //   actually represent to which gid pair our current iterator
        //   position is pointing.
        segment_iterator curr_chunk_;

        // This represent the local position in the current base_gid pair
        //  pointed by the curr_chunk_description_type_ (defined above) iterator.
        size_type local_index_;
    }; // End of const_segmented_vector_iterator

    ///////////////////////////////////////////////////////////////////////////
    namespace traits
    {
        template <typename T>
        struct segmented_iterator_traits<
            const_segmented_vector_iterator<T> >
        {
            typedef boost::mpl::true_ is_segmented_iterator;

            typedef const_segmented_vector_iterator<T> iterator;
            typedef typename iterator::segment_iterator segment_iterator;
            typedef typename iterator::local_return_type local_iterator;

            //      Conceptually this function is suppose to denote which segment,
            //  the iterator is currently pointing to (i.e. just global iterator).
            //      As we are having the gid and base index in a pair and we have a
            //  std::vector which store these pairs. So we are returning the
            //  const_iterator so that we can have access to both gid and and
            //  base_index and the iterator is also comparable against == and !=
            //  operator
            static segment_iterator segment(iterator const& seg_iter)
            {
                return seg_iter.curr_chunk_description_type_;
            }

            //      This function should specify which is the current segment and
            //  in that exact position to which local iterator is pointing.
            //      Now we are returning the pair of the gid and local index to
            //  represent the position.
            static local_iterator local(iterator const& seg_iter)
            {
                return std::make_pair((seg_iter.curr_chunk_description_type_)->second,
                    seg_iter.local_index_);
            }

            static iterator compose(segment_iterator, local_iterator);

            //      This function should specify the local iterator which is at the
            //  beginning of the chunk.
            //      We are returning the pair of the gid and local index as 0. Though
            //  we can get the local iterator at the beginning if we are having the
            //  gid but we are returning this to have same API for each algorithm.
            static local_iterator begin(segment_iterator const& chunk_chunk_description_type)
            {
                return std::make_pair(chunk_chunk_description_type->second, 0);
            }

            //      This function should specify the local iterator which is at the
            //  end of the chunk.
            //
            //      We are returning the pair of the gid and local index as size.
            //  Though we can get the local iterator at the end if we are having the
            //  gid but we are returning this to have same API for each algorithm.
            static local_iterator end(segment_iterator const& chunk_chunk_description_type)
            {
                return std::make_pair(
                        chunk_chunk_description_type->second,
                        chunk_vector_stubs::size_async(
                                        (chunk_chunk_description_type->second).get()
                                                             ).get()
                                     );
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class segmented_vector_iterator
      : public const_segmented_vector_iterator<T>
    {
    private:
        typedef segmented_vector_iterator<T> self_type;
        typedef const_segmented_vector_iterator<T> base_type;

        segmented_vector_iterator(base_type const& other)
          : base_type(other)
        {}

    public:
        typedef typename base_type::chunks_vector_type chunks_vector_type;
        typedef typename base_type::size_type size_type;

        typedef typename chunks_vector_type::iterator segment_iterator;

        segmented_vector_iterator()
          : base_type()
        {}

        segmented_vector_iterator(segment_iterator curr_chunk,
                size_type local_index)
          : base_type(curr_chunk, local_index)
        {}

        /** @brief Copy Constructor
         *
         *  @param other   The segmented_vector_iterator object which
         *                  is to be copied
         */
        segmented_vector_iterator(self_type const& other)
          : base_type(other)
        {}

        /** @brief Copy one segmented_vector_iterator into other.
         *
         *  @param other The segmented_vector_iterator objects which
         *                is to be copied
         *
         *  @return This return the reference to the newly created
         *           segmented_vector_iterator
         */
        self_type & operator = (self_type const& other)
        {
            base_type::operator=(other);
            return *this;
        }

        /** @brief Return the segmented_vector_iterator pointing to the
         *  position which is \a n units ahead of the current position.
         */
        self_type operator+ (size_type n) const
        {
            base_type temp = *this;
            return self_type(temp + n);
        }

        /** @brief Return the segmented_vector_iterator pointing to the
         *          position which is \a n units behind the current position.
         */
        self_type operator- (size_type n) const
        {
            base_type temp = *this;
            return self_type(temp - n);
        }

        /** @brief Increment the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the incremented segmented_vector_iterator object
         */
        self_type operator++ ()  // prefix behavior
        {
            base_type::operator++();
            return *this;
        }

        /** @brief Increment the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original segmented_vector_iterator object
         */
        self_type operator++(int) // postfix behavior
        {
            // return_temp object should be return to simulate the postfix behavior
            base_type temp = *this;
            ++*this;
            return temp;
        }

        /** @brief Decrement the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the decremented segmented_vector_iterator object
         */
        self_type operator-- () // prefix behavior
        {
            base_type::operator--();
            return *this;
        }

        /** @brief Decrement the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original segmented_vector_iterator object
         */
        self_type operator -- (int) // postfix behavior
        {
            // eturn_temp object should be return to simulate the postfix behavior
            base_type temp = *this;
            --*this;
            return temp;
        }

        /** @brief Increment the segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the incremented object
         */
        self_type & operator+= (size_type n)
        {
            *this = *this + n;
            return *this;       // return self_type to make (a = (b += n)) work
        } // end of +=

        /** @brief Decrement the segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the decremented object
         */
        self_type & operator-= (size_type n)
        {
            *this = *this - n;
            return *this;       // return self_type to make (a = (b -= n)) work
        } // end of +=
    }; // end of segmented_vector_iterator

    ///////////////////////////////////////////////////////////////////////////
    namespace traits
    {
        template <typename T>
        struct segmented_iterator_traits<segmented_vector_iterator<T> >
        {
            typedef boost::mpl::true_ is_segmented_iterator;

            typedef segmented_vector_iterator<T> iterator;
            typedef typename iterator::segment_iterator segment_iterator;
            typedef typename iterator::local_return_type local_iterator;

            //      Conceptually this function is suppose to denote which segment,
            //  the iterator is currently pointing to (i.e. just global iterator).
            //      As we are having the gid and base index in a pair and we have a
            //  std::vector which store these pairs. So we are returning the
            //  const_iterator so that we can have access to both gid and and
            //  base_index and the iterator is also comparable against == and !=
            //  operator
            static segment_iterator segment(iterator const& seg_iter)
            {
                return seg_iter.curr_chunk_description_type_;
            }

            //      This function should specify which is the current segment and
            //  in that exact position to which local iterator is pointing.
            //      Now we are returning the pair of the gid and local index to
            //  represent the position.
            static local_iterator local(iterator const& seg_iter)
            {
                return std::make_pair((seg_iter.curr_chunk_description_type_)->second,
                    seg_iter.local_index_);
            }

            static iterator compose(segment_iterator, local_iterator);

            //      This function should specify the local iterator which is at the
            //  beginning of the chunk.
            //      We are returning the pair of the gid and local index as 0. Though
            //  we can get the local iterator at the beginning if we are having the
            //  gid but we are returning this to have same API for each algorithm.
            static local_iterator begin(segment_iterator const& chunk_chunk_description_type)
            {
                return std::make_pair(chunk_chunk_description_type->second, 0);
            }

            //      This function should specify the local iterator which is at the
            //  end of the chunk.
            //
            //      We are returning the pair of the gid and local index as size.
            //  Though we can get the local iterator at the end if we are having the
            //  gid but we are returning this to have same API for each algorithm.
            static local_iterator end(segment_iterator const& chunk_chunk_description_type)
            {
                return std::make_pair(
                        chunk_chunk_description_type->second,
                        chunk_vector_stubs::size_async(
                                        (chunk_chunk_description_type->second).get()
                                                             ).get()
                                     );
            }
        };
    }
} // end of hpx namespace

#endif //  SEGMENTED_ITERATOR_HPP
