//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SEGMENTED_ITERATOR_HPP
#define SEGMENTED_ITERATOR_HPP

/** @file hpx/components/vector/segmented_iterator.hpp
 *
 *  @brief This file contain the implementation of segmented vector iterator for
 *          the hpx::vector.
 *
 */


// PROGRAMMER DOCUMENTATION:
 //     The idea of these iterator is taken from
 //     http://lafstern.org/matt/segmented.pdf with some modification
 //


#include <hpx/include/util.hpp>

// headers for checking the ranges of the Datatypes
#include <cstdint>
#include <boost/integer.hpp>

#include <hpx/components/vector/chunk_vector_component.hpp>

#define VALUE_TYPE double

namespace hpx
{
    //For checking the state of the iterator

    //INVALID STATE:  Represent the iterator goes below the 0'th position on the
    // first gid in the vector which actually mean that -ve in the overall index

    /**  @brief This enum define the iterator state. */
    enum iter_state{
        invalid = 0, /**< This represent the iterator is in invalid state */
        valid = 1      /**< This represent the iterator is in valid state */
        };



    /** @brief This class implement const iterator functionality for hpx::vector.
    *
    *   This contain the implementation of the all random access iterator API
    *    need. This Class also contain some additional API which is needed to
    *    iterate over segmented data structure.
    */
    class const_segmented_vector_iterator
    {
    public:
        typedef std::size_t             size_type;

    private:
        // This typedef helps to call object of same class.
        typedef const_segmented_vector_iterator     self_type;
        typedef hpx::naming::id_type                hpx_id;
        typedef hpx::lcos::shared_future<hpx_id>    hpx_id_shared_future;
        typedef hpx::stubs::chunk_vector            chunk_vector_stubs;

        //PROGRAMMER DOCUMENTATION:
        //  This represent the return type of the local(), begin() and end() API
        //   which are important for segmented_vector iterator.
        typedef std::pair<hpx_id_shared_future, size_type > local_return_type;

    protected:
        //PROGRAMMER DOCUMENTATION:
        //  For the following two typedefs refer to hpx::vector class in
        //  hpx/components/vector/vector.hpp
        typedef std::pair<size_type, hpx_id_shared_future>  bfg_pair;
        typedef std::vector< bfg_pair >                     vector_type;


//        // PROGRAMMER DOCUMENTATION: This is the helper function for
//        std::size_t diff_helper(vector_type::const_iterator src,
//                                vector_type::const_iterator dest) const
//        {
//            std::size_t diff = 0;
//            //Calculating the total number of element in chunk starting from
//            //src to (dest - 1). The dest - 1 reduce one calculation of
//            // size_of_chunk (as that size is given by LAST_OBJECT.local_index)
//            while(src != dest)
//            {
//                diff += base_type::size_async((src->second).get()).get();
//                src++;
//            }
//            return diff;
//        }

    public:
        //
        // constructors
        //
        const_segmented_vector_iterator(){}
        const_segmented_vector_iterator(vector_type::const_iterator curr_bfg_pair,
                                        size_type local_index,
                                        iter_state state)
        : curr_bfg_pair_(curr_bfg_pair), local_index_(local_index),
         state_(state) {}


        /** @brief Copy Constructor
         *
         *  @param other   The const_segmented_vector_iterator object which
         *                  is to be copied
         */
        const_segmented_vector_iterator(self_type const& other)
        {
            this->curr_bfg_pair_ = other.curr_bfg_pair_;
            this->local_index_ = other.local_index_;
            this->state_ = other.state_;
        }

        //COPY ASSIGNMENT
        //  PROGRAMMER DOCUMENTATION:
        //  Return self_type& allow a=b=c;
        /** @brief Copy one const_segmented_vector_iterator into other.
         *
         *  @param other The const_segmented_vector_iterator objects which
         *                is to be copied
         *
         *  @return This return the reference to the newly created
         *           const_segmented_vector_iterator
         */
        self_type & operator = (self_type const & other)
        {
            this->curr_bfg_pair_ = other.curr_bfg_pair_;
            this->local_index_ = other.local_index_;
            this->state_ = other.state_;
            return *this;
        }

        //COMPARISON API
        /** @brief Compare the two iterators for equality.
         *
         *  @param other The iterator objects which is to be compared
         *
         *  @return Return true if both are equal, false otherwise
         */
        bool operator == (self_type const & other) const
        {
            return (this->curr_bfg_pair_ == other.curr_bfg_pair_
                     &&
                    this->state_ == other.state_
                     &&
                    this->local_index_ == other.local_index_);
        }

        /** @brief Compare the two iterators for inequality.
         *
         *  @param other The iterator objects which is to be compared
         *
         *  @return Return false if both are equal, false otherwise
         */
        bool operator != (self_type const & other) const
        {
            return !(*this == other);
        }

        //DEREFERENCE
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
        VALUE_TYPE operator * () const
        {
            return (chunk_vector_stubs::get_value_noexpt_async(
                                            (curr_bfg_pair_->second).get(),
                                            local_index_)
                    ).get();
        }

        //INCREMENT
        //  PROGRAMMER DOCUMENTATION:
        //    ALGO:
        //      1. If vector is in INVALID STATE (Refer programmer documentation
        //         for enum above) then we have to increment until we get the
        //         begin of vector. Then change the state to valid.
        //      2. Else just increment the local_index until you go beyond
        //         the actual size of that chunk. If you go beyond then go to the
        //         next gid in the available vector
        //      3. The step 2 is repeated until you hit last valid gid in the list
        //         for the last valid gid you just increment local_index
        //
        /** @brief Increment the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the incremented const_segmented_vector_iterator object
         */
        self_type operator ++ ()  //prefix behavior
        {
            if(this->state_ == hpx::iter_state::invalid)
            {
                ++local_index_;
                if(local_index_ == 0)
                {
                    this->state_ = hpx::iter_state::valid;
                }
            }
            else
            {
                ++local_index_;
                hpx_id invalid_id;
                if(//this condition does not cause function call hence it must
                   // be first
                   ((curr_bfg_pair_ + 1)->second).get() != invalid_id
                   &&
                   local_index_ >= chunk_vector_stubs::size_async(
                                                (curr_bfg_pair_->second).get()
                                                                  ).get()
                   )
                {
                    ++curr_bfg_pair_;
                    local_index_ = 0;
                }
            }
            return *this;
        }

        /** @brief Increment the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original const_segmented_vector_iterator object
         */
        self_type operator ++ (int) //postfix behavior
        {
            //temp object should be return to simulate the postfix behavior
            self_type temp = *this;
            ++(*this);
            return temp;
        }

        //DECREMENT
        // PROGRAMMER DOCUMENTATION:
        //  ALGO:
        //    1. If local_index equal to zero then we have to check is it first
        //       VALID gid in the list. If so the decrement make iterator to
        //       invalid state. Other wise we have to move the immediate previous
        //       gid.
        //    2. Else just decrement the local_index.
        //
        /** @brief Decrement the const_segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the decremented const_segmented_vector_iterator object
         */
        self_type operator -- () //prefix behavior
        {
            //If it is just first gid just decrement the local index
           if( local_index_ == 0)
           {
               if(curr_bfg_pair_->first != 0)
               {
                    --curr_bfg_pair_;
                    local_index_ =
                        ( chunk_vector_stubs::size_async(
                                        (curr_bfg_pair_->second).get()
                                                         ).get() - 1
                         );
               }
               else
               {
                   --local_index_;
                   this->state_ = hpx::iter_state::invalid;
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
        self_type operator -- (int) //postfix behavior
        {
             //temp object should be return to simulate the postfix behavior
            self_type temp = *this;
            --(*this);
            return temp;
        }

        //ARITHMATIC OPERATOR
        /** @brief Return the const_segmented_vector_iterator pointing to the
         *  position which is \a n units ahead of the current position.
         */
        self_type operator + (size_type n) const
        {
            //copying the current states of the iterator
            vector_type::const_iterator temp_curr_bfg_pair = this->curr_bfg_pair_;
            size_type temp_local_index = this->local_index_;
            hpx::iter_state temp_state = this->state_;

            //temp variables
            hpx_id invalid_id;
            bool same_chunk = true;
            size_type size = 0;

            if(temp_state == hpx::iter_state::invalid)
            {
                //calculate the length through which it is invalid
                size_type diff = (std::numeric_limits<size_type>::max() -
                                    temp_local_index + 1);

                if(n < diff )
                {
                    return self_type(temp_curr_bfg_pair,
                                    (temp_local_index + n),
                                     temp_state);
                }
                else
                {
                    n = n - diff;
                    temp_local_index = 0;
                    temp_state = hpx::iter_state::valid;
                }
            }

            //Calculating the size of the first chunk
            size = chunk_vector_stubs::size_async(
                                    (temp_curr_bfg_pair->second).get()
                                                  ).get()
                                     - temp_local_index;

            while( n >= size)
            {
                 //Break this loop if this is previous to LAST gid
                 // i.e. last valid gid in the list
                if(((temp_curr_bfg_pair + 1)->second).get() == invalid_id )
                    break;

                same_chunk = false;
                n = n - size;
                ++temp_curr_bfg_pair;
                // calculate the size of current chunk
                size = chunk_vector_stubs::size_async(
                                    (temp_curr_bfg_pair->second).get()
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
            return self_type(temp_curr_bfg_pair,
                             temp_local_index,
                             temp_state);
        }//End of a + n

        /** @brief Return the const_segmented_vector_iterator pointing to the
         *          position which is \a n units behind the current position.
         */
        self_type operator - (size_type n) const
        {
            //copying the current states of the iterator
            vector_type::const_iterator temp_curr_bfg_pair = this->curr_bfg_pair_;
            size_type temp_local_index = this->local_index_;
            hpx::iter_state temp_state = this->state_;

            //Temp variables
            bool same_chunk = true;
            size_type size = 0;
            if(temp_state == hpx::iter_state::invalid)
            {
                return self_type(temp_curr_bfg_pair,
                                 (temp_local_index - n),
                                 temp_state );
            }
            else
            {
                //
                //this calculate remaining elements in current chunk
                //

                //This size tells how many need to go out side of current gid
                size = temp_local_index + 1;
                while (n >= size)
                {
                    //this condition is only met when iterator is going invalid
                    if(temp_curr_bfg_pair->first == 0)
                    {
                        temp_state = hpx::iter_state::invalid;
                        break;
                    }
                    same_chunk = false;
                    n = n - size;
                    --temp_curr_bfg_pair;
                    size = chunk_vector_stubs::size_async(
                                    (temp_curr_bfg_pair->second).get()
                                                          ).get();
                }//end of while
            }//end of else

            if(same_chunk)
            {
                temp_local_index -= n;
            }
            else
            {
                temp_local_index = size - (n + 1);
            }

            return self_type(temp_curr_bfg_pair,
                             temp_local_index,
                             temp_state);
        }//end of a - n

//        //TODO this returning int64_t which has half range with size_t
//        boost::int64_t operator - (self_type const& other) const
//        {
//            if(this->curr_bfg_pair_ == other.curr_bfg_pair_)
//            {
//                return static_cast<boost::int64_t>(this->local_index_
//                                                   - other.local_index_);
//            }
//            else if(this->curr_bfg_pair_ > other.curr_bfg_pair_) //Answer is positive
//            {
//                std::size_t diff = diff_helper(other.curr_bfg_pair_,
//                                                this->curr_bfg_pair_);
//                //Adding the part from (*this) chunk
//                diff = diff + (this->local_index_ + 1);
//                //Subtracting extra part from from the other chunk
//                diff = diff - (other.local_index_ + 1);
//                //TODO this should be the exception not the assert
//                HPX_ASSERT( diff <= std::numeric_limits<boost::int64_t>::max());
//                return static_cast<boost::int64_t>(diff);
//            }
//            else if(this->curr_bfg_pair_ < other.curr_bfg_pair_) //Answer is negative
//            {
//                std::size_t diff = diff_helper(this->curr_bfg_pair_,
//                                                other.curr_bfg_pair_);
//                 //Subtracting extra part from (*this) chunk
//                diff = diff - (this->local_index_ + 1);
//                 //Adding the part from from the other chunk
//                diff = diff + (other.local_index_ + 1);
//                //TODO this should be the exception not the assert
//                HPX_ASSERT( diff <= std::numeric_limits<boost::int64_t>::max());
//                return static_cast<boost::int64_t>(diff);
//            }
//            else{HPX_ASSERT(0);}
//        }//end of a - b

        //RELATIONAL OPERATOR
        /** @brief Compare the two iterator for less than relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is less than
         *           other, false otherwise
         */
        bool operator < (self_type const& other) const
        {
             //If both are from diff gid
            if (this->curr_bfg_pair_ < other.curr_bfg_pair_)
            {
                return true;
            }
             //Now if bot are from same gid
            else if (this->curr_bfg_pair_ == other.curr_bfg_pair_)
            {
                //as invalid state = 0 and valid = 1
                if(this->state_ < other.state_)
                    return true;
                //if both are same then check local index
                else if(this->state_ == other.state_
                        &&
                        this->local_index_ < other.local_index_)
                    return true;
            }
                return false;
        }// End of <

        /** @brief Compare the two iterator for greater than relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is greater than
         *           other, false otherwise
         */
        bool operator > (self_type const& other) const
        {
            if (this->curr_bfg_pair_ > other.curr_bfg_pair_)
            {
                return true;
            }
            else if (this->curr_bfg_pair_ == other.curr_bfg_pair_)
            {
                if(this->state_ > other.state_)
                    return true;
                else if (this->state_ == other.state_
                         &&
                         this->local_index_ > other.local_index_)
                    return true;
            }
                return false;
        }// End of >

        /** @brief Compare the two iterator for less than or equal to relation.
         *
         *  @param other This the iterator objects which is to be compared
         *
         *  @return Return true if object with which it called is less than or
         *           equal to the other, false otherwise
         */
        bool operator <= (self_type const& other) const
        {
            if ( (*this) < other || (*this) == other )
            {
                return true;
            }
            else
                return false;
        }// End of <=

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
            if ( (*this) > other || (*this) == other )
            {
                return true;
            }
            else
                return false;
        }// End of >=

        //COMPOUND ASSIGNMENT
        /** @brief Increment the const_segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the incremented object
         */
        self_type & operator +=(size_type n)
        {
            *this = *this + n;
           //return self_type to make (a = (b += n)) work
            return *this;
        }//End of +=

        /** @brief Decrement the const_segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the decremented object
         */
        self_type & operator -=(size_type n)
        {
            *this = *this - n;
            //return self_type to make (a = (b -= n)) work
            return *this;
        }//End of +=

        //OFFSET DEREFERENCE
        /** @brief Dereferences the iterator which is at \a n position ahead of the
         *          current iterator position and returns the value of the element.
         *
         *  @return Value in the element which is at n position ahead of the
         *           current iterator
         */
        VALUE_TYPE operator[](size_type n) const
        {
            self_type temp = *this;
            temp = temp + n;
            return *temp;
        }

        //
        // API related to Segmented Iterators
        //

        //  PROGRAMMER DOCUMENTATION:
        //      Conceptually this function is suppose to denote which segment,
        //  the iterator is currently pointing to (i.e. just global iterator).
        //      As we are having the gid and base index in a pair and we have a
        //  std::vector which store these pairs. So we are returning the
        //  const_iterator so that we can have access to both gid and and
        //  base_index and the iterator is also comparable against == and !=
        //  operator
        static vector_type::const_iterator segment(self_type const& seg_iter)
        {
            return seg_iter.curr_bfg_pair_;
        }

        //  PROGRAMMER DOCUMENTATION:
        //      This function should specify which is the current segment and
        //  in that exact position to which local iterator is pointing.
        //      Now we are returning the pair of the gid and local index to
        //  represent the position.
        static local_return_type local(self_type const& seg_iter)
        {
            return std::make_pair(
                                  (seg_iter.curr_bfg_pair_)->second,
                                   seg_iter.local_index_
                                 );
        }

        //  PROGRAMMER DOCUMENTATION:
        //      This function should specify the local iterator which is at the
        //  beginning of the chunk.
        //      We are returning the pair of the gid and local index as 0. Though
        //  we can get the local iterator at the beginning if we are having the
        //  gid but we are returning this to have same API for each algorithm.
        static local_return_type begin(vector_type::const_iterator chunk_bfg_pair)
        {
            return std::make_pair(chunk_bfg_pair->second, 0);
        }

        //  PROGRAMMER DOCUMENTATION:
        //      This function should specify the local iterator which is at the
        //  end of the chunk.
        //      We are returning the pair of the gid and local index as size.
        //  Though we can get the local iterator at the end if we are having the
        //  gid but we are returning this to have same API for each algorithm.
        static local_return_type end(vector_type::const_iterator chunk_bfg_pair)
        {
            return std::make_pair(
                    chunk_bfg_pair->second,
                    chunk_vector_stubs::size_async(
                                    (chunk_bfg_pair->second).get()
                                                         ).get()
                                 );
        }

        //
        // Destructor
        //
        /** @brief Default destructor for const_segmented_vector_iterator.*/
        ~const_segmented_vector_iterator()
        {
            //DEFAULT destructor
        }

    protected:

        //  PROGRAMMER DOCUMENTATION:
        //  This is the iterator pointing the the vector which stores the
        //   (base_index, gid) pair. (For base_index and gid refer to
        //   hpx::vector class in hpx/components/vector/vector.hpp). This
        //   actually represent to which gid pair our current iterator
        //   position is pointing.
        vector_type::const_iterator         curr_bfg_pair_;

        // This represent the local position in the current base_gid pair
        //  pointed by the curr_bfg_pair_ (defined above) iterator.
        size_type                           local_index_;

        //This represent the state of the iterator
        hpx::iter_state                     state_;

    };//end of const_segmented_vector_iterator

    class segmented_vector_iterator : public const_segmented_vector_iterator
    {
        typedef hpx::segmented_vector_iterator          self_type;
        typedef hpx::const_segmented_vector_iterator    base_type;

        segmented_vector_iterator(base_type const& other): base_type(other){}
    public:
        typedef base_type::vector_type                  vector_type;
        typedef base_type::size_type                    size_type;

        segmented_vector_iterator():base_type() {}
        segmented_vector_iterator(vector_type::const_iterator curr_bfg_pair,
                                  size_type local_index,
                                  iter_state state)
                        :base_type(curr_bfg_pair,
                                   local_index,
                                   state) {}

        /** @brief Copy Constructor
         *
         *  @param other   The segmented_vector_iterator object which
         *                  is to be copied
         */
        segmented_vector_iterator(self_type const& other)
                        : base_type(other.curr_bfg_pair_,
                                    other.local_index_,
                                    other.state_) {}

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
            this->curr_bfg_pair_ = other.curr_bfg_pair_;
            this->local_index_ = other.local_index_;
            this->state_ = other.state_;
            return *this;
        }

        /** @brief Return the segmented_vector_iterator pointing to the
         *  position which is \a n units ahead of the current position.
         */
        self_type operator + (size_type n) const
        {
            base_type temp = *this;
            return self_type(temp + n);
        }

        /** @brief Return the segmented_vector_iterator pointing to the
         *          position which is \a n units behind the current position.
         */
        self_type operator - (size_type n) const
        {
            base_type temp = *this;
            return self_type(temp - n);
        }

        /** @brief Increment the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the incremented segmented_vector_iterator object
         */
        self_type operator ++ ()  //prefix behavior
        {
            base_type temp = *(this);
            *this = ++temp;
            return *this;
        }

        /** @brief Increment the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original segmented_vector_iterator object
         */
        self_type operator ++ (int) //postfix behavior
        {
            //return_temp object should be return to simulate the postfix behavior
            base_type temp = *this;
            base_type return_temp = temp;
            *this = ++temp;
            return return_temp;
        }

        /** @brief Decrement the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the decremented segmented_vector_iterator object
         */
        self_type operator -- () //prefix behavior
        {
            base_type temp = *(this);
            *this = --temp;
            return *this;
        }

        /** @brief Decrement the segmented_vector_iterator position by one
         *          unit.
         *
         *  @return Return the original segmented_vector_iterator object
         */
        self_type operator -- (int) //postfix behavior
        {
            //return_temp object should be return to simulate the postfix behavior
            base_type temp = *this;
            base_type return_temp = temp;
            *this = --temp;
            return return_temp;
        }

        /** @brief Increment the segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the incremented object
         */
        self_type & operator +=(size_type n)
        {
            *this = *this + n;
           //return self_type to make (a = (b += n)) work
            return *this;
        }//End of +=

        /** @brief Decrement the segmented_vector_iterator by \a n.
         *
         *  @return Returns the reference to the decremented object
         */
        self_type & operator -=(size_type n)
        {
            *this = *this - n;
            //return self_type to make (a = (b -= n)) work
            return *this;
        }//End of +=

    };//End of segmented_vector_iterator

}//end of hpx namespace

#endif //  SEGMENTED_ITERATOR_HPP
