//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/vector/vector.hpp

#ifndef HPX_VECTOR_HPP
#define HPX_VECTOR_HPP

/// \brief The hpx::vector and its API's are defined here.
///
/// The hpx::vector is a segmented data structure which is a collection of one
/// or more hpx::chunk_vectors. The hpx::vector stores the global IDs of each
/// hpx::chunk_vector and the index (with respect to whole vector) of the first
/// element in that hpx::chunk_vector. These two are stored in std::pair.

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/runtime/components/new.hpp>

#include <hpx/components/vector/segmented_iterator.hpp>
#include <hpx/components/vector/chunk_vector_component.hpp>
#include <hpx/components/vector/chunking_policy.hpp>

#include <cstdint>
#include <iostream>
#include <memory>

#include <boost/integer.hpp>
#include <boost/foreach.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief This is the vector class which define hpx::vector functionality.
    ///
    ///  This contains the client side implementation of the hpx::vector. This
    ///  class defines the synchronous and asynchronous API's for each of the
    ///  exposed functionalities.
    ///
    template <typename T>
    class vector
    {
    public:
        typedef std::allocator<T> allocator_type;

        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef T value_type;
        typedef T reference;
        typedef T const const_reference;
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;
        typedef typename std::allocator_traits<allocator_type>::const_pointer
            const_pointer;

        typedef vector_iterator<T> iterator;
        typedef const_vector_iterator<T> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        typedef local_vector_iterator<T> local_iterator;
        typedef const_local_vector_iterator<T> const_local_iterator;

        typedef segment_vector_iterator<T> segment_iterator;
        typedef const_segment_vector_iterator<T> const_segment_iterator;

    private:
        typedef hpx::server::chunk_vector chunk_vector_server;
        typedef hpx::stubs::chunk_vector<T> chunk_vector_stub;
        typedef hpx::chunk_vector<T> chunk_vector_client;

        // Each chunk is described by it's corresponding client object, its
        // size, and base index.
        struct chunk_description
        {
            chunk_vector_client chunk_;
            size_type size_;
            size_type base_index_;
        };

        // The list of chunks belonging to this vector.
        typedef std::vector<chunk_description> chunks_vector_type;

        // overall size of the vector
        size_type size_;

        // This is the vector representing the base_index and corresponding
        //  global ID's of chunk_vector.
        chunks_vector_type chunks_;

        // parameters taken from distribution policy
        BOOST_SCOPED_ENUM(distribution_policy) policy_;     // policy to use
        size_type block_size_;                              // cycle length

    private:
        friend class vector_iterator<T>;
        friend class const_vector_iterator<T>;

    public:
        // Return the sequence number of the segment corresponding to the
        // given global index
        std::size_t get_segment(size_type global_index) const
        {
            return 0;
        }

        // Return the local index inside the segment corresponding to the
        // given global index
        std::size_t get_local_index(size_type global_index, std::size_t part) const
        {
            return 0;
        }

        // Return the global index corresponding to the local index inside the
        // given segment.
        std::size_t get_global_index(std::size_t segment, size_type local_index) const
        {
            return 0;
        }

        // Return the local iterator referencing an element inside a segment
        // based on the given global index.
        local_iterator get_local_iterator(size_type global_index) const
        {
            std::size_t part = get_segment(global_index);
            std::size_t local_index = get_local_index(global_index, part);
            return local_iterator(chunks_[part].chunk_, local_index);
        }

        const_local_iterator get_const_local_iterator(size_type global_index) const
        {
            std::size_t part = get_segment(global_index);
            std::size_t local_index = get_local_index(global_index, part);
            return const_local_iterator(chunks_[part].chunk_, local_index);
        }

        // Return the segment iterator referencing a segment based on the
        // given global index.
        segment_iterator get_segment_iterator(size_type global_index)
        {
            typename chunks_vector_type::iterator it =
                chunk_.begin() + get_segment(global_index);
            return segment_iterator(this, it);
        }

        const_segment_iterator get_const_segment_iterator(
            size_type global_index) const
        {
            typename chunks_vector_type::const_iterator it =
                chunk_.cbegin() + get_segment(global_index);
            return const_segment_iterator(this, it);
        }

    protected:
        // This function is called when we are creating the vector from
        // chunk_vector. It create the chunk_description_type (For
        // chunk_description_type refer typedef
        // section) for each chunk_vector and that vector to the
        // base_sf_of_gid_pair which is std::vector of chunk_description_type.
        template <typename T>
        void create(VALUE_TYPE const& val, T const& policy)
        {
//             std::vector<id_type> const& localities = policy.get_localities();
//             size_type num_chunks = policy.get_num_chunks();
//
//             size_type chunk_index             = 0;
//             size_type chunk_size              = 0;
//             size_type index_so_far            = 0;
//             size_type offset_chunk_count      = 0;
//
//             policy.set_big_chunk(big_chunk);
//             size_type offset;
//             size_type local_chunk_count; //
//             size_type extra_local_chunk_count;
//             size_type extra_chunk_size;
//             size_type num_of_chunk_bc; // for block_cyclic
//
//             if (num_chunks == 1)
//             {
//                 if (policy_ == distribution_policy::block_cyclic)
//                 {
//                     block_size         = policy.get_block_size();
//                     num_of_chunk_bc    = size_/block_size;
//                 }
//                 else
//                 {
//                     chunk_size    = big_chunk/localities_.size();
//                 }
//                 offset             = big_chunk%localities_.size();
//                 local_chunk_count  = 1;
//             }
//             else if(num_chunks_ > 1)
//             {
//                 if (policy_ == distribution_policy::block_cyclic)
//                 {
//                     block_size       = policy.get_block_size();
//                     num_of_chunk_bc = size_/block_size;
//                 }
//                 else
//                 {
//                     chunk_size    = big_chunk/num_chunks_;
//                 }
//
//                 offset             = big_chunk%num_chunks_;
//
//                 if(state == block_cyclic)
//                     offset = big_chunk%block_size;
//
//                 local_chunk_count  = num_chunks_/localities_.size();
//                 if (local_chunk_count == 0) local_chunk_count = 1;
//
//                 offset_chunk_count = num_chunks_%localities_.size();
//                 if (localities_.size() > num_chunks_) offset_chunk_count = 0;
//             }
//
//             extra_local_chunk_count = local_chunk_count;
//             extra_chunk_size        = chunk_size;
//             if(num_chunks_ < localities_.size()) localities_.resize(num_chunks_);
//             BOOST_FOREACH(hpx::naming::id_type const& node, localities_)
//             {
//                 if(offset_chunk_count > 0)
//                 {
//                     extra_local_chunk_count = local_chunk_count+1;
//                     --offset_chunk_count;
//                 }
//
//                 for (std::size_t my_index = 0;
//                          my_index < extra_local_chunk_count;
//                          ++my_index)
//                 {
//                     if(offset > 0)
//                     {
//                         extra_chunk_size = chunk_size+1;
//                         --offset;
//                     }
//                     if(state == hpx::dis_state::dis_block) // for block
//                     {
//                         chunks_.push_back(
//                             std::make_pair(
//                                 index_so_far,
//                                 hpx::components::new_<chunk_vector_server>(
//                                 node, extra_chunk_size, val, index_so_far, policy )
//                                            )
//                                                      );
//                         index_so_far = extra_chunk_size + index_so_far;
//                     }
//                     else if(state == hpx::dis_state::dis_cyclic) // for cyclic
//                     {
//                       chunks_.push_back(
//                             std::make_pair(
//                                 chunk_index,
//                                 hpx::components::new_<chunk_vector_server>(
//                                 node, extra_chunk_size, val, chunk_index, policy)
//                                            )
//                                                      );
//                        ++chunk_index;
//                     }
//                     // for block_cyclic
//                     else if(state == hpx::dis_state::block_cyclic)
//                     {
//                         if(num_chunks_ == 1)
//                         {
//                             if((num_of_chunk_bc%localities_.size()) > 0)
//                             {
//                               if(chunk_index<(num_of_chunk_bc%localities_.size()))
//                               {
//                                   extra_chunk_size =
//                                      ((num_of_chunk_bc /
//                                            localities_.size())*block_size)+
//                                            block_size;
//                               }
//                               else if ((chunk_index+1) ==
//                                      num_of_chunk_bc%localities_.size())
//                               {
//                                   extra_chunk_size =
//                                       ((num_of_chunk_bc/localities_.size())*block_size)+
//                                           (size_%block_size);
//                               }
//                               else
//                               {
//                                   extra_chunk_size =
//                                    (num_of_chunk_bc/localities_.size())*block_size;
//                               }
//
//                             }
//                             else
//                             {
//                                 extra_chunk_size =
//                                  (num_of_chunk_bc/localities_.size())*block_size;
//                             }
//                         }
//                         else if(num_chunks_ > 1)
//                         {
//                             if((num_of_chunk_bc%num_chunks_) > 0)
//                             {
//                                if(chunk_index<(num_of_chunk_bc%num_chunks_))
//                                {
//                                    extra_chunk_size =
//                                      ((num_of_chunk_bc/num_chunks_)*block_size) +
//                                      block_size;
//                                }
//                                else if (chunk_index == num_of_chunk_bc%num_chunks_)
//                                {
//                                    extra_chunk_size =
//                                        ((num_of_chunk_bc/num_chunks_)*block_size)+
//                                           (size_%block_size);
//                                }
//                                else
//                                {
//                                    extra_chunk_size =
//                                        (num_of_chunk_bc/num_chunks_)*block_size;
//                                }
//                             }
//                             else
//                             {
//                                 extra_chunk_size =
//                                     (num_of_chunk_bc/num_chunks_)*block_size;
//                             }
//                         }
//                         chunks_.push_back(
//                             std::make_pair(
//                                 chunk_index*block_size,
//                                 hpx::components::new_<chunk_vector_server>(
//                                 node, extra_chunk_size, val,
//                                 (chunk_index*block_size), policy)
//                             ));
//                        ++chunk_index;
//                     }
//
//                     if(extra_chunk_size > chunk_size)
//                         extra_chunk_size = chunk_size;
//                 }
//                 if (extra_local_chunk_count > local_chunk_count)
//                     extra_local_chunk_count  = local_chunk_count;
//             }
//
//             // We must always have at least one chunk_vector.
//             HPX_ASSERT(!chunks_.empty());
        } // End of create function

        //  Return the bgf_pair in which the element represented by pos must be
        //  present. Here one assumption is made that in any case the
        //  num_elements in the hpx::vector must be less than max possible value
        //  of the size_type
//         chunk_vector_type::const_iterator get_base_gid_pair(size_type pos) const
//         {
//             size_type chunk_size = 0;
//             size_type offset     = 0;
//             size_type distance   = 0;
//
//             // Return the iterator to the first element which does not
//             // comparable less than value (i.e equal or greater)
//             chunk_vector_type::const_iterator it;
//             if(state == hpx::dis_state::dis_block)
//             {
//                 if(num_chunks_ == 1)
//                 {
//                     chunk_size = size_/localities_.size();
//                     offset     = size_%localities_.size();
//                 }
//                 else if(num_chunks_ > 1)
//                 {
//                     chunk_size = size_/num_chunks_;
//                     offset     = size_%num_chunks_;
//                 }
//                 if(offset > 0)
//                 {
//                     if((offset*(chunk_size+1)) > pos )
//                         distance = pos/(chunk_size+1);
//                     else
//                     {
//                         distance = offset +
//                             ((pos - (offset*(chunk_size+1)))/chunk_size);
//                     }
//                 }
//                 if(offset == 0) distance =  pos/chunk_size;
//                 it = chunks_.begin() + distance;
//             }
//             else if(state == hpx::dis_state::dis_cyclic)
//             {
//                 if(num_chunks_ > 1)
//                     it = chunks_.begin()+(pos%num_chunks_);
//                 else if (num_chunks_ == 1)
//                 {
//                     it = chunks_.begin()+
//                             (pos%chunks_.size());
//                 }
//             }
//             else if(state == hpx::dis_state::block_cyclic)
//             {
//                 if(num_chunks_>1)
//                 {
//                     it = chunks_.begin()+
//                                     ((pos/block_size)%num_chunks_);
//                 }
//                 else if(num_chunks_ == 1)
//                 {
//                     it = chunks_.begin()+
//                                      ((pos/block_size)%localities_.size());
//                 }
//             }
//
//             //  Second condition avoid the boundary case where the get_value can
//             //  be called on invalid gid. This occurs when pos = -1
//             //  (maximum value)
//             if(it->second == pos && (it->first).get() != invalid_id)
//             {
//                  return it;
//             }
//             else //It takes care of the case if "it" is at the LAST
//             {
//                 return (it);
//             }
//         }//End of get_gid

//        //Note num_chunks == represent then chunk vector index
//        future<size_type>
//            size_helper(size_type num_chunks) const
//        {
//            if(num_chunks < 1)
//            {
//                HPX_ASSERT(num_chunks >= 0);
//                return chunk_vector_stub::size_async(
//                          ((chunks_.at(num_chunks)).second).get()
//                                                      );
//            }
//            else
//                return hpx::lcos::local::dataflow(
//                    [](future<size_type> s1,
//                       future<size_type> s2) -> size_type
//                        {
//                            return s1.get() + s2.get();
//                        },
//                    chunk_vector_stub::size_async(
//                        ((chunks_.at(num_chunks)).second).get()
//                                                         ),
//                    size_helper(num_chunks - 1)
//                                                );
//            }//end of size_helper


//         //FASTER VERSION OF SIZE_HELPER
//
//         // PROGRAMMER DOCUMENTATION:
//         //  This helper function return the number of element in the hpx::vector.
//         //  Here we are dividing the sequence of chunk_description_types into half and
//         //  computing the size of the individual chunk_vector and then adding
//         //  them. Note this create the binary tree of height equal to log
//         //  (num_chunk_description_types in chunks_). Hence it might be efficient
//         //  than previous implementation
//         //
//         // NOTE: This implementation does not need all the chunk_vector of same
//         //       size.
//         //
//         future<size_type> size_helper(chunk_vector_type::const_iterator it_begin,
//                                 chunk_vector_type::const_iterator it_end) const
//         {
//             if((it_end - it_begin) == 1 )
//                 return chunk_vector_stub::size_async((it_begin->first).get());
//             else
//             {
//                 int mid = (it_end - it_begin)/2;
//                 future<size_type> left_tree_size = size_helper(it_begin,
//                                                          it_begin + mid);
//                 future<size_type> right_tree_size = hpx::async(
//                                                     launch::async,
//                                                     hpx::util::bind(
//                                                         &vector::size_helper,
//                                                         this,
//                                                         (it_begin + mid),
//                                                         it_end
//                                                                     )
//                                                         );
//
//             return hpx::lcos::local::dataflow(
//                         [](future<size_type> s1, future<size_type> s2) -> size_type
//                         {
//                             return s1.get() + s2.get();
//                         },
//                         std::move(left_tree_size),
//                         std::move(right_tree_size)
//                                             );
//             }
//         }//end of size_helper
//
// //        future<size_type>
// //            max_size_helper(size_type num_chunks) const
// //        {
// //            if(num_chunks < 1)
// //            {
// //                HPX_ASSERT(num_chunks >= 0);
// //                return chunk_vector_stub::max_size_async(
// //                        ((chunks_.at(num_chunks)).second).get()
// //                                                                );
// //            }
// //            else
// //                return hpx::lcos::local::dataflow(
// //                    [](future<size_type> s1,
// //                       future<size_type> s2) -> size_type
// //                    {
// //                        return s1.get() + s2.get();
// //                    },
// //                    chunk_vector_stub::max_size_async(
// //                        ((chunks_.at(num_chunks)).second).get()
// //                                                             ),
// //                    max_size_helper(num_chunks - 1)
// //                                                );
// //            }//end of max_size_helper
//
//
//         //FASTER VERSION OF MAX_SIZE_HELPER
//
//         // PROGRAMMER DOCUMENTATION:
//         //  This helper function return the number of element in the hpx::vector.
//         //  Here we are dividing the sequence of chunk_description_types into half and
//         //  computing the max_size of the individual chunk_vector and then adding
//         //  them. Note this create the binary tree of height. Equal to log
//         //  (num_chunk_description_types in chunks_). Hence it might be efficient
//         //  than previous implementation
//         //
//         // NOTE: This implementation does not need all the chunk_vector of same
//         //       size.
//         //
//         future<size_type> max_size_helper(chunk_vector_type::const_iterator it_begin,
//                                     chunk_vector_type::const_iterator it_end) const
//         {
//             if((it_end - it_begin) == 1 )
//                 return chunk_vector_stub::max_size_async(
//                                                     (it_begin->first).get()
//                                                                 );
//             else
//             {
//                 int mid = (it_end - it_begin)/2;
//                 future<size_type> left_tree_size = max_size_helper(it_begin,
//                                                              it_begin + mid);
//                 future<size_type> right_tree_size = hpx::async(
//                                                 launch::async,
//                                                 hpx::util::bind(
//                                                     &vector::max_size_helper,
//                                                     this,
//                                                     (it_begin + mid),
//                                                     it_end
//                                                                 )
//                                                         );
//
//                 return hpx::lcos::local::dataflow(
//                             [](future<size_type> s1, future<size_type> s2) -> size_type
//                             {
//                                 return s1.get() + s2.get();
//                             },
//                             std::move(left_tree_size),
//                             std::move(right_tree_size)
//                                                  );
//             }
//         }//end of max_size_helper
//
//
// //        future<size_type>
// //            capacity_helper(size_type num_chunks) const
// //        {
// //            if(num_chunks < 1)
// //            {
// //                HPX_ASSERT(num_chunks >= 0);
// //                return chunk_vector_stub::capacity_async(
// //                          ((chunks_.at(num_chunks)).second).get()
// //                                                         );
// //            }
// //            else
// //                return hpx::lcos::local::dataflow(
// //                    [](future<size_type> s1,
// //                       future<size_type> s2) -> size_type
// //                    {
// //                        return s1.get() + s2.get();
// //                    },
// //                    chunk_vector_stub::capacity_async(
// //                        ((chunks_.at(num_chunks)).second).get()
// //                                                       ),
// //                    capacity_helper(num_chunks - 1)
// //                                                );
// //            }//end of capacity_helper
//
//         //FASTER VERSION OF CAPACITY_HELPER
//
//         // PROGRAMMER DOCUMENTATION:
//         //  This helper function return the number of element in the hpx::vector.
//         //  Here we are dividing the sequence of chunk_description_types into half and
//         //  computing the capacity of the individual chunk_vector and then adding
//         //  them. Note this create the binary tree of height Equal to log
//         //  (num_chunk_description_types in chunks_). Hence it might be efficient
//         //  than previous implementation.
//         //
//         // NOTE: This implementation does not need all the chunk_vector of same
//         //       size.
//         //
//         future<size_type> capacity_helper(chunk_vector_type::const_iterator it_begin,
//                                     chunk_vector_type::const_iterator it_end) const
//         {
//             if((it_end - it_begin) == 1 )
//                 return chunk_vector_stub::capacity_async(
//                                                     (it_begin->first).get()
//                                                           );
//             else
//             {
//                 int mid = (it_end - it_begin)/2;
//                 future<size_type> left_tree_size = capacity_helper(it_begin,
//                                                              it_begin + mid);
//                 future<size_type> right_tree_size = hpx::async(
//                                                 launch::async,
//                                                 hpx::util::bind(
//                                                     &vector::capacity_helper,
//                                                     this,
//                                                     (it_begin + mid),
//                                                     it_end
//                                                                 )
//                                                         );
//
//                 return hpx::lcos::local::dataflow(
//                             [](future<size_type> s1, future<size_type> s2) -> size_type
//                             {
//                                 return s1.get() + s2.get();
//                             },
//                             std::move(left_tree_size),
//                             std::move(right_tree_size)
//                                                  );
//             }
//         }//end of capacity_helper
//
//         // PROGRAMMER DOCUMENTATION:
//         //   This is the helper function to maintain consistency in the
//         //   base_index across all the chunk_description_type. It helps for the resize() and
//         //   assign() function. This is needed as one necessary condition is the
//         //   base_index for chunk_description_type must be unique for each chunk_vector.
//         //
//         void adjust_base_index(chunk_vector_type::iterator begin,
//                                chunk_vector_type::iterator end,
//                                size_type new_chunk_size)
//         {
//             size_type i = 0;
//             for(chunk_vector_type::iterator it = begin; it != end; it++, i++)
//             {
//                 it->second = i * new_chunk_size;
//             }
//         }//end of adjust_base_index

    public:
        /// Default Constructor which create hpx::vector with
        /// \a num_chunks = 1 and \a chunk_size = 0. Hence overall size
        /// of the vector is 0.
        ///
        vector()
          : size_(0),
            policy_(distribution_policy::block)
        {
//             create(T(), hpx::block);
        }

        /// Constructor which create hpx::vector with the given overall \a size
        ///
        /// \param size   The overall size of the vector
        ///
        explicit vector(size_type size)
          : size_(size),
            policy_(distribution_policy::block)
        {
//            create(T(), hpx::block);
        }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val.
        ///
        /// \param size   The overall size of the vector
        /// \param val    Default value for the elements in vector
        ///
        vector(size_type size, T const& val)
          : size_(size),
            policy_(distribution_policy::block)
        {
//             create(val, hpx::block);
        }

        /// Constructor which create and initialize vector of size
        /// \a size using the given distribution policy.
        ///
        /// \param size   The overall size of the vector
        /// \param policy The distribution policy to use (default: block)
        ///
         template <typename DistPolicy>
         vector(size_type size, DistPolicy const& policy)
          : size_(size),
            policy_(policy.get_policy())
         {
//             create(T(), policy);
         }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val and
        /// using the given distribution policy.
        ///
        /// \param size   The overall size of the vector
        /// \param val    Default value for the elements in vector
        /// \param policy The distribution policy to use (default: block)
        ///
        template <typename DistPolicy>
        vector(size_type size, T const& val, DistPolicy const& policy)
          : size_(size),
            policy_(policy.get_policy())
        {
//             create(val, policy);
        }

    private:
        struct value_proxy
        {
            value_proxy(vector& v, size_type index)
              : v_(v), index_(index)
            {}

            operator T() const
            {
                return v_.get_value(index_);
            }

            template <typename T_>
            value_proxy& operator=(T_ && value)
            {
                v_.set_value(index_, std::forward<T_>(value));
                return *this;
            }

            vector& v_;
            size_type index_;
        };

    public:
        /** @brief Array subscript operator. This does not throw any exception.
         *
         *  @param pos Position of the element in the vector [Note the first
         *              position in the chunk is 0]
         *
         *  @return Return the value of the element at position represented by
         *           \a pos [Note that this is not the reference to the element]
         *
         */
        value_proxy operator[](size_type pos)
        {
            return value_proxy(*this, pos);
        }
        T operator[](size_type pos) const
        {
            return get_value(pos);
        }

//             chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//             if(state == hpx::dis_state::dis_block)
//                 return chunk_vector_stub::get_value_noexpt_async(
//                                                         (it->first).get(),
//                                                         (pos - (it->second))
//                                                              ).get();
//              else if(state == hpx::dis_state::dis_cyclic)
//              {
//                  if(num_chunks_ > 1)
//                  {
//                      return chunk_vector_stub::get_value_noexpt_async(
//                                                         (it->first).get(),
//                                                         (pos/num_chunks_)
//                                                             ).get();
//                  }
//                  else if (num_chunks_ ==1)
//                  {
//                      return chunk_vector_stub::get_value_noexpt_async(
//                                                         (it->first).get(),
//                                                         (pos/localities_.size())
//                                                             ).get();
//                  }
//              }
//              else if(state == hpx::dis_state::block_cyclic)
//              {
//                  if(num_chunks_ >1)
//                  {
//                      return chunk_vector_stub::get_value_async(
//                                                      (it->first).get(),
//                                                      ((((pos/block_size)/
//                                                      num_chunks_)*block_size)+
//                                                      (pos%block_size))
//                                                                ).get();
//                  }
//                  else if(num_chunks_ == 1)
//                  {
//                      return chunk_vector_stub::get_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size))
//                                                                ).get();
//                  }
//              }
//
//         }
//
//
//         /** @brief Copy assignment operator.
//          *
//          *  @param other    This the hpx::vector object which is to be copied
//          *
//          *  @return This return the reference to the newly created vector
//          */
//         vector& operator=(vector const& other)
//         {
//             this->chunks_ = other.chunks_;
//             return *this;
//         }


        ///////////////////////////////////////////////////////////////////////
        // Capacity related API's in vector class
        ///////////////////////////////////////////////////////////////////////

        /// \brief Compute the size as the number of elements it contains.
        ///
        /// \return Return the number of elements in the vector
        ///
        size_type size() const
        {
            return size_;
        }

//         /** @brief Asynchronous API for size().
//          *
//          * @return This return the hpx::future of return value of size()
//          */
//         future<size_type> size_async() const
//         {
//             HPX_ASSERT(chunks_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return size_helper(chunks_.begin(),
//                                chunks_.end() - 1);
//         }
//
//         //MAX_SIZE
//         /**  @brief Compute the maximum size of hpx::vector in terms of
//          *           number of elements.
//          *  @return Return maximum number of elements the vector can hold
//          */
//         size_type max_size() const
//         {
//             HPX_ASSERT(chunks_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return max_size_helper(chunks_.begin(),
//                                    chunks_.end() - 1
//                                    ).get();
//         }
//
//         /**  @brief Asynchronous API for max_size().
//          *
//          *  @return Return the hpx::future of return value of max_size()
//          */
//         future<size_type> max_size_async() const
//         {
//             HPX_ASSERT(chunks_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return max_size_helper(chunks_.begin(),
//                                    chunks_.end() - 1);
//         }
//
// //            //RESIZE (without value)
// //
// //            void resize(size_type n)
// //            {
// //                if(n == 0)
// //                    HPX_THROW_EXCEPTION(hpx::invalid_vector_error,
// //                                        "resize",
// //                                        "Invalid Vector: new_chunk_size should be greater than zero");
// //
// //                std::vector<future<void>> resize_lazy_sync;
// //                //Resizing the vector chunks
// //                //AS we have to iterate until we hit LAST
// //                BOOST_FOREACH(chunk_vector_type const& p, std::make_pair(chunks_.begin(), chunks_.end() - 1) )
// //                {
// //                    resize_lazy_sync.push_back(chunk_vector_stub::resize_async((p.second).get(), n));
// //                }
// //                HPX_ASSERT(chunks_.size() > 1); //As this function changes the size we should have LAST always.
// //                //waiting for the resizing
// //                hpx::wait_all(resize_lazy_sync);
// //                adjust_base_index(chunks_.begin(), chunks_.end() - 1, n);
// //            }
// //            future<void> resize_async(size_type n)
// //            {
// //                //static_cast to resolve ambiguity of the overloaded function
// //                return hpx::async(launch::async, hpx::util::bind(static_cast<void(vector::*)(std::size_t)>(&vector::resize), this, n));
// //            }
//
//         // RESIZE (with value)
//         // SEMANTIC DIFFERENCE:
//         //    It is resize with respective chunk not whole vector
//         /** @brief Resize each chunk so that it contain n elements. If
//          *          the \a val is not it use default constructor instead.
//          *
//          *  This function resize the each chunk so that it contains \a n
//          *   elements. [Note that the \a n does not represent the total size of
//          *   vector it is the size of each chunk. This mean if \a n is 10 and
//          *   num_chunks is 5 then total size of vector after resize is 10*5 = 50]
//          *
//          *  @param n    New size of the each chunk
//          *  @param val  Value to be copied if \a n is greater than the current
//          *               size [Default is 0]
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         void resize(size_type n, VALUE_TYPE const& val = VALUE_TYPE())
//         {
//             if(n == 0)
//                 HPX_THROW_EXCEPTION(
//                     hpx::invalid_vector_error,
//                     "resize",
//                     "Invalid Vector: new_chunk_size should be greater than zero"
//                                     );
//
//             std::vector<future<void>> resize_lazy_sync;
//             BOOST_FOREACH(chunk_description_type const& p,
//                           std::make_pair(chunks_.begin(),
//                                          chunks_.end() - 1)
//                          )
//             {
//                 resize_lazy_sync.push_back(
//                                 chunk_vector_stub::resize_async(
//                                                         (p.first).get(),
//                                                          n,
//                                                          val)
//                                            );
//             }
//             hpx::wait_all(resize_lazy_sync);
//
//             //To maintain the consistency in the base_index of each chunk_description_type.
//             adjust_base_index(chunks_.begin(),
//                               chunks_.end() - 1,
//                               n);
//         }
//
//         /** @brief Asynchronous API for resize().
//          *
//          *  @param n    New size of the each chunk
//          *  @param val  Value to be copied if \a n is greater than the current size
//          *               [Default is 0]
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         future<void> resize_async(size_type n,
//                                  VALUE_TYPE const& val = VALUE_TYPE())
//         {
//             //static_cast to resolve ambiguity of the overloaded function
//             return hpx::async(launch::async,
//                               hpx::util::bind(
//                                 static_cast<
//                                 void(vector::*)(size_type,
//                                                 VALUE_TYPE const&)
//                                             >
//                                             (&vector::resize),
//                                               this,
//                                               n,
//                                               val)
//                               );
//         }
//
//         //CAPACITY
//
//         /** @brief Compute the size of currently allocated storage capacity for
//          *          vector.
//          *
//          *  @return Returns capacity of vector, expressed in terms of elements
//          */
//         size_type capacity() const
//         {
//             HPX_ASSERT(chunks_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return capacity_helper(chunks_.begin(),
//                                    chunks_.end() - 1
//                                    ).get();
//         }
//
//         /** @brief Asynchronous API for capacity().
//          *
//          *  @return Returns the hpx::future of return value of capacity()
//          */
//         future<size_type> capacity_async() const
//         {
//             HPX_ASSERT(chunks_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return capacity_helper(chunks_.begin(),
//                                    chunks_.end() - 1);
//         }
//
//         //EMPTY
//         /** @brief Return whether the vector is empty.
//          *
//          *  @return Return true if vector size is 0, false otherwise
//          */
//         bool empty() const
//         {
//             return !(this->size());
//         }
//
//         /** @brief Asynchronous API for empty().
//          *
//          *  @return The hpx::future of return value empty()
//          */
//         future<bool> empty_async() const
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::empty, this));
//         }
//
//         //RESERVE
//         /** @brief Request the change in each chunk capacity so that it
//          *          can hold \a n elements. Throws the
//          *          \a hpx::length_error exception.
//          *
//          *  This function request for each chunk capacity should be at
//          *   least enough to contain \a n elements. For all chunk in vector
//          *   if its capacity is less than \a n then their reallocation happens
//          *   to increase their capacity to \a n (or greater). In other cases
//          *   the chunk capacity does not got affected. It does not change the
//          *   chunk size. Hence the size of the vector does not affected.
//          *
//          * @param n Minimum capacity of chunk
//          *
//          * @exception hpx::length_error If \a n is greater than maximum size for
//          *             at least one chunk then function throw
//          *             \a hpx::length_error exception.
//          */
//         void reserve(size_type n)
//         {
//             std::vector<future<void>> reserve_lazy_sync;
//             BOOST_FOREACH(chunk_description_type const& p,
//                           std::make_pair(chunks_.begin(),
//                                          chunks_.end() - 1)
//                           )
//             {
//                 reserve_lazy_sync.push_back(
//                         chunk_vector_stub::reserve_async((p.first).get(), n)
//                                             );
//             }
//             hpx::wait_all(reserve_lazy_sync);
//         }
//
//         /** @brief Asynchronous API for reserve().
//          *
//          *  @param n Minimum capacity of chunk
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          *
//          *  @exception hpx::length_error If \a n is greater than maximum size
//          *              for at least one chunk then function throw
//          *              \a hpx::length_error exception.
//          */
//         future<void> reserve_async(size_type n)
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::reserve, this, n));
//         }
//
//
//         //
//         //  Element access API's in vector class
//         //

        /** @brief Returns the element at position \a pos in the vector
         *          container. It throws the \a hpx::out_of_range exception.
         *
         *  @param pos Position of the element in the vector [Note the first
         *          position in the chunk is 0]
         *
         *  @return Return the value of the element at position represented by
         *           \a pos [Note that this is not the reference to the element]
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        T get_value(size_type pos) const
        {
            return T();
        }

//             try{
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//                 if(state == hpx::dis_state::dis_block)
//                     return chunk_vector_stub::get_value_async(
//                                                            (it->first).get(),
//                                                            (pos - (it->second))
//                                                           ).get();
//                 else if(state == hpx::dis_state::dis_cyclic)
//                 {
//                     if(num_chunks_ > 1)
//                     {
//                         return chunk_vector_stub::get_value_async(
//                                                         (it->first).get(),
//                                                         (pos/num_chunks_)
//                                                                ).get();
//                     }
//                     else if(num_chunks_ == 1)
//                     {
//                         return chunk_vector_stub::get_value_async(
//                                                         (it->first).get(),
//                                                         (pos/localities_.size())
//                                                               ).get();
//                     }
//                 }
//                 else if(state == hpx::dis_state::block_cyclic)
//                 {
//                    if(num_chunks_ >1)
//                    {
//                        return chunk_vector_stub::get_value_async(
//                                                      (it->first).get(),
//                                                      ((((pos/block_size)/
//                                                      num_chunks_)*block_size)+
//                                                      (pos%block_size))
//                                                                ).get();
//                    }
//                    else if(num_chunks_ == 1)
//                    {
//                        return chunk_vector_stub::get_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size))
//                                                                ).get();
//                     }
//                 }
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(hpx::out_of_range,
//                                     "get_value",
//                                     "Value of 'pos' is out of range");
//             }
//         }//end of get_value
//
//         /** @brief Asynchronous API for get_value(). It throws the
//          *          \a hpx::out_of_range exception.
//          *
//          *  @param pos Position of the element in the vector [Note the first
//          *          position in the chunk is 0]
//          *
//          *  @return Return the hpx::future to value of the element at position
//          *           represented by \a pos [Note that this is not the reference
//          *           to the element]
//          *
//          *  @exception hpx::out_of_range The \a pos is bound checked and if
//          *              \a pos is out of bound then it throws the
//          *              \a hpx::out_of_range exception.
//          */
//         hpx::future< VALUE_TYPE > get_value_async(size_type pos) const
//         {
//             // Here you can call the get_val_sync API but you have already an
//             // API to do that which reduce one function call
//             try{
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//                 if(state == hpx::dis_state::dis_block)
//                 {
//                     return chunk_vector_stub::get_value_async(
//                                                         (it->first).get(),
//                                                         (pos - (it->second))
//                                                           );
//                 }
//                 else if(state == hpx::dis_state::dis_cyclic)
//                 {
//                     if(num_chunks_ > 1)
//                     {
//                         return chunk_vector_stub::get_value_async(
//                                                         (it->first).get(),
//                                                         (pos/num_chunks_)
//                                                                );
//                     }
//                     else if(num_chunks_ == 1)
//                     {
//                         return chunk_vector_stub::get_value_async(
//                                                         (it->first).get(),
//                                                         (pos/localities_.size())
//                                                               );
//                     }
//                 }
//                 else if(state == hpx::dis_state::block_cyclic)
//                 {
//                    if(num_chunks_ >1)
//                    {
//                        return chunk_vector_stub::get_value_async(
//                                                      (it->first).get(),
//                                                      ((((pos/block_size)/
//                                                      num_chunks_)*block_size)+
//                                                      (pos%block_size))
//                                                                );
//                    }
//                    else if(num_chunks_ == 1)
//                    {
//                        return chunk_vector_stub::get_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size))
//                                                                );
//                    }
//                 }
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(
//                     hpx::out_of_range,
//                     "get_value_async",
//                     "Value of 'pos' is out of range");
//             }
//         }//end of get_value_async
//
//         //FRONT (never throws exception)
//         /** @brief Access the value of first element in the vector.
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the value of the first element in the vector
//          */
//         VALUE_TYPE front() const
//         {
//             return chunk_vector_stub::front_async(
//                                     (chunks_.front().first).get()
//                                                   ).get();
//         }//end of front_value
//
//         /** @brief Asynchronous API for front().
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the hpx::future to return value of front()
//          */
//         hpx::future< VALUE_TYPE > front_async() const
//         {
//             return chunk_vector_stub::front_async(
//                                     (chunks_.front().first).get()
//                                                   );
//         }//end of front_async
//
//         //BACK (never throws exception)
//         /** @brief Access the value of last element in the vector.
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the value of the last element in the vector
//          */
//         VALUE_TYPE back() const
//         {
//             // As the LAST pair is there and then decrement operator to that
//             // LAST is undefined hence used the end() function rather than back()
//             return chunk_vector_stub::back_async(
//                             ((chunks_.end() - 2)->first).get()
//                                                  ).get();
//         }//end of back_value
//
//         /** @brief Asynchronous API for back().
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return hpx::future to the return value of back()
//          */
//         hpx::future< VALUE_TYPE > back_async() const
//         {
//             //As the LAST pair is there
//             return chunk_vector_stub::back_async(
//                             ((chunks_.end() - 2)->first).get()
//                                                  );
//         }//end of back_async
//
//
//
//         //
//         // Modifier component action
//         //
//
//         //ASSIGN
//         /** @brief Assigns new contents to each chunk, replacing its
//          *          current contents and modifying each chunk size
//          *          accordingly.
//          *
//          *  @param n     New size of each chunk
//          *  @param val   Value to fill the chunk with
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         void assign(size_type n, VALUE_TYPE const& val)
//         {
//             if(n == 0)
//                 HPX_THROW_EXCEPTION(
//                     hpx::invalid_vector_error,
//                     "assign",
//                     "Invalid Vector: new_chunk_size should be greater than zero"
//                                     );
//
//             std::vector<future<void>> assign_lazy_sync;
//             BOOST_FOREACH(chunk_description_type const& p,
//                           std::make_pair(chunks_.begin(),
//                                          chunks_.end() - 1)
//                           )
//             {
//                 assign_lazy_sync.push_back(
//                     chunk_vector_stub::assign_async((p.first).get(), n, val)
//                                           );
//             }
//             hpx::wait_all(assign_lazy_sync);
//             adjust_base_index(chunks_.begin(),
//                               chunks_.end() - 1,
//                               n);
//         }//End of assign
//
//         /** @brief Asynchronous API for assign().
//          *
//          *  @param n     New size of each chunk
//          *  @param val   Value to fill the chunk with
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          */
//         future<void> assign_async(size_type n, VALUE_TYPE const& val)
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::assign,
//                                               this,
//                                               n,
//                                               val)
//                               );
//         }
//
//         //PUSH_BACK
//         /** @brief Add new element at the end of vector. The added element
//          *          contain the \a val as value.
//          *
//          *  The value is added to the back to the last chunk.
//          *
//          *  @param val Value to be copied to new element
//          */
//         void push_back(VALUE_TYPE const& val)
//         {
//             chunk_vector_stub::push_back_async(
//                             ((chunks_.end() - 2 )->first).get(),
//                                                 val
//                                                 ).get();
//         }
//
//         /** @brief Asynchronous API for push_back().
//          *
//          *  @param val Value to be copied to new element
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          */
//         future<void> push_back_async(VALUE_TYPE const& val)
//         {
//             return chunk_vector_stub::push_back_async(
//                             ((chunks_.end() - 2)->first).get(),
//                                                         val
//                                                         );
//         }
//
//         //PUSH_BACK (With rval)
//         /** @brief Add new element at the end of vector. The added element
//          *          contain the \a val as value.
//          *
//          *  The value is added to the back to the last chunk.
//          *
//          *  @param val Value to be moved to new element
//          */
//         void push_back(VALUE_TYPE const&& val)
//         {
//             chunk_vector_stub::push_back_rval_async(
//                             ((chunks_.end() - 2)->first).get(),
//                                                     std::move(val)
//                                                      ).get();
//         }
//
//         /** @brief Asynchronous API for push_back(VALUE_TYPE const&& val).
//          *
//          *  @param val Value to be moved to new element
//          */
//         future<void> push_back_async(VALUE_TYPE const&& val)
//         {
//             return chunk_vector_stub::push_back_rval_async(
//                             ((chunks_.end() - 2)->first).get(),
//                                                     std::move(val)
//                                                             );
//         }
//
//         //POP_BACK (Never throw exception)
// //            void pop_back()
// //            {
// //                chunk_vector_stub::pop_back_async(( (chunks_.end() - 2)->second).get()).get();
// //                //TODO if following change the affect back() and further pop_back function
// //                //checking if last element from the particular gid is popped up then delete that..
// //                // (-2)I am retaining one gid in vector as otherwise it goes to invalid state and it makes a compulsion that we need to keep at least one element that is not good
// //                if(chunk_vector_stub::empty_async(( (chunks_.end() - 2)->second).get()).get() && chunks_.size() > 2)
// //                    chunks_.pop_back();
// //                HPX_ASSERT(chunks_.size() > 1); //As this function changes the size we should have LAST always.
// //            }
//
        //
        //  set_value API's in vector class
        //
        /** @brief Copy the value of \a val in the element at position \a pos in
         *          the vector container. It throws the \a hpx::out_of_range
         *          exception.
         *
         *  @param pos   Position of the element in the vector [Note the first
         *                position in the vector is 0]
         *  @param val   The value to be copied
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        template <typename T_>
        void set_value(size_type pos, T_ && val)
        {
        }

//             try{
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//
//                 if(state == hpx::dis_state::dis_block)
//                     return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos - (it->second)),
//                                                          val
//                                                               ).get();
//                  else if(state == hpx::dis_state::dis_cyclic)
//                  {
//                      if(num_chunks_ > 1)
//                          return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos/num_chunks_),
//                                                         val
//                                                                ).get();
//                      else if(num_chunks_ == 1)
//                           return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos/localities_.size()),
//                                                         val
//                                                               ).get();
//
//                  }
//                  else if(state == hpx::dis_state::block_cyclic)
//                  {
//                     if(num_chunks_ >1)
//                        return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                       num_chunks_)*block_size)+
//                                                       (pos%block_size)),
//                                                         val
//                                                                ).get();
//                     else if(num_chunks_ == 1)
//                        return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size)),
//                                                         val
//                                                                ).get();
//
//                 }
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(hpx::out_of_range,
//                                     "set_value",
//                                     "Value of 'pos' is out of range"
//                                     );
//             }
//         }//end of set_value
//
//         /** @brief Asynchronous API for set_value(). It throws the
//          *          \a hpx::out_of_range exception.
//          *
//          *  @param pos   Position of the element in the vector [Note the first
//          *                position in the vector is 0]
//          *  @param val   The value to be copied
//          *
//          *  @exception hpx::out_of_range The \a pos is bound checked and if
//          *              \a pos is out of bound then it throws the
//          *              \a hpx::out_of_range exception.
//          */
//         future<void> set_value_async(size_type pos, VALUE_TYPE const& val)
//         {
//             try{
//                 // This reduce one function call as we are directly calling
//                 // chunk vector API
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//                if(state == hpx::dis_state::dis_block)
//                     return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos - (it->second)),
//                                                         val   );
//                else if(state == hpx::dis_state::dis_cyclic)
//                {
//                   if(num_chunks_ > 1)
//                      return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos/num_chunks_),
//                                                         val   );
//                   else if (num_chunks_ == 1)
//                       return chunk_vector_stub::set_value_async(
//                                                         (it->first).get(),
//                                                         (pos/localities_.size()),
//                                                         val   );
//
//                }
//                else if(state == hpx::dis_state::block_cyclic)
//                {
//                    if(num_chunks_ >1)
//                       return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                       num_chunks_)*block_size)+
//                                                       (pos%block_size)),
//                                                         val
//                                                                );
//                    else if(num_chunks_ == 1)
//                        return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size)),
//                                                         val
//                                                                );
//
//                }
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(hpx::out_of_range,
//                                     "set_value_async",
//                                     "Value of 'pos' is out of range");
//             }
//         }//end of set_value_async
//
//         //SET_VALUE (with rval)
//         /** @brief Move the val in the element at position \a pos in the vector
//          *          container. It throws the \a hpx::out_of_range exception.
//          *
//          *  @param pos   Position of the element in the vector [Note the
//          *                first position in the vector is 0]
//          *  @param val   The value to be moved
//          *
//          *  @exception hpx::out_of_range The \a pos is bound checked and if
//          *              \a pos is out of bound then it throws the
//          *              \a hpx::out_of_range exception.
//          */
//         void set_value(size_type pos, VALUE_TYPE const&& val)
//         {
//            try{
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//                 if(state == hpx::dis_state::dis_block)
//                 {
//
//                     return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos - (it->second)),
//                                                     std::move(val)
//                                                                  ).get();
//                 }
//                 else if (state == hpx::dis_state::dis_cyclic)
//                 {
//                     if(num_chunks_ > 1)
//                        return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos/num_chunks_),
//                                                     std::move(val)
//                                                                 ).get();
//                      else if(num_chunks_ == 1)
//                         return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos/localities_.size()),
//                                                     std::move(val)
//                                                                 ).get();
//
//
//                 }
//                else if(state == hpx::dis_state::block_cyclic)
//                {
//                    if(num_chunks_ >1)
//                       return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                       num_chunks_)*block_size)+
//                                                       (pos%block_size)),
//                                                         std::move(val)
//                                                                ).get();
//                    else if(num_chunks_ == 1)
//                        return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size)),
//                                                         std::move(val)
//                                                                ).get();
//
//                 }
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(hpx::out_of_range,
//                                     "set_value",
//                                     "Value of 'pos' is out of range");
//             }
//         }//end of set_value
//
//         /** @brief Asynchronous API for
//          *          set_value(std::size_t pos, VALUE_TYPE const&& val).
//          *          It throws the \a hpx::out_of_range exception.
//          *
//          *  @param pos   Position of the element in the vector [Note the
//          *                first position in the vector is 0]
//          *  @param val   The value to be moved
//          *
//          *  @exception hpx::out_of_range The \a pos is bound checked and if
//          *              \a pos is out of bound then it throws the
//          *              \a hpx::out_of_range exception.
//          */
//         future<void> set_value_async(size_type pos, VALUE_TYPE const&& val)
//         {
//             try{
//                 chunk_vector_type::const_iterator it = get_base_gid_pair(pos);
//                 if(state == hpx::dis_state::dis_block)
//                     return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos - (it->second)),
//                                                         std::move(val)
//                                                                  );
//                 else if (state == hpx::dis_state::dis_cyclic)
//                 {
//                     if(num_chunks_ > 1)
//                       return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos/num_chunks_),
//                                                     std::move(val)
//                                                                 );
//
//                     else if(num_chunks_ == 1)
//                         return chunk_vector_stub::set_value_rval_async(
//                                                     (it->first).get(),
//                                                     (pos/localities_.size()),
//                                                     std::move(val)
//                                                                 );
//
//                }
//                else if(state == hpx::dis_state::block_cyclic)
//                {
//                    if(num_chunks_ >1)
//                       return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                       num_chunks_)*block_size)+
//                                                       (pos%block_size)),
//                                                         std::move(val)
//                                                                );
//                    else if(num_chunks_ == 1)
//                        return chunk_vector_stub::set_value_async(
//                                                       (it->first).get(),
//                                                       ((((pos/block_size)/
//                                                  localities_.size())*block_size)+
//                                                       (pos%block_size)),
//                                                         std::move(val)
//                                                                );
//                }
//
//             }
//             catch(hpx::exception const& /*e*/){
//                 HPX_THROW_EXCEPTION(hpx::out_of_range,
//                                     "set_value_async",
//                                     "Value of 'pos' is out of range");
//             }
//         }//end of set_value_async
//
//
//
//             //CLEAR
//             //TODO if number of chunks is kept constant every time then clear should modified (clear each chunk_vector one by one).
// //            void clear()
// //            {
// //                //It is keeping one gid hence iterator does not go in an invalid state
// //                chunks_.erase(chunks_.begin() + 1,
// //                                           chunks_.end()-1);
// //                chunk_vector_stub::clear_async((chunks_[0].second).get()).get();
// //                HPX_ASSERT(chunks_.size() > 1); //As this function changes the size we should have LAST always.
// //            }
//
//             //
//             // HPX CUSTOM API's
//             //
//
// //            //CREATE_CHUNK
// //            //TODO This statement can create Data Inconsistency :
// //             //If size of chunks_ calculated and added to the base_index but not whole creation is completed and in betwen this som push_back on hpx::vector is done then that operation is losted
// //            void create_chunk(hpx::naming::id locality, std::size_t chunk_size = 0, VALUE_TYPE val = 0.0)
// //            {
// //                chunks_.push_back(
// //                        std::make_pair(
// //                            chunks_.size(),
// //                             hpx::components::new_<chunk_chunk_vector_type>(locality, chunk_size, val)
// //                                      )
// //                                                    );
// //            }//end of create chunk

        ///////////////////////////////////////////////////////////////////////
        /// \brief Return the iterator at the beginning of the vector.
        iterator begin()
        {
            return iterator(this, 0);
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator begin() const
        {
            return const_iterator(this, 0);
        }

        /// \brief Return the iterator at the end of the vector.
        iterator end()
        {
            return iterator(this, size_);
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator end() const
        {
            return const_iterator(this, size_);
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator cbegin() const
        {
            return const_iterator(this, 0);
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator cend() const
        {
            return const_iterator(this, size_);
        }
    };
}

#endif // VECTOR_HPP
