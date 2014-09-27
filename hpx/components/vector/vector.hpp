
//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef VECTOR_HPP
#define VECTOR_HPP

/** @file hpx/components/vector/vector.hpp
 *
 *  @brief The hpx::vector and its API's are defined here.
 *
 *  The hpx::vector is segmented data structure which is collection of one or
 *   more hpx::chunk_vector. The hpx::vector stores the global ID's of each
 *   hpx::chunk_vector and the index (with respect to whole vector) of the first
 *   element in that hpx::chunk_vector. These two are stored in std::pair. Note that
 *   the hpx::vector must contain at least one chunk_vector at any time during
 *   its life cycle. The hpx::vector is considered INVALID if it is created with
 *   zero chunk_vector or n (n>0) chunk_vector with chunk_size 0.
 */

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/runtime/components/new.hpp>

// headers for checking the ranges of the Datatypes
#include <cstdint>
#include <iostream>
#include <boost/integer.hpp>

#include <hpx/components/vector/segmented_iterator.hpp>
#include <hpx/components/vector/chunk_vector_component.hpp>
#include <hpx/components/vector/chunking_policy.hpp>

#include <boost/foreach.hpp>

//TODO Remove all unnecessary comments from file

/** @brief Defines the type of value stored by elements in the
 *          hpx::chunk_vector.
 */
#define VALUE_TYPE double

/**
 *  @namespace hpx
 *  @brief Main namespace of the HPX.
 *
 *  This holds all hpx API such as some distributed data structure eg. vector,
 *   components, dataflow, parallel Algorithms implemented in hpx etc.
 *
 */
namespace hpx{

    /** @brief This is the vector class which define hpx::vector functionality.
    *
    *   This contain the implementation of the hpx::vector. This class defined
    *    the synchronous and asynchronous API's for each functionality.
    */
    enum dis_state{
        dis_block         = 0, /**< This represent the block */
        dis_cyclic        = 1,      /**< This represent the cyclic */
        dis_block_cyclic  = 2
        };


    class vector
    {
    public:
        //  Shorsegmented_vector_iterator. It also maintain to
        //  the same API just like standard vector.
        typedef hpx::segmented_vector_iterator          iterator;
        typedef hpx::const_segmented_vector_iterator    const_iterator;
        typedef std::size_t                             size_type;
        typedef hpx::lcos::future<size_type>            size_future;
        typedef hpx::lcos::future<void>                 void_future;
        typedef hpx::lcos::future<bool>                 bool_future;
        hpx::dis_state                                  state;
    private:
        //Short name for chunk_vector class in hpx::server namespace.
        typedef hpx::server::chunk_vector           chunk_vector_server;
        typedef hpx::stubs::chunk_vector            chunk_vector_stubs;
        typedef hpx::naming::id_type                hpx_id;
        typedef hpx::lcos::shared_future<hpx_id>    hpx_id_shared_future;
        //PROGRAMMER DOCUMENTATION:
        //  This pair consist of the base index of the first (0'th) element in
        //   the chunk_vector (This base index is represented with respect to
        //   the complete hpx::vector) and shared future of the Global ID of the
        //   chunk_vector. Taken as future of hpx_id's as it delay the .get() as
        //   far as possible. The shared_future is necessary as .get() function
        //   can be called more than once.
        typedef std::pair<size_type, hpx_id_shared_future> bfg_pair;

        // The standard vector of the bfg_pair defined above.
        typedef std::vector< bfg_pair >                    vector_type;

        // This typedef helps to call object of same class.
        typedef hpx::vector                                self_type;
        size_type                                          num_chunk; // number of chunk
        size_type                                          size_of_vec; // big_chunk
        size_type                                          block_size; // for block_cyclic 
        // This holds all the localities available  
        std::vector<hpx::naming::id_type> localities;
    protected:
        // PROGRAMMER DOCUMENTATION:
        //   This function is called when we are creating the vector from
        //    chunk_vector. It create the bfg_pair (For bfg_pair refer typedef
        //    section) for each chunk_vector and that vector to the
        //    base_sf_of_gid_pair which is std::vector of bfg_pair.
        //
        //   In the end we are appending the extra bfg_pair called LAST. This is
        //    the junk pair which helps to keep track the end of the hpx::vector.
        //    Note that in any case the LAST must have to be there other wise
        //    hpx::vector cause undefined behavior and program may crash.
        //
        //   The base_index value for each bfg_pair must be unique.
        //
        //   What is LAST?
        //     This is the pair containing the max possible value for the
        //     size_type and invalid global id which is created by default
        //     when we create the hpx::naming::id_type variable.
        //
        //   LAST also helps in preventing the crash of the system due to invalid
        //   state of the iterator (segmented_iterator) defined for hpx::vector.
        
        template<typename T>
        void create(size_type big_chunk, VALUE_TYPE val,
                      T policy)
        {
            size_type chunk_index             = 0;
            localities                        = policy.get_localities();
            num_chunk                         = policy.get_num_chunk();
            size_type chunk_size              = 0;
            size_type index_so_far            = 0;            
            size_type offset_chunk_count      = 0;
            size_of_vec                       = big_chunk;   
            policy.set_big_chunk(big_chunk);
            size_type offset; 
            size_type local_chunk_count; // 
            size_type extra_local_chunk_count;
            size_type extra_chunk_size; 
            size_type num_of_chunk_bc; // for block_cyclic 
                       

            if(num_chunk == 1)
            {   
                if(state == hpx::dis_state::dis_block_cyclic)
                {
                    block_size         = policy.get_block_size();    
                    num_of_chunk_bc    = size_of_vec/block_size;
                }
                else chunk_size    = big_chunk/localities.size(); 
                offset             = big_chunk%localities.size(); 
                local_chunk_count  = 1; 
            }
            else if(num_chunk > 1)
            {
                if(state == hpx::dis_state::dis_block_cyclic)
                {
                    block_size       = policy.get_block_size();    
                    num_of_chunk_bc = size_of_vec/block_size;     
                }
                else chunk_size    = big_chunk/num_chunk; 
                offset             = big_chunk%num_chunk;
                if(state == hpx::dis_state::dis_block_cyclic)
                    offset = big_chunk%block_size; 
                local_chunk_count  = num_chunk/localities.size(); 
                if(local_chunk_count == 0) local_chunk_count = 1;  
                offset_chunk_count = num_chunk%localities.size(); 
                if(localities.size() > num_chunk) offset_chunk_count = 0; 
            }
            
            extra_local_chunk_count = local_chunk_count;
            extra_chunk_size        = chunk_size;
            if(num_chunk < localities.size()) localities.resize(num_chunk); 
            BOOST_FOREACH(hpx::naming::id_type const& node, localities)
            {
                if(offset_chunk_count > 0)
                {
                    extra_local_chunk_count = local_chunk_count+1;                    
                    --offset_chunk_count;
                }
                 
                for (std::size_t my_index = 0;
                         my_index < extra_local_chunk_count;
                         ++my_index)
                {
                    if(offset > 0)
                    {
                        extra_chunk_size = chunk_size+1;
                        --offset;
                    }
                    if(state == hpx::dis_state::dis_block) // for block
                    {
                        base_sf_of_gid_pair_.push_back(
                            std::make_pair(
                                index_so_far,
                                hpx::components::new_<chunk_vector_server>(
                                node, extra_chunk_size, val               )
                                           )
                                                     );
                        index_so_far = extra_chunk_size + index_so_far;
                    } 
                    else if(state == hpx::dis_state::dis_cyclic) // for cyclic
                    {
                      base_sf_of_gid_pair_.push_back(
                            std::make_pair(
                                chunk_index,
                                hpx::components::new_<chunk_vector_server>(
                                node, extra_chunk_size, val                     )
                                           )
                                                     );
                       ++chunk_index;
                    }
                    // for block_cyclic
                    else if(state == hpx::dis_state::dis_block_cyclic)
                    {
                        if(num_chunk == 1)
                        {
                            if((num_of_chunk_bc%localities.size()) > 0) 
                            {
                              if(chunk_index<(num_of_chunk_bc%localities.size()))  
                              {
                                  extra_chunk_size =
                                     ((num_of_chunk_bc / 
                                           localities.size())*block_size)+ 
                                           block_size;
                              }
                              else if ((chunk_index+1) ==
                                     num_of_chunk_bc%localities.size())
                              {        
                                  extra_chunk_size =
                                      ((num_of_chunk_bc/localities.size())*block_size)+
                                          (size_of_vec%block_size);
                              }
                              else 
                              {   
                                  extra_chunk_size = 
                                   (num_of_chunk_bc/localities.size())*block_size;
                              }
                              
                            }
                            else
                            {
                                extra_chunk_size = 
                                 (num_of_chunk_bc/localities.size())*block_size;
                            }
                        }
                        else if(num_chunk > 1)
                        {
                            if((num_of_chunk_bc%num_chunk) > 0)
                            {
                               if(chunk_index<(num_of_chunk_bc%num_chunk))  
                               {     
                                   extra_chunk_size = 
                                     ((num_of_chunk_bc/num_chunk)*block_size) + 
                                     block_size;
                               }
                               else if (chunk_index == num_of_chunk_bc%num_chunk)
                               {      
                                   extra_chunk_size =
                                       ((num_of_chunk_bc/num_chunk)*block_size)+
                                          (size_of_vec%block_size);
                               }
                               else
                               {  
                                   extra_chunk_size = 
                                       (num_of_chunk_bc/num_chunk)*block_size;
                               }
                            }
                            else 
                            {
                                extra_chunk_size = 
                                    (num_of_chunk_bc/num_chunk)*block_size;
                            }
                        } 
                        base_sf_of_gid_pair_.push_back(
                            std::make_pair(
                                chunk_index*block_size,
                                hpx::components::new_<chunk_vector_server>(
                                node, extra_chunk_size, val                     )
                                          )
                                                     );
                       ++chunk_index;

                    
                    }
 
                    if(extra_chunk_size > chunk_size)
                        extra_chunk_size = chunk_size;
                }
                if(extra_local_chunk_count > local_chunk_count)
                    extra_local_chunk_count  = local_chunk_count;
            }
               // Pushing LAST at the back
            hpx_id invalid_id;
            base_sf_of_gid_pair_.push_back(
                std::make_pair(std::numeric_limits<size_type>::max(),
                                hpx::make_ready_future(invalid_id)
                              )
                                          );
             //We must always have at least one chunk_vector and LAST.
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
        } // End of create function


        //  PROGRAMMER DOCUMENTATION:
        //  Return the bgf_pair in which the element represented by pos must be
        //  present. Here one assumption is made that in any case the
        //  num_elements in the hpx::vector must be less than max possible value
        //  of the size_type
        vector_type::const_iterator get_base_gid_pair(size_type pos) const
        {
            hpx_id_shared_future sf;
            hpx_id invalid_id;
            size_type chunk_size = 0;
            size_type offset     = 0;
            size_type distance   = 0;
            // Return the iterator to the first element which does not
            // comparable less than value (i.e equal or greater)
            vector_type::const_iterator it; 
            if(state == hpx::dis_state::dis_block)
            {
                /*it =    std::lower_bound(
                            base_sf_of_gid_pair_.begin(),
                            // As we don't need to compare with LAST, if not
                            // matches. LowerBound return the last iterator so
                            // the return should be LAST not the .end() hence
                            // --it (below condition) make sense or else it
                            // doesn't
                            base_sf_of_gid_pair_.end() - 1,
                            std::make_pair(pos, sf),
                            [](bfg_pair const& middle,
                                   bfg_pair const& val)-> bool
                            {
                                return middle.first < val.first;
                            }
                                        );
                */
                if(num_chunk == 1) 
                {    
                    chunk_size = size_of_vec/localities.size();
                    offset     = size_of_vec%localities.size(); 
                }
                else if(num_chunk > 1)
                {
                    chunk_size = size_of_vec/num_chunk;
                    offset     = size_of_vec%num_chunk;
                }
                if(offset > 0)
                {
                    if((offset*(chunk_size+1)) > pos ) 
                        distance = pos/(chunk_size+1);
                    else
                    {
                        distance = offset +
                            ((pos - (offset*(chunk_size+1)))/chunk_size);
                    }
                }
                if(offset == 0) distance =  pos/chunk_size;
                it = base_sf_of_gid_pair_.begin() + distance; 
            }
            else if(state == hpx::dis_state::dis_cyclic)
            { 
                if(num_chunk > 1)
                    it = base_sf_of_gid_pair_.begin()+(pos%num_chunk);
                else if (num_chunk == 1)
                {
                    it = base_sf_of_gid_pair_.begin()+
                            (pos%base_sf_of_gid_pair_.size());
                }
            }
            else if(state == hpx::dis_state::dis_block_cyclic)
            {
                 
                if(num_chunk>1)     
                {
                    it = base_sf_of_gid_pair_.begin()+
                                    ((pos/block_size)%num_chunk);
                }
                else if(num_chunk == 1)
                {
                    it = base_sf_of_gid_pair_.begin()+
                                     ((pos/block_size)%localities.size());
                }
            }
            //  Second condition avoid the boundary case where the get_value can
            //  be called on invalid gid. This occurs when pos = -1
            //  (maximum value)
            if(it->first == pos && (it->second).get() != invalid_id)
            {
                 return it;
            }
            else //It takes care of the case if "it" is at the LAST
            { 
                return (it);
            }
        }//End of get_gid

//        //Note num_chunks == represent then chunk vector index
//        size_future
//            size_helper(size_type num_chunks) const
//        {
//            if(num_chunks < 1)
//            {
//                HPX_ASSERT(num_chunks >= 0);
//                return chunk_vector_stubs::size_async(
//                          ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                      );
//            }
//            else
//                return hpx::lcos::local::dataflow(
//                    [](size_future s1,
//                       size_future s2) -> size_type
//                        {
//                            return s1.get() + s2.get();
//                        },
//                    chunk_vector_stubs::size_async(
//                        ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                         ),
//                    size_helper(num_chunks - 1)
//                                                );
//            }//end of size_helper


        //FASTER VERSION OF SIZE_HELPER

        // PROGRAMMER DOCUMENTATION:
        //  This helper function return the number of element in the hpx::vector.
        //  Here we are dividing the sequence of bfg_pairs into half and
        //  computing the size of the individual chunk_vector and then adding
        //  them. Note this create the binary tree of height equal to log
        //  (num_bfg_pairs in base_sf_of_gid_pair_). Hence it might be efficient
        //  than previous implementation
        //
        // NOTE: This implementation does not need all the chunk_vector of same
        //       size.
        //
        size_future size_helper(vector_type::const_iterator it_begin,
                                vector_type::const_iterator it_end) const
        {
            if((it_end - it_begin) == 1 )
                return chunk_vector_stubs::size_async((it_begin->second).get());
            else
            {
                int mid = (it_end - it_begin)/2;
                size_future left_tree_size = size_helper(it_begin,
                                                         it_begin + mid);
                size_future right_tree_size = hpx::async(
                                                    launch::async,
                                                    hpx::util::bind(
                                                        &vector::size_helper,
                                                        this,
                                                        (it_begin + mid),
                                                        it_end
                                                                    )
                                                        );

            return hpx::lcos::local::dataflow(
                        [](size_future s1, size_future s2) -> size_type
                        {
                            return s1.get() + s2.get();
                        },
                        std::move(left_tree_size),
                        std::move(right_tree_size)
                                            );
            }
        }//end of size_helper

//        size_future
//            max_size_helper(size_type num_chunks) const
//        {
//            if(num_chunks < 1)
//            {
//                HPX_ASSERT(num_chunks >= 0);
//                return chunk_vector_stubs::max_size_async(
//                        ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                                );
//            }
//            else
//                return hpx::lcos::local::dataflow(
//                    [](size_future s1,
//                       size_future s2) -> size_type
//                    {
//                        return s1.get() + s2.get();
//                    },
//                    chunk_vector_stubs::max_size_async(
//                        ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                             ),
//                    max_size_helper(num_chunks - 1)
//                                                );
//            }//end of max_size_helper


        //FASTER VERSION OF MAX_SIZE_HELPER

        // PROGRAMMER DOCUMENTATION:
        //  This helper function return the number of element in the hpx::vector.
        //  Here we are dividing the sequence of bfg_pairs into half and
        //  computing the max_size of the individual chunk_vector and then adding
        //  them. Note this create the binary tree of height. Equal to log
        //  (num_bfg_pairs in base_sf_of_gid_pair_). Hence it might be efficient
        //  than previous implementation
        //
        // NOTE: This implementation does not need all the chunk_vector of same
        //       size.
        //
        size_future max_size_helper(vector_type::const_iterator it_begin,
                                    vector_type::const_iterator it_end) const
        {
            if((it_end - it_begin) == 1 )
                return chunk_vector_stubs::max_size_async(
                                                    (it_begin->second).get()
                                                                );
            else
            {
                int mid = (it_end - it_begin)/2;
                size_future left_tree_size = max_size_helper(it_begin,
                                                             it_begin + mid);
                size_future right_tree_size = hpx::async(
                                                launch::async,
                                                hpx::util::bind(
                                                    &vector::max_size_helper,
                                                    this,
                                                    (it_begin + mid),
                                                    it_end
                                                                )
                                                        );

                return hpx::lcos::local::dataflow(
                            [](size_future s1, size_future s2) -> size_type
                            {
                                return s1.get() + s2.get();
                            },
                            std::move(left_tree_size),
                            std::move(right_tree_size)
                                                 );
            }
        }//end of max_size_helper


//        size_future
//            capacity_helper(size_type num_chunks) const
//        {
//            if(num_chunks < 1)
//            {
//                HPX_ASSERT(num_chunks >= 0);
//                return chunk_vector_stubs::capacity_async(
//                          ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                         );
//            }
//            else
//                return hpx::lcos::local::dataflow(
//                    [](size_future s1,
//                       size_future s2) -> size_type
//                    {
//                        return s1.get() + s2.get();
//                    },
//                    chunk_vector_stubs::capacity_async(
//                        ((base_sf_of_gid_pair_.at(num_chunks)).second).get()
//                                                       ),
//                    capacity_helper(num_chunks - 1)
//                                                );
//            }//end of capacity_helper

        //FASTER VERSION OF CAPACITY_HELPER

        // PROGRAMMER DOCUMENTATION:
        //  This helper function return the number of element in the hpx::vector.
        //  Here we are dividing the sequence of bfg_pairs into half and
        //  computing the capacity of the individual chunk_vector and then adding
        //  them. Note this create the binary tree of height Equal to log
        //  (num_bfg_pairs in base_sf_of_gid_pair_). Hence it might be efficient
        //  than previous implementation.
        //
        // NOTE: This implementation does not need all the chunk_vector of same
        //       size.
        //
        size_future capacity_helper(vector_type::const_iterator it_begin,
                                    vector_type::const_iterator it_end) const
        {
            if((it_end - it_begin) == 1 )
                return chunk_vector_stubs::capacity_async(
                                                    (it_begin->second).get()
                                                          );
            else
            {
                int mid = (it_end - it_begin)/2;
                size_future left_tree_size = capacity_helper(it_begin,
                                                             it_begin + mid);
                size_future right_tree_size = hpx::async(
                                                launch::async,
                                                hpx::util::bind(
                                                    &vector::capacity_helper,
                                                    this,
                                                    (it_begin + mid),
                                                    it_end
                                                                )
                                                        );

                return hpx::lcos::local::dataflow(
                            [](size_future s1, size_future s2) -> size_type
                            {
                                return s1.get() + s2.get();
                            },
                            std::move(left_tree_size),
                            std::move(right_tree_size)
                                                 );
            }
        }//end of capacity_helper

        // PROGRAMMER DOCUMENTATION:
        //   This is the helper function to maintain consistency in the
        //   base_index across all the bfg_pair. It helps for the resize() and
        //   assign() function. This is needed as one necessary condition is the
        //   base_index for bfg_pair must be unique for each chunk_vector.
        //
        void adjust_base_index(vector_type::iterator begin,
                               vector_type::iterator end,
                               size_type new_chunk_size)
        {
            size_type i = 0;
            for(vector_type::iterator it = begin; it != end; it++, i++)
            {
                it->first = i * new_chunk_size;
            }
        }//end of adjust_base_index


    public:

        //
        // Constructors
        //
        /** @brief Default Constructor which create hpx::vector with \a num_chunk = 1
         *          and \a chunk_size = 0. Hence overall size of the vector is 0.
         */
        explicit vector()
        {
            std::vector<hpx::naming::id_type> localities;
            localities.push_back(hpx::find_here());
            state = dis_block;
            create(0, VALUE_TYPE(), hpx::block(localities));    
        }

        //  This is the problem if num_chunk > 1 and chunk_size = 0; thats why
        //  commented. This constructor complicates the push_back operation as
        //  on which gid we have to push back and create function as all base
        //  are same
//        explicit vector(size_type num_chunks)
//        {
//            create(num_chunks, 0, VALUE_TYPE());
//        }

        /** @brief Constructor which create hpx::vector with size \a chunk_size.
         *          This use default constructor to initialized object.
         *
         *  @param num_chunks   The number of chunks to be created
         *  @param chunk_size   The size of each chunk
         */
        explicit vector(size_type big_chunk)
        {
            // If num_chunks = 0 no operation can be carried on that vector as
            // every further operation throw exception and if ( num_chunks > 1
            // && chunk_size == 0 ) then it violates the condition that the base
            // index should be unique
            /*if(num_chunks == 0 || (num_chunks > 1 && chunk_size == 0) )
                HPX_THROW_EXCEPTION(
                    hpx::invalid_vector_error,
                    "vector",
                    "Invalid Vector: num_chunks, chunk_size should be greater than zero"
                    );
            */
            std::vector<hpx::naming::id_type> localities;
            localities.push_back(hpx::find_here()); 
            state = dis_block;
            create(big_chunk, VALUE_TYPE(), hpx::block(localities));
        }

        /** @brief Constructor which create and initialize vector with all
         *          elements with \a val.
         *
         *  @param num_chunks   The number of chunks to be created
         *  @param chunk_size   The size of each chunk
         *  @param val          Default value for the element in vector
         */
        explicit vector(size_type big_chunk,
                        VALUE_TYPE val)
        {
            /*if(num_chunks == 0 || (num_chunks > 1 && chunk_size == 0))
                HPX_THROW_EXCEPTION(
                    hpx::invalid_vector_error,
                    "vector",
                    "Invalid Vector: num_chunks, chunk_size should be greater than zero"
                                    );
             */
            std::vector<hpx::naming::id_type> localities;
            localities.push_back(hpx::find_here()); 
            state = dis_block;
            create(big_chunk, val, hpx::block(localities));
        }
        
         template<typename T>
         explicit vector(size_type big_chunk,
                         T policy)
         {
            if(policy.get_locality_size() == 0 )
            {
                std::vector<hpx::naming::id_type> localities;
                localities.push_back(hpx::find_here());
                policy.set_locality(localities);
            }
            if(policy.get_policy() == "block") state = dis_block;
            else if(policy.get_policy() == "cyclic") state = dis_cyclic; 
            else if(policy.get_policy() == "block_cyclic") 
            {
                state = dis_block_cyclic;
            }
            create(big_chunk, VALUE_TYPE(), policy);               


         } 
        /** @brief Constructor which create and initialize vector with all
         *          elements with \a val.
         *
         *  @param num_chunks   The number of chunks to be created
         *  @param chunk_size   The size of each chunk
         *  @param val          Default value for the element in vector
         */
        template<typename T>
        explicit vector(size_type big_chunk,
                        VALUE_TYPE val,
                        T  policy)
        {
            /*if(num_chunks == 0 || (num_chunks > 1 && chunk_size == 0))
                HPX_THROW_EXCEPTION(
                    hpx::invalid_vector_error,
                    "vector",
                    "Invalid Vector: num_chunks, chunk_size should be greater than zero"
                                    );
             */
            if(policy.get_locality_size() == 0 )
            {
                std::vector<hpx::naming::id_type> localities;
                localities.push_back(hpx::find_here());
                policy.set_locality(localities);
            } 
            if(policy.get_policy() == "block") state = dis_block;
            else if(policy.get_policy() == "cyclic") state = dis_cyclic;
            else if(policy.get_policy() == "block_cyclic") 
            {
                state = dis_block_cyclic;
            }
            create(big_chunk, val, policy);
        }
        /** @brief Copy Constructor for vector.
         *
         *  @param other    This the hpx::vector object which is to be copied
         */
        explicit vector(self_type const& other) //Copy Constructor
        {
            this->base_sf_of_gid_pair_ = other.base_sf_of_gid_pair_;
        }

        //
        // OVERLOADED OPERATOR API's
        //

        /** @brief Array subscript operator. This does not throw any exception.
         *
         *  @param pos Position of the element in the vector [Note the first
         *              position in the chunk is 0]
         *
         *  @return Return the value of the element at position represented by
         *           \a pos [Note that this is not the reference to the element]
         *
         */
        VALUE_TYPE operator [](size_type pos) const
        {
            vector_type::const_iterator it = get_base_gid_pair(pos);
            if(state == hpx::dis_state::dis_block) 
                return chunk_vector_stubs::get_value_noexpt_async(
                                                        (it->second).get(),
                                                        (pos - (it->first))
                                                             ).get();
             else if(state == hpx::dis_state::dis_cyclic)
             {
                 if(num_chunk > 1)
                 {
                     return chunk_vector_stubs::get_value_noexpt_async(
                                                        (it->second).get(),
                                                        (pos/num_chunk)
                                                            ).get();
                 } 
                 else if (num_chunk ==1)
                 {
                     return chunk_vector_stubs::get_value_noexpt_async(
                                                        (it->second).get(),
                                                        (pos/localities.size())
                                                            ).get();
                 }
             }
             else if(state == hpx::dis_state::dis_block_cyclic) 
             {
                 return chunk_vector_stubs::get_value_noexpt_async(
                                                         (it->second).get(),
                                                         (pos%block_size) 
                                                            ).get();
             } 
        }
        

        /** @brief Copy assignment operator.
         *
         *  @param other    This the hpx::vector object which is to be copied
         *
         *  @return This return the reference to the newly created vector
         */
        self_type& operator=(self_type const& other)
        {
            this->base_sf_of_gid_pair_ = other.base_sf_of_gid_pair_;
            return *this;
        }


        //
        // Capacity related API's in vector class
        //

        //SIZE
        /** @brief Compute the size as the number of elements it contains.
         *
         *  @return Return the number of elements in the vector
         */
        size_type size() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return size_helper(base_sf_of_gid_pair_.begin(),
                               base_sf_of_gid_pair_.end() - 1
                              ).get();
        }

        /** @brief Asynchronous API for size().
         *
         * @return This return the hpx::future of return value of size()
         */
        size_future size_async() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return size_helper(base_sf_of_gid_pair_.begin(),
                               base_sf_of_gid_pair_.end() - 1);
        }

        //MAX_SIZE
        /**  @brief Compute the maximum size of hpx::vector in terms of
         *           number of elements.
         *  @return Return maximum number of elements the vector can hold
         */
        size_type max_size() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return max_size_helper(base_sf_of_gid_pair_.begin(),
                                   base_sf_of_gid_pair_.end() - 1
                                   ).get();
        }

        /**  @brief Asynchronous API for max_size().
         *
         *  @return Return the hpx::future of return value of max_size()
         */
        size_future max_size_async() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return max_size_helper(base_sf_of_gid_pair_.begin(),
                                   base_sf_of_gid_pair_.end() - 1);
        }

//            //RESIZE (without value)
//
//            void resize(size_type n)
//            {
//                if(n == 0)
//                    HPX_THROW_EXCEPTION(hpx::invalid_vector_error,
//                                        "resize",
//                                        "Invalid Vector: new_chunk_size should be greater than zero");
//
//                std::vector<void_future> resize_lazy_sync;
//                //Resizing the vector chunks
//                //AS we have to iterate until we hit LAST
//                BOOST_FOREACH(bfg_pair const& p, std::make_pair(base_sf_of_gid_pair_.begin(), base_sf_of_gid_pair_.end() - 1) )
//                {
//                    resize_lazy_sync.push_back(chunk_vector_stubs::resize_async((p.second).get(), n));
//                }
//                HPX_ASSERT(base_sf_of_gid_pair_.size() > 1); //As this function changes the size we should have LAST always.
//                //waiting for the resizing
//                hpx::wait_all(resize_lazy_sync);
//                adjust_base_index(base_sf_of_gid_pair_.begin(), base_sf_of_gid_pair_.end() - 1, n);
//            }
//            void_future resize_async(size_type n)
//            {
//                //static_cast to resolve ambiguity of the overloaded function
//                return hpx::async(launch::async, hpx::util::bind(static_cast<void(vector::*)(std::size_t)>(&vector::resize), this, n));
//            }

        // RESIZE (with value)
        // SEMANTIC DIFFERENCE:
        //    It is resize with respective chunk not whole vector
        /** @brief Resize each chunk so that it contain n elements. If
         *          the \a val is not it use default constructor instead.
         *
         *  This function resize the each chunk so that it contains \a n
         *   elements. [Note that the \a n does not represent the total size of
         *   vector it is the size of each chunk. This mean if \a n is 10 and
         *   num_chunk is 5 then total size of vector after resize is 10*5 = 50]
         *
         *  @param n    New size of the each chunk
         *  @param val  Value to be copied if \a n is greater than the current
         *               size [Default is 0]
         *
         *  @exception hpx::invalid_vector_error If the \a n is equal to zero
         *              then it throw \a hpx::invalid_vector_error exception.
         */
        void resize(size_type n, VALUE_TYPE const& val = VALUE_TYPE())
        {
            if(n == 0)
                HPX_THROW_EXCEPTION(
                    hpx::invalid_vector_error,
                    "resize",
                    "Invalid Vector: new_chunk_size should be greater than zero"
                                    );

            std::vector<void_future> resize_lazy_sync;
            BOOST_FOREACH(bfg_pair const& p,
                          std::make_pair(base_sf_of_gid_pair_.begin(),
                                         base_sf_of_gid_pair_.end() - 1)
                         )
            {
                resize_lazy_sync.push_back(
                                chunk_vector_stubs::resize_async(
                                                        (p.second).get(),
                                                         n,
                                                         val)
                                           );
            }
            hpx::wait_all(resize_lazy_sync);

            //To maintain the consistency in the base_index of each bfg_pair.
            adjust_base_index(base_sf_of_gid_pair_.begin(),
                              base_sf_of_gid_pair_.end() - 1,
                              n);
        }

        /** @brief Asynchronous API for resize().
         *
         *  @param n    New size of the each chunk
         *  @param val  Value to be copied if \a n is greater than the current size
         *               [Default is 0]
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed or
         *           not]
         *
         *  @exception hpx::invalid_vector_error If the \a n is equal to zero
         *              then it throw \a hpx::invalid_vector_error exception.
         */
        void_future resize_async(size_type n,
                                 VALUE_TYPE const& val = VALUE_TYPE())
        {
            //static_cast to resolve ambiguity of the overloaded function
            return hpx::async(launch::async,
                              hpx::util::bind(
                                static_cast<
                                void(vector::*)(size_type,
                                                VALUE_TYPE const&)
                                            >
                                            (&vector::resize),
                                              this,
                                              n,
                                              val)
                              );
        }

        //CAPACITY

        /** @brief Compute the size of currently allocated storage capacity for
         *          vector.
         *
         *  @return Returns capacity of vector, expressed in terms of elements
         */
        size_type capacity() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return capacity_helper(base_sf_of_gid_pair_.begin(),
                                   base_sf_of_gid_pair_.end() - 1
                                   ).get();
        }

        /** @brief Asynchronous API for capacity().
         *
         *  @return Returns the hpx::future of return value of capacity()
         */
        size_future capacity_async() const
        {
            HPX_ASSERT(base_sf_of_gid_pair_.size() > 1);
            //Here end -1 is because we have the LAST in the vector
            return capacity_helper(base_sf_of_gid_pair_.begin(),
                                   base_sf_of_gid_pair_.end() - 1);
        }

        //EMPTY
        /** @brief Return whether the vector is empty.
         *
         *  @return Return true if vector size is 0, false otherwise
         */
        bool empty() const
        {
            return !(this->size());
        }

        /** @brief Asynchronous API for empty().
         *
         *  @return The hpx::future of return value empty()
         */
        bool_future empty_async() const
        {
            return hpx::async(launch::async,
                              hpx::util::bind(&vector::empty, this));
        }

        //RESERVE
        /** @brief Request the change in each chunk capacity so that it
         *          can hold \a n elements. Throws the
         *          \a hpx::length_error exception.
         *
         *  This function request for each chunk capacity should be at
         *   least enough to contain \a n elements. For all chunk in vector
         *   if its capacity is less than \a n then their reallocation happens
         *   to increase their capacity to \a n (or greater). In other cases
         *   the chunk capacity does not got affected. It does not change the
         *   chunk size. Hence the size of the vector does not affected.
         *
         * @param n Minimum capacity of chunk
         *
         * @exception hpx::length_error If \a n is greater than maximum size for
         *             at least one chunk then function throw
         *             \a hpx::length_error exception.
         */
        void reserve(size_type n)
        {
            std::vector<void_future> reserve_lazy_sync;
            BOOST_FOREACH(bfg_pair const& p,
                          std::make_pair(base_sf_of_gid_pair_.begin(),
                                         base_sf_of_gid_pair_.end() - 1)
                          )
            {
                reserve_lazy_sync.push_back(
                        chunk_vector_stubs::reserve_async((p.second).get(), n)
                                            );
            }
            hpx::wait_all(reserve_lazy_sync);
        }

        /** @brief Asynchronous API for reserve().
         *
         *  @param n Minimum capacity of chunk
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed or
         *           not]
         *
         *  @exception hpx::length_error If \a n is greater than maximum size
         *              for at least one chunk then function throw
         *              \a hpx::length_error exception.
         */
        void_future reserve_async(size_type n)
        {
            return hpx::async(launch::async,
                              hpx::util::bind(&vector::reserve, this, n));
        }


        //
        //  Element access API's in vector class
        //

        //GET_VALUE
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
        VALUE_TYPE get_value(size_type pos) const
        {
            try{
                vector_type::const_iterator it = get_base_gid_pair(pos);
                if(state == hpx::dis_state::dis_block)
                    return chunk_vector_stubs::get_value_async(
                                                           (it->second).get(),
                                                           (pos - (it->first))
                                                          ).get();
                else if(state == hpx::dis_state::dis_cyclic)
                {
                    if(num_chunk > 1)
                    {
                        return chunk_vector_stubs::get_value_async(
                                                        (it->second).get(),
                                                        (pos/num_chunk)
                                                               ).get();
                    }
                    else if(num_chunk == 1)
                    {
                        return chunk_vector_stubs::get_value_async(
                                                        (it->second).get(),
                                                        (pos/localities.size())
                                                              ).get();
                    }
                }
                else if(state == hpx::dis_state::dis_block_cyclic)
                {
                   if(num_chunk >1)
                   {
                       return chunk_vector_stubs::get_value_async(
                                                     (it->second).get(),
                                                     ((((pos/block_size)/
                                                     num_chunk)*block_size)+
                                                     (pos%block_size))
                                                               ).get();
                   }
                   else if(num_chunk == 1)
                   {
                       return chunk_vector_stubs::get_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size))
                                                               ).get();
                    }
                }
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(hpx::out_of_range,
                                    "get_value",
                                    "Value of 'pos' is out of range");
            }
        }//end of get_value

        /** @brief Asynchronous API for get_value(). It throws the
         *          \a hpx::out_of_range exception.
         *
         *  @param pos Position of the element in the vector [Note the first
         *          position in the chunk is 0]
         *
         *  @return Return the hpx::future to value of the element at position
         *           represented by \a pos [Note that this is not the reference
         *           to the element]
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        hpx::future< VALUE_TYPE > get_value_async(size_type pos) const
        {
            // Here you can call the get_val_sync API but you have already an
            // API to do that which reduce one function call
            try{
                vector_type::const_iterator it = get_base_gid_pair(pos);
                if(state == hpx::dis_state::dis_block)
                {
                    return chunk_vector_stubs::get_value_async(
                                                        (it->second).get(),
                                                        (pos - (it->first))
                                                          );
                } 
                else if(state == hpx::dis_state::dis_cyclic)
                {
                    if(num_chunk > 1)
                    {
                        return chunk_vector_stubs::get_value_async(
                                                        (it->second).get(),
                                                        (pos/num_chunk)
                                                               );
                    }
                    else if(num_chunk == 1)
                    {
                        return chunk_vector_stubs::get_value_async(
                                                        (it->second).get(),
                                                        (pos/localities.size())
                                                              );
                    }
                }
                else if(state == hpx::dis_state::dis_block_cyclic)
                {
                   if(num_chunk >1)
                   {
                       return chunk_vector_stubs::get_value_async(
                                                     (it->second).get(),
                                                     ((((pos/block_size)/
                                                     num_chunk)*block_size)+
                                                     (pos%block_size))
                                                               );
                   } 
                   else if(num_chunk == 1)
                   {
                       return chunk_vector_stubs::get_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size))
                                                               );
                   }
                }
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(
                    hpx::out_of_range,
                    "get_value_async",
                    "Value of 'pos' is out of range");
            }
        }//end of get_value_async

        //FRONT (never throws exception)
        /** @brief Access the value of first element in the vector.
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return the value of the first element in the vector
         */
        VALUE_TYPE front() const
        {
            return chunk_vector_stubs::front_async(
                                    (base_sf_of_gid_pair_.front().second).get()
                                                  ).get();
        }//end of front_value

        /** @brief Asynchronous API for front().
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return the hpx::future to return value of front()
         */
        hpx::future< VALUE_TYPE > front_async() const
        {
            return chunk_vector_stubs::front_async(
                                    (base_sf_of_gid_pair_.front().second).get()
                                                  );
        }//end of front_async

        //BACK (never throws exception)
        /** @brief Access the value of last element in the vector.
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return the value of the last element in the vector
         */
        VALUE_TYPE back() const
        {
            // As the LAST pair is there and then decrement operator to that
            // LAST is undefined hence used the end() function rather than back()
            return chunk_vector_stubs::back_async(
                            ((base_sf_of_gid_pair_.end() - 2)->second).get()
                                                 ).get();
        }//end of back_value

        /** @brief Asynchronous API for back().
         *
         *  Calling the function on empty container cause undefined behavior.
         *
         * @return Return hpx::future to the return value of back()
         */
        hpx::future< VALUE_TYPE > back_async() const
        {
            //As the LAST pair is there
            return chunk_vector_stubs::back_async(
                            ((base_sf_of_gid_pair_.end() - 2)->second).get()
                                                 );
        }//end of back_async



        //
        // Modifier component action
        //

        //ASSIGN
        /** @brief Assigns new contents to each chunk, replacing its
         *          current contents and modifying each chunk size
         *          accordingly.
         *
         *  @param n     New size of each chunk
         *  @param val   Value to fill the chunk with
         *
         *  @exception hpx::invalid_vector_error If the \a n is equal to zero
         *              then it throw \a hpx::invalid_vector_error exception.
         */
        void assign(size_type n, VALUE_TYPE const& val)
        {
            if(n == 0)
                HPX_THROW_EXCEPTION(
                    hpx::invalid_vector_error,
                    "assign",
                    "Invalid Vector: new_chunk_size should be greater than zero"
                                    );

            std::vector<void_future> assign_lazy_sync;
            BOOST_FOREACH(bfg_pair const& p,
                          std::make_pair(base_sf_of_gid_pair_.begin(),
                                         base_sf_of_gid_pair_.end() - 1)
                          )
            {
                assign_lazy_sync.push_back(
                    chunk_vector_stubs::assign_async((p.second).get(), n, val)
                                          );
            }
            hpx::wait_all(assign_lazy_sync);
            adjust_base_index(base_sf_of_gid_pair_.begin(),
                              base_sf_of_gid_pair_.end() - 1,
                              n);
        }//End of assign

        /** @brief Asynchronous API for assign().
         *
         *  @param n     New size of each chunk
         *  @param val   Value to fill the chunk with
         *
         *  @exception hpx::invalid_vector_error If the \a n is equal to zero
         *              then it throw \a hpx::invalid_vector_error exception.
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed or
         *           not]
         */
        void_future assign_async(size_type n, VALUE_TYPE const& val)
        {
            return hpx::async(launch::async,
                              hpx::util::bind(&vector::assign,
                                              this,
                                              n,
                                              val)
                              );
        }

        //PUSH_BACK
        /** @brief Add new element at the end of vector. The added element
         *          contain the \a val as value.
         *
         *  The value is added to the back to the last chunk.
         *
         *  @param val Value to be copied to new element
         */
        void push_back(VALUE_TYPE const& val)
        {
            chunk_vector_stubs::push_back_async(
                            ((base_sf_of_gid_pair_.end() - 2 )->second).get(),
                                                val
                                                ).get();
        }

        /** @brief Asynchronous API for push_back().
         *
         *  @param val Value to be copied to new element
         *
         *  @return This return the hpx::future of type void [The void return
         *           type can help to check whether the action is completed or
         *           not]
         */
        void_future push_back_async(VALUE_TYPE const& val)
        {
            return chunk_vector_stubs::push_back_async(
                            ((base_sf_of_gid_pair_.end() - 2)->second).get(),
                                                        val
                                                        );
        }

        //PUSH_BACK (With rval)
        /** @brief Add new element at the end of vector. The added element
         *          contain the \a val as value.
         *
         *  The value is added to the back to the last chunk.
         *
         *  @param val Value to be moved to new element
         */
        void push_back(VALUE_TYPE const&& val)
        {
            chunk_vector_stubs::push_back_rval_async(
                            ((base_sf_of_gid_pair_.end() - 2)->second).get(),
                                                    std::move(val)
                                                     ).get();
        }

        /** @brief Asynchronous API for push_back(VALUE_TYPE const&& val).
         *
         *  @param val Value to be moved to new element
         */
        void_future push_back_async(VALUE_TYPE const&& val)
        {
            return chunk_vector_stubs::push_back_rval_async(
                            ((base_sf_of_gid_pair_.end() - 2)->second).get(),
                                                    std::move(val)
                                                            );
        }

        //POP_BACK (Never throw exception)
//            void pop_back()
//            {
//                chunk_vector_stubs::pop_back_async(( (base_sf_of_gid_pair_.end() - 2)->second).get()).get();
//                //TODO if following change the affect back() and further pop_back function
//                //checking if last element from the particular gid is popped up then delete that..
//                // (-2)I am retaining one gid in vector as otherwise it goes to invalid state and it makes a compulsion that we need to keep at least one element that is not good
//                if(chunk_vector_stubs::empty_async(( (base_sf_of_gid_pair_.end() - 2)->second).get()).get() && base_sf_of_gid_pair_.size() > 2)
//                    base_sf_of_gid_pair_.pop_back();
//                HPX_ASSERT(base_sf_of_gid_pair_.size() > 1); //As this function changes the size we should have LAST always.
//            }

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
        void set_value(size_type pos, VALUE_TYPE const& val)
        {
            try{
                vector_type::const_iterator it = get_base_gid_pair(pos);
                
                if(state == hpx::dis_state::dis_block) 
                    return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos - (it->first)),
                                                         val
                                                              ).get();
                 else if(state == hpx::dis_state::dis_cyclic)
                 {
                     if(num_chunk > 1)
                         return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos/num_chunk),
                                                        val
                                                               ).get();
                     else if(num_chunk == 1)
                          return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos/localities.size()),
                                                        val
                                                              ).get();

                 }
                 else if(state == hpx::dis_state::dis_block_cyclic)
                 {
                    if(num_chunk >1)
                       return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                      num_chunk)*block_size)+
                                                      (pos%block_size)),
                                                        val
                                                               ).get();
                    else if(num_chunk == 1)
                       return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size)),
                                                        val
                                                               ).get();
 
                }
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(hpx::out_of_range,
                                    "set_value",
                                    "Value of 'pos' is out of range"
                                    );
            }
        }//end of set_value

        /** @brief Asynchronous API for set_value(). It throws the
         *          \a hpx::out_of_range exception.
         *
         *  @param pos   Position of the element in the vector [Note the first
         *                position in the vector is 0]
         *  @param val   The value to be copied
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        void_future set_value_async(size_type pos, VALUE_TYPE const& val)
        {
            try{
                // This reduce one function call as we are directly calling
                // chunk vector API
                vector_type::const_iterator it = get_base_gid_pair(pos);
               if(state == hpx::dis_state::dis_block) 
                    return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos - (it->first)),
                                                        val   );
               else if(state == hpx::dis_state::dis_cyclic)
               {
                  if(num_chunk > 1)  
                     return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos/num_chunk),
                                                        val   );
                  else if (num_chunk == 1)
                      return chunk_vector_stubs::set_value_async(
                                                        (it->second).get(),
                                                        (pos/localities.size()),
                                                        val   );
 
               }
               else if(state == hpx::dis_state::dis_block_cyclic)
               {
                   if(num_chunk >1)
                      return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                      num_chunk)*block_size)+
                                                      (pos%block_size)),
                                                        val
                                                               );
                   else if(num_chunk == 1)
                       return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size)),
                                                        val
                                                               );
 
               }
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(hpx::out_of_range,
                                    "set_value_async",
                                    "Value of 'pos' is out of range");
            }
        }//end of set_value_async

        //SET_VALUE (with rval)
        /** @brief Move the val in the element at position \a pos in the vector
         *          container. It throws the \a hpx::out_of_range exception.
         *
         *  @param pos   Position of the element in the vector [Note the
         *                first position in the vector is 0]
         *  @param val   The value to be moved
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        void set_value(size_type pos, VALUE_TYPE const&& val)
        {
           try{
                vector_type::const_iterator it = get_base_gid_pair(pos);
                if(state == hpx::dis_state::dis_block)
                {

                    return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos - (it->first)),
                                                    std::move(val)
                                                                 ).get();
                }
                else if (state == hpx::dis_state::dis_cyclic)
                {
                    if(num_chunk > 1)  
                       return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos/num_chunk),
                                                    std::move(val)
                                                                ).get();
                     else if(num_chunk == 1)
                        return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos/localities.size()),
                                                    std::move(val)
                                                                ).get();

                
                }
               else if(state == hpx::dis_state::dis_block_cyclic)
               {
                   if(num_chunk >1)
                      return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                      num_chunk)*block_size)+
                                                      (pos%block_size)),
                                                        std::move(val)
                                                               ).get();
                   else if(num_chunk == 1)
                       return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size)),
                                                        std::move(val)
                                                               ).get();

                }
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(hpx::out_of_range,
                                    "set_value",
                                    "Value of 'pos' is out of range");
            }
        }//end of set_value

        /** @brief Asynchronous API for
         *          set_value(std::size_t pos, VALUE_TYPE const&& val).
         *          It throws the \a hpx::out_of_range exception.
         *
         *  @param pos   Position of the element in the vector [Note the
         *                first position in the vector is 0]
         *  @param val   The value to be moved
         *
         *  @exception hpx::out_of_range The \a pos is bound checked and if
         *              \a pos is out of bound then it throws the
         *              \a hpx::out_of_range exception.
         */
        void_future set_value_async(size_type pos, VALUE_TYPE const&& val)
        {
            try{
                vector_type::const_iterator it = get_base_gid_pair(pos);
                if(state == hpx::dis_state::dis_block)
                    return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos - (it->first)),
                                                        std::move(val)
                                                                 );
                else if (state == hpx::dis_state::dis_cyclic)
                {
                    if(num_chunk > 1)
                      return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos/num_chunk),
                                                    std::move(val)
                                                                );
   
                    else if(num_chunk == 1)
                        return chunk_vector_stubs::set_value_rval_async(
                                                    (it->second).get(),
                                                    (pos/localities.size()),
                                                    std::move(val)
                                                                );
 
               }
               else if(state == hpx::dis_state::dis_block_cyclic)
               {
                   if(num_chunk >1)
                      return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                      num_chunk)*block_size)+
                                                      (pos%block_size)),
                                                        std::move(val)
                                                               );
                   else if(num_chunk == 1)
                       return chunk_vector_stubs::set_value_async(
                                                      (it->second).get(),
                                                      ((((pos/block_size)/
                                                 localities.size())*block_size)+
                                                      (pos%block_size)),
                                                        std::move(val)
                                                               );
               }
              
            }
            catch(hpx::exception const& /*e*/){
                HPX_THROW_EXCEPTION(hpx::out_of_range,
                                    "set_value_async",
                                    "Value of 'pos' is out of range");
            }
        }//end of set_value_async



            //CLEAR
            //TODO if number of chunks is kept constant every time then clear should modified (clear each chunk_vector one by one).
//            void clear()
//            {
//                //It is keeping one gid hence iterator does not go in an invalid state
//                base_sf_of_gid_pair_.erase(base_sf_of_gid_pair_.begin() + 1,
//                                           base_sf_of_gid_pair_.end()-1);
//                chunk_vector_stubs::clear_async((base_sf_of_gid_pair_[0].second).get()).get();
//                HPX_ASSERT(base_sf_of_gid_pair_.size() > 1); //As this function changes the size we should have LAST always.
//            }

            //
            // HPX CUSTOM API's
            //

//            //CREATE_CHUNK
//            //TODO This statement can create Data Inconsistency :
//             //If size of base_sf_of_gid_pair_ calculated and added to the base_index but not whole creation is completed and in betwen this som push_back on hpx::vector is done then that operation is losted
//            void create_chunk(hpx::naming::id locality, std::size_t chunk_size = 0, VALUE_TYPE val = 0.0)
//            {
//                base_sf_of_gid_pair_.push_back(
//                        std::make_pair(
//                            base_sf_of_gid_pair_.size(),
//                             hpx::components::new_<chunk_vector_type>(locality, chunk_size, val)
//                                      )
//                                                    );
//            }//end of create chunk

            //
            // Iterator API's in vector class
            //

        // PROGRAMMER DOCUMENTATION:
        //  The beginning id represented by the 0'th position of the first
        //   chunk_vector.
        //
        /**  @brief Return the iterator at the beginning of the vector. */
        iterator begin()
        {
            return iterator(base_sf_of_gid_pair_.begin(), 0, valid);
        }//end of begin

        /**  @brief Return the const_iterator at the beginning of the vector.*/
        const_iterator begin() const
        {
            return const_iterator(base_sf_of_gid_pair_.begin(), 0, valid);
        }//end of begin

        // PROGRAMMER DOCUMENTATION:
        //   The end of vector is represented by the last position
        //   (std::vector's last iterator position) of the chunk_vector in
        //   bfg_pair which immediately precedes the LAST (for LAST Refer
        //   the Create PROGRAMMER DOCUMENTATION).
        /**  @brief Return the iterator at the end of the vector. */
        iterator end()
        {
            return iterator((base_sf_of_gid_pair_.end() - 2),
                            chunk_vector_stubs::size_async(
                                ((base_sf_of_gid_pair_.end() - 2)->second).get()
                                                           ).get(),
                            valid);
        }//end of end

        /**  @brief Return the const_iterator at the end of the vector. */
        const_iterator end() const
        {
            return const_iterator((base_sf_of_gid_pair_.end() - 2),
                            chunk_vector_stubs::size_async(
                                ((base_sf_of_gid_pair_.end() - 2)->second).get()
                                                           ).get(),
                            valid);
        }//end of end

        // PROGRAMMER DOCUMENTATION:
        //  The beginning id represented by the 0'th position of the first
        //   chunk_vector.
        //
        /**  @brief Return the const_iterator at the beginning of the vector. */
        const_iterator cbegin() const
        {
            return const_iterator(base_sf_of_gid_pair_.begin(), 0, valid);
        }//end of cbegin

        // PROGRAMMER DOCUMENTATION:
        //   The end of vector is represented by the last position
        //   (std::vector's last iterator position) of the chunk_vector in
        //   bfg_pair which immediately precedes the LAST (for LAST Refer
        //   the Create PROGRAMMER DOCUMENTATION).
        /**  @brief Return the const_iterator at the end of the vector. */
        const_iterator cend() const
        {
            return const_iterator((base_sf_of_gid_pair_.end() - 2),
                            chunk_vector_stubs::size_async(
                                ((base_sf_of_gid_pair_.end() - 2)->second).get()
                                                           ).get(),
                            valid);
        }//end of cend

        //
        // Destructor
        //
        /**  @brief Default destructor to the vector class. */
        ~vector()
        {
            //DEFAULT destructor
        }


    private:
        //PROGRAMMER DOCUMENTATION:
        // This is the vector representing the base_index and corresponding
        //  global ID's of chunk_vector. Taken as future of hpx_id's as it
        //  delay the .get() as far as possible. Shared future is mandatory
        //  as .get() can be called any number of time. The base_index value
        //  for each bfg_pair must be unique.

        /** @brief This is the vector representing the base_index and
         *          corresponding global ID's of chunk.
         */
        vector_type base_sf_of_gid_pair_;

    };//end of class vector

} // end of hpx namespace

#endif // VECTOR_HPP
