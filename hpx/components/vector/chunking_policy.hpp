//  Copyright (c) 2014 Bibek Ghimire
//
#ifndef BLOCK_CHUNKING_POLICY_HPP
#define BLOCK_CHUNKING_POLICY_HPP

#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

namespace hpx
{

    struct block_chunking_policy
    {
    /////////////////////////////////////////////////////////////
    // This class help to specify the block chunking policy user 
    // want there distributed data structure to be 
    // 

    public:
       block_chunking_policy(){num_chunk = 1; } 
         
       block_chunking_policy operator()(std::size_t num_chunk_) const
       { 
           return block_chunking_policy(num_chunk_);                           
       }

       block_chunking_policy operator()(std::vector<hpx::naming::id_type> loc) const
       {
           return block_chunking_policy( loc);
       }

       block_chunking_policy operator()(std::size_t num_chunk_, 
                                        std::vector<hpx::naming::id_type> loc) const
       {
           return block_chunking_policy(num_chunk_, loc);
       }

       std::vector<hpx::naming::id_type> get_localities() const
       {
           return localities;
       }  
       std::size_t get_block_size()
       {
           return big_chunk;
       }       
 
 
       std::size_t get_num_chunk() const
       {
           return num_chunk;
       }
  
       std::string get_policy() const
       {
           return "block";
       }
    
       std::size_t get_big_chunk() const
       {
           return big_chunk;
       }

       void set_big_chunk(std::size_t b_chunk)
       {
           big_chunk = b_chunk;
       }
       
       void set_locality(std::vector<hpx::naming::id_type> localities_)
       {
           localities = localities_;   
       }
       
       std::size_t get_locality_size()
       {

           return localities.size();
       }

       template<class Archive>
       void serialize(Archive & ar, const unsigned int version)
       {
            ar & big_chunk & num_chunk & localities;
       }


    private:
      
       block_chunking_policy(std::size_t num_chunk_, 
                             std::vector<hpx::naming::id_type> loc)
         : num_chunk(num_chunk_),localities(loc)
       {} 
       
       block_chunking_policy(std::vector<hpx::naming::id_type> loc)
         : num_chunk(1),localities(loc)
       {}     
       
       block_chunking_policy(std::size_t num_chunk_)
         : num_chunk(num_chunk_)
       {}
       
       std::size_t big_chunk;
       std::size_t num_chunk;
       std::vector<hpx::naming::id_type> localities; 

    };

    static block_chunking_policy const block;


    struct cyclic_chunking_policy
    {
    /////////////////////////////////////////////////////////////
    // This class help to specify the block chunking policy user 
    // want there distributed data structure to be 
    // 

    public:
       cyclic_chunking_policy(){num_chunk = 1;} 
         
       cyclic_chunking_policy operator()(std::size_t num_chunk_) const
       { 
           return cyclic_chunking_policy(num_chunk_);                           
       }

       cyclic_chunking_policy operator()(std::vector<hpx::naming::id_type> loc) const
       {
           return cyclic_chunking_policy( loc);
       }

       cyclic_chunking_policy operator()(std::size_t num_chunk_, 
                                        std::vector<hpx::naming::id_type> loc) const
       {
           return cyclic_chunking_policy(num_chunk_, loc);
       }

       std::vector<hpx::naming::id_type> get_localities() const
       {
           return localities;
       }  
       std::size_t get_block_size()
       {
           return 1;
       }       
 
 
       std::size_t get_num_chunk() const
       {
           return num_chunk;
       }
  
       std::string get_policy() const
       {
           return "cyclic";
       }
    
       std::size_t get_big_chunk() const
       {
           return big_chunk;
       }

       void set_big_chunk(std::size_t b_chunk)
       {
           big_chunk = b_chunk;
       }
       
       void set_locality(std::vector<hpx::naming::id_type> localities_)
       {
           localities = localities_;   
       }
       
       std::size_t get_locality_size()
       {

           return localities.size();
       }
    private:
      
       cyclic_chunking_policy(std::size_t num_chunk_, 
                             std::vector<hpx::naming::id_type> loc)
         : num_chunk(num_chunk_),localities(loc)
       {} 
       
       cyclic_chunking_policy(std::vector<hpx::naming::id_type> loc)
         : num_chunk(1),localities(loc)
       {}     
       
       cyclic_chunking_policy(std::size_t num_chunk_)
         : num_chunk(num_chunk_)
       {}
       
       std::size_t big_chunk;
       std::size_t num_chunk;
       std::vector<hpx::naming::id_type> localities; 
    };

    static cyclic_chunking_policy const cyclic;


    struct block_cyclic_chunking_policy
    {
    /////////////////////////////////////////////////////////////
    // This class help to specify the block chunking policy user 
    // want there distributed data structure to be 
    // 

    public:
       block_cyclic_chunking_policy(){}
               
       block_cyclic_chunking_policy operator()(std::size_t block_size_) const
       { 
           return block_cyclic_chunking_policy(block_size_);                           
       }

       block_cyclic_chunking_policy operator()(
           std::size_t block_size_, 
           std::vector<hpx::naming::id_type> loc) const
       {
           return block_cyclic_chunking_policy(block_size_, loc);
       }

       block_cyclic_chunking_policy operator()(
           std::size_t block_size_, 
           std::size_t num_chunk_,           
           std::vector<hpx::naming::id_type> loc) const
       {
           return block_cyclic_chunking_policy(block_size_, num_chunk_, loc);
       }

       std::vector<hpx::naming::id_type> get_localities() const
       {
           return localities;
       }  
      
       std::size_t get_num_chunk() const
       {
           return num_chunk;
       }
  
       std::string get_policy() const
       {
           return "block_cyclic";
       }
       
       std::size_t get_block_size()
       {
           return block_size;
       }       
 
       std::size_t get_big_chunk() const
       {
           return big_chunk;
       }

       void set_big_chunk(std::size_t b_chunk)
       {
           big_chunk = b_chunk;
       }
       
       void set_locality(std::vector<hpx::naming::id_type> localities_)
       {
           localities = localities_;   
       }
       
       std::size_t get_locality_size()
       {

           return localities.size();
       }
    private:
      
       block_cyclic_chunking_policy(std::size_t block_size_,
                                    std::size_t num_chunk_, 
                                    std::vector<hpx::naming::id_type> loc)
         : num_chunk(num_chunk_), block_size(block_size_),localities(loc)
       {} 
       
       block_cyclic_chunking_policy(std::size_t block_size_,
                            std::vector<hpx::naming::id_type> loc)
         : num_chunk(1),block_size(block_size_),localities(loc)
       {}     
       
       block_cyclic_chunking_policy(std::size_t block_size_)
         : num_chunk(1),block_size(block_size_)
       {}
       
       std::size_t big_chunk;
       std::size_t num_chunk;
       std::size_t block_size;
       std::vector<hpx::naming::id_type> localities; 
    };

    static block_cyclic_chunking_policy const block_cyclic;





/*
    struct block_mapper
    {
    
        block_mapper(){}

        block_mapper (std::size_t big_chunk_, 
                          hpx::block_chunking_policy policy_)
        {
            big_chunk = big_chunk_;
            policy    = policy_; 
        }
       
        private:       
 
            std::size_t big_chunk;
            hpx::block_chunking_policy policy;
    };

*/



}

#endif
