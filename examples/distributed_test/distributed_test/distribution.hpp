//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv)
#define HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv

#include <hpx/hpx_fwd.hpp>                                                       
#include <hpx/lcos/future.hpp>                                                  
#include <hpx/components/distributing_factory/distributing_factory.hpp>          
                                                                                 
#include <vector>                                                                
                                                                                 
//#include "stubs/partition3d.hpp"                                                 
#include "datastructure.hpp"

namespace distributed
{
    class HPX_COMPONENT_EXPORT distribution
    {
    public:
        distribution();
        ~distribution();
    
        void create(std::string const& symbolic_name_base
            , std::size_t num_iinstances, std::vector<std::size_t> temp_data);
        
        void split_client_data(std::size_t num_instances
            , std::vector<std::size_t> &data_received
            , std::vector<std::vector<std::size_t>> &dd_vector);
        std::size_t get_data_at(std::size_t n);

    private:
        //find GID of object containing specified cardinality/data
        //TO DO
        typedef hpx::components::distributing_factory distributing_factory;
        typedef distributing_factory::async_create_result_type 
            async_create_relsult_type;
    private:
        std::vector<hpx::naming::id_type> comp_instances_;
        bool comp_created_;
        datastructure data_struct_;
        std::size_t max_comp_size_;
    };
}

#endif //HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv
