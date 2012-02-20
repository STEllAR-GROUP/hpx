

#if !defined(HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv)
#define HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv

#include <hpx/hpx_fwd.hpp>                                                       
#include <hpx/lcos/promise.hpp>                                                  
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
            , std::size_t num_instances, std::size_t my_cardinality
            , std::size_t initial_length, std::size_t initial_value);

    private:
        //find GID of object containing specified cardinality/data
        //TO DO
        typedef hpx::components::distributing_factory distributing_factory;
        typedef distributing_factory::async_create_result_type 
            async_create_relsult_type;
    private:
        std::vector<hpx::naming::id_type> localities_;
        bool comp_created_;
        datastructure data_struct_;
    };
}

#endif //HPX_AJGKWzDvv6WY3TBEXyrCoEBHXyNsAGMOd7kMxopv
