
#if !defined(HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx)
#define HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/datastructure.hpp"

namespace distributed
{
    ////////////////////////////////////////////////////////////////////////////
    class datastructure
        : public hpx::components::client_base<
            datastructure, distributed::stubs::datastructure>
    {
    private:
        typedef hpx::components::client_base<
            datastructure, distributed::stubs::datastructure> base_type;
    
    public:
        // create new data and initialize in sync
        // can be initialized with gid or w/o gid!!
        datastructure(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
            : base_type(distributed::stubs::datastructure::create_sync(hpx::find_here()))
        {
            data_init(symbolic_name, num_instances, my_cardinality
                , initial_length, initial_value);    
        }
        datastructure(hpx::naming::id_type gid, std::string const& symbolic_name
                , std::size_t num_instances, std::size_t my_cardinality
                , std::size_t initial_length, std::size_t initial_value)
            : base_type(distributed::stubs::datastructure::create_sync(gid))
        {
            data_init(symbolic_name, num_instances, my_cardinality
                , initial_length, initial_value);
        }
        datastructure(hpx::naming::id_type gid)
            : base_type(gid)
        {}
        datastructure()
        {}
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::promise<void>
        data_init_async(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
        {
            return stubs::datastructure::data_init_async(this->gid_,
                symbolic_name, num_instances, my_cardinality, initial_length
                , initial_value);
        }
        
        void data_init(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
        {
            stubs::datastructure::data_init(this->gid_, symbolic_name,
                num_instances, my_cardinality, initial_length, initial_value);
        }
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::promise<void>
        data_write_async(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::vector<std::size_t> client_data)
        {
            return stubs::datastructure::data_write_async(this->gid_,
                symbolic_name, num_instances, my_cardinality, client_data);
        }
        
        void data_write(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::vector<std::size_t> client_data)
        {
            stubs::datastructure::data_write(this->gid_, symbolic_name,
                num_instances, my_cardinality, client_data);
        }
        //////////////////////////////////////////////////////////////////////
    };
}


#endif //HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx
