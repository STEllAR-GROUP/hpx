#if !defined(HPX_NiDD89EMe9LJLWXanDp5xpL2VchOqjvdhqDcpmcs)
#define HPX_NiDD89EMe9LJLWXanDp5xpL2VchOqjvdhqDcpmcs

#include <hpx/hpx_fwd.hpp>
//j#include <hpx/runtime/components/component_factory_base.hpp>                     
//#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/include/components.hpp>
//#include <hpx/include/async.hpp>                                                 
#include <hpx/lcos/future.hpp>                                                   
#include <hpx/components/distributing_factory/distributing_factory.hpp>          
//#include <hpx/lcos/future_wait.hpp>                                            
//#include <hpx/lcos/local/packaged_task.hpp>            
#include <hpx/components/dataflow/dataflow.hpp>
                                                                                 
#include <boost/foreach.hpp>                                                     
#include <boost/assert.hpp>                                                      
                                                                                 
#include <utility>                                                               
#include <cstring>                                                               
//#include <vector>

#include "./server/str_search.hpp"
////////////////////////////////////////////////////////////////////////////////
namespace text
{
    struct compare_by_first 
    { 
        template<typename T>
        bool operator()(const T& x, const T& y) const { return x.first < y.first;}
    };

    typedef std::vector<std::pair<std::size_t, hpx::naming::id_type> > 
        card_id_vec_type;

    class HPX_COMPONENT_EXPORT text_split
    {
    public:
        text_split();
        ~text_split();
        void create();
        std::string process(char character, std::string str);
    private:
        std::vector<hpx::naming::id_type> loc_comps_;
        std::string input_string_;
        card_id_vec_type card_id_vec_;

    };
}
#endif //HPX_NiDD89EMe9LJLWXanDp5xpL2VchOqjvdhqDcpmcs
