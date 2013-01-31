
#if !defined(HPX_YKKs9J9MWOPUNwti709taq6QWsgc6rxQZZZzY5Sw)
#define HPX_YKKs9J9MWOPUNwti709taq6QWsgc6rxQZZZzY5Sw

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <vector>

#include "./server/tictactoe.hpp"
 
////////////////////////////////////////////////////////////////////////////////
namespace game 
{
    class HPX_COMPONENT_EXPORT dist_factory
    {
    public:
        dist_factory();
        ~dist_factory();
        /// toss whether player one gets x or player 2
        char create();
    private:
        std::vector<hpx::naming::id_type> loc_comps_;
         
    };
}

#endif ///HPX_YKKs9J9MWOPUNwti709taq6QWsgc6rxQZZZzY5Sw
