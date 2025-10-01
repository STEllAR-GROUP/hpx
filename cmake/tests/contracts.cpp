#ifdef HPX_HAVE_CONTRACTS
    #if __cplusplus >= 202602L
        #define HPX_PRE(x) pre((x)) //notes:Moving HPX_PRE() from body to function definition neccessary
        #define HPX_CONTRACT_ASSERT(x) contract_assert((x))  
        #define HPX_POST(x) post((x))


        #ifdef HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS 
            #define HPX_ASSERT(x) HPX_CONTRACT_ASSERT((x))
        #endif


    #else
        #pragma message("Warning: Contracts require C++26 or later. Falling back to HPX_ASSERT.")        
        #define HPX_PRE(x) HPX_ASSERT((x))
        #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))  
        #define HPX_POST(x) HPX_ASSERT((x))
    #endif
#else
    #define HPX_PRE(x)
    #define HPX_CONTRACT_ASSERT(x)
    #define HPX_POST(x)
#endif
