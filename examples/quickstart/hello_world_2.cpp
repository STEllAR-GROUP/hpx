#include <hpx/config.hpp>
#include <hpx/iostream.hpp>
#include <contracts> 


#ifdef HPX_HAVE_CONTRACTS
    #define HPX_PRE(x) pre((x)) 
    #define HPX_CONTRACT_ASSERT(x) contract_assert((x))  
    #define HPX_POST(x) post((x))
#endif


void handle_contract_violation(const std::contracts::contract_violation& violation)
{
    if (violation.semantic() == std::contracts::evaluation_semantic::observe)
    {
        std::cerr << "Successfully overridden violation handler for observe mode\n";
        std::cerr << violation.location().function_name() << ":"
                  << violation.location().line()
                  << ": observing violation: "
                  << violation.comment()
                  << "\n";
        return; // don’t abort, just report
    }
    
    // fallback to default for enforce/quick_enforce
    invoke_default_contract_violation_handler(violation);
}

int f(const int x) 
HPX_PRE(false)
HPX_POST(true)
{
    HPX_CONTRACT_ASSERT(false);
    
    return x;
}


int main() {
  auto a = f(0);
  std::cout << "Should be shown only in observe mode" << std::endl;
}



// if(HPX_WITH_CONTRACTS)

//   add_executable(contract_violation_test_observe contract_violation_test.cpp)
//   target_compile_options(contract_violation_test_observe PRIVATE
//       -std=c++23 -fcontracts -stdlib=libc++
//       -fcontract-evaluation-semantic=observe
//   )
//   add_test(NAME contract_violation_test_observe
//       COMMAND $<TARGET_FILE:contract_violation_test_observe>)
//   set_tests_properties(contract_violation_test_observe PROPERTIES
//       PASS_REGULAR_EXPRESSION "Successfully overridden violation handler for observe mode")

//   add_executable(contract_violation_test_enforce contract_violation_test.cpp)
//   target_compile_options(contract_violation_test_enforce PRIVATE
//       -std=c++23 -fcontracts -stdlib=libc++
//       -fcontract-evaluation-semantic=enforce
//   )
//   add_test(NAME contract_violation_test_enforce
//       COMMAND $<TARGET_FILE:contract_violation_test_enforce>)
//   set_tests_properties(contract_violation_test_enforce PROPERTIES WILL_FAIL TRUE)

//   add_executable(contract_violation_test_quick_enforce contract_violation_test.cpp)
//   target_compile_options(contract_violation_test_quick_enforce PRIVATE
//       -std=c++23 -fcontracts -stdlib=libc++
//       -fcontract-evaluation-semantic=quick_enforce
//   )
//   add_test(NAME contract_violation_test_quick_enforce
//       COMMAND $<TARGET_FILE:contract_violation_test_quick_enforce>)
//   set_tests_properties(contract_violation_test_quick_enforce PROPERTIES WILL_FAIL TRUE)

//   add_executable(contract_violation_test_ignore contract_violation_test.cpp)
//   target_compile_options(contract_violation_test_ignore PRIVATE
//       -std=c++23 -fcontracts -stdlib=libc++
//       -fcontract-evaluation-semantic=ignore
//   )
//   add_test(NAME contract_violation_test_ignore
//       COMMAND $<TARGET_FILE:contract_violation_test_ignore>)

// endif()
