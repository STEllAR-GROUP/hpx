#include &lt;hpx/hpx_init.hpp&gt;
#include &lt;hpx/iostream.hpp&gt;
#include &lt;hpx/future.hpp&gt;

int hpx_main()
{
    // Print a simple message using HPX
    hpx::cout &lt;&lt; "Hello from HPX! HPX is working correctly." &lt;&lt; hpx::endl;
    
    // Test basic future functionality
    hpx::future&lt;int&gt; f = hpx::async([]() { return 42; });
    hpx::cout &lt;&lt; "The answer is: " &lt;&lt; f.get() &lt;&lt; hpx::endl;
    
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
