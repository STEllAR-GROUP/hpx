
This is a simple test of the Boost.Plugin library. For
now, works only on Linux.

To test:

1. Set BOOST_ROOT to the root of your Boost installation 
   (version 1.32 is required!)
  
2. Build everything with "bjam"

3. Copy built library to ".", under the name "library.so":

   cp liblibrary.so/gcc/debug/shared-linkable-true/liblibrary.so library.so
     
4. Adjust the library path:

   export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
   
5. Run the executable:

   application/gcc/debug/application
   
   
      
  
