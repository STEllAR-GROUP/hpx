#include <hpx/lcos/future.hpp>
#include <thread>

int main(){
  hpx::future<int> fA = hpx::async([])(){};{
    func(valA);
  };
  hpx::future<string> fB = hpx::async(hpx::launch::async, [](){
     func(valB);
  });

  fA.wait();
  fB.wait();

  hpx::cout << fA.get() << `\n`;
  hpx::cout << fB.get() << `\n`;
}
