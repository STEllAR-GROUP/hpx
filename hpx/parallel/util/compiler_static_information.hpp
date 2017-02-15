
#if !defined(HPX_COMPILER_STATIC_INFORMATION)
#define HPX_COMPILER_STATIC_INFORMATION

#include <vector>
#include <initializer_list>
#include <iterator>
#include <assert.h>

namespace hpx { namespace parallel {

		template <typename T>
		class features_container {

			// The exctracted static information from clang about the current loop include:
			// f0 = number of threads
			// f1 = number of total operations
			// f2 = number of float operations
			// f3 = number of comparison operations
			// f4 = number of iterations
			// f5 = deepest loop level

			// Initially clang assignes the value of f0 and f4 to be 0,
    		// which are the number of threads and number of iterations.
    		// These values will be determined at runtime with for_each.
			std::vector<T> features;

			//cost function : w0 + w1 * f1 + w2 * f2 + ....
			bool execution_policy_cost_fnc(std::vector<T>& modified_features, std::vector<T>& weights) {
				assert((weights.size() > 0 && modified_features.size() > 0) &&
            		"Requires at least one weight and one feature.");

				T result = weights[0];

				typename std::vector<T>::iterator w = weights.begin() + 1;
				typename std::vector<T>::iterator f = modified_features.begin();

				while(f != modified_features.end()) {
					result += *w * *f;
					++w;
					++f;
				}
				return result > 0;
			}

		public:
			features_container(std::initializer_list<T> l) {
				features.resize(l.size());

				// assigning values for features
				typename std::initializer_list<T>::iterator it = l.begin();
				typename std::vector<T>::iterator f = features.begin();

				while(it != l.end()) {
					*f = *it;
					++it;
					++f;
				}
			}

			//getting values of features of the current loop
			std::vector<T>& get_features() {
				return features;
			}

			//determining the best execution policy for the current loop
			bool execution_policy_determination(std::vector<T>& modified_features, std::vector<T>& weights) {
				return execution_policy_cost_fnc(modified_features, weights);
			}
		};

		// This class contains weights from learning network
		template <typename T>
		class weights_container {

			std::vector<T> weights;

		public:
			weights_container(std::initializer_list<T> l) {
				weights.resize(l.size());

				//assigning values for weights
				typename std::initializer_list<T>::iterator it = l.begin();
				typename std::vector<T>::iterator w = weights.begin();
				
				while(it != l.end()) {
					*w = *it;
					++it;
					++w;
				}
			}

			//getting values of computed weights of the learning network
			std::vector<T>& get_weights() {
				return weights;
			}
		};
	}
}

#endif
