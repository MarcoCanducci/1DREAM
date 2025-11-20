#include "LAAT.h"


float floatRand(size_t a) 
{
	if(random_deterministic == 1)
	{
		mt19937 gen(a);
    return UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_FLOAT(gen);
	}
	return UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_FLOAT(GLOBAL_GEN_RAND);
}



// double doubleRand(size_t a, size_t b) 
// {
// 	if(random_deterministic == 1)
// 	{
// 		seed_seq sseq{a,b};
// 		static thread_local mt19937 gen;
// 		gen.seed(sseq);
//     return UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_DOUBLE(gen);
// 	}
// 	return UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_DOUBLE(GLOBAL_GEN_RAND);
// }

size_t sizetRand(size_t a, size_t max_value) 
{
	uniform_int_distribution<size_t> dis(0, max_value-1);

	if(random_deterministic == 1)
	{
		mt19937 gen(a);
    return dis(gen);
	}

	return dis(GLOBAL_GEN_RAND);
}