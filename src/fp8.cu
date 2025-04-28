#include <stdio.h>

#include "cutlass/float8.h"


__global__ void fp8_test() {
  cutlass::float_ue8m0_t sf0(1.0f);
  printf("sf0: %x\n", sf0.raw());
  printf("sf0: %hx\n", sf0.raw());
  printf("sf0: %hu\n", sf0.raw());
  printf("sf0: %hhx\n", sf0.raw());
  printf("sf0: %hhu\n", sf0.raw());
  printf("sf0: %f\n", sf0.convert_to_float(sf0));
  printf("sf0, exp biased: %d\n", sf0.exponent_biased());
  printf("sf0, exp unbiased: %d\n", sf0.exponent());
  printf("sf0, mantissa: %d\n", sf0.mantissa());

  cutlass::float_ue8m0_t sf1(9.5f);
  float accum1 = sf0 * sf1;
  printf("accum1: %f\n", accum1);
  float accum2 = sf0.convert_to_float(sf0) * sf1.convert_to_float(sf1);
  printf("accum2: %f, sf0: %f, sf1: %f\n", accum2, sf0.convert_to_float(sf0), sf1.convert_to_float(sf1));
}


int main() {
  fp8_test<<<1, 1>>>();
  CUTE_CHECK_LAST();
}
