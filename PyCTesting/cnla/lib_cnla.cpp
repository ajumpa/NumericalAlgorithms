#include <iostream>

extern "C" {
  void QR(double *a, size_t h, size_t w)
  {

    //std::cout << "a2_h & a2_w" << a2_h << a2_w << std::endl; 
    for (size_t i = 0; i < h; i++) {
      for (size_t j = 0; j < w; j++) {
        if (i == j) a[i * h + j] = i * j;
        printf("%f ", a[i * h + j]);
      }
      printf("\n");
    }
    printf("\n");
  }
}
