#ifndef CAFFE_QUANTIZE_H_
#define CAFFE_QUANTIZE_H_

#include <stdio.h>
#include <assert.h>

namespace caffe {


template <typename Dtype>
__global__ void _l1(const Dtype *a, int n, Dtype *scratch){
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < 128){
    scratch[id] = 0;
    for(int i=id;i<n;i+=128){
      scratch[id] += fabs(a[i]);
    }
  }
}

template <typename Dtype>
__global__ void _max_elem(const Dtype *a, int n, Dtype *scratch){
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < 128){
    scratch[id] = -10000000.0;
    for(int i=id;i<n;i+=128){
      if(a[i] > scratch[id]) scratch[id] = a[i];
    }
  }
}

template <typename Dtype>
__global__ void _min_elem(const Dtype *a, int n, Dtype *scratch){
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < 128){
    scratch[id] = 10000000.0;
    for(int i=id;i<n;i+=128){
      if(a[i] < scratch[id]) scratch[id] = a[i];
    }
  }
}

template <typename Dtype>
__global__ void _sum(Dtype *a, int n, Dtype * result){
  *result = 0.0;
  for(int i=0;i<n;i++){
    *result += a[i];
  }
  //printf("%f\n", *result);
}

template <typename Dtype>
__global__ void _max(Dtype *a, int n, Dtype * result){
  *result = -10000000.0;
  for(int i=0;i<n;i++){
    if(a[i] > *result) *result = a[i];
  }
  //printf("%f\n", *result);
}

template <typename Dtype>
__global__ void _min(Dtype *a, int n, Dtype * result){
  *result = 10000000.0;
  for(int i=0;i<n;i++){
    if(a[i] < *result) *result = a[i];
  }
  //printf("%f\n", *result);
}

template <typename Dtype>
__global__ void _zipml_search_elem_5level_heuristics(const Dtype *a, int n, Dtype * qmin, Dtype * qmax, Dtype *scratch){ //40kernels
  // Search with the following heuristic to get 5 levels:
  // -3k, -k, 0, k, 3k
  //    TODO: this is not the fastest version (the join order is wrong).
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  const int NMARKERS = 5;
  if(id<40){
    Dtype largest = fabs(*qmax) > fabs(*qmin) ? fabs(*qmax) : fabs(*qmin);
    Dtype markers[5];
    int intk = 0;
    for(Dtype k=0.1; k<0.4; k+= 0.1){ // for each marker setting, calculate the quantization error
      // set marker
      markers[0] = -3*k; markers[1] = -k;
      markers[2] = 0;
      markers[3] = k; markers[4] = 3*k;

      Dtype diff = 0.0;      
      for(int i=id;i<n;i+=40){ // mysterous 40 -- just when start this kernel,  use <<<40,1>>> grid.
        Dtype data = a[i] / largest;
        if(data < markers[0]){
          diff += (data - markers[0]) * (data - markers[0]);
        }else if(data >= markers[NMARKERS - 1]){
          diff += (data - markers[NMARKERS - 1]) * (data - markers[NMARKERS - 1]);
        }else{
          for(int j=1;j<NMARKERS;j++){
            if(markers[j-1] <= data && data < markers[j]){
              if(data - markers[j-1] < markers[j] - data){
                diff += (data - markers[j-1]) * (data - markers[j-1]);
              }else{
                diff += (data - markers[j]) * (data - markers[j]);
              }
            }
          }
        }
      }
      scratch[id*3 + intk] = diff;
      intk ++;
    }
  }
}

template <typename Dtype>
__global__ void _zipml_search_5level_heuristics(Dtype *a, int n, Dtype * result){
    Dtype kscores[3]; // each k has a score
    kscores[0] = 0.0; kscores[1] = 0.0; kscores[2] = 0.0;
    for(int intk=0;intk<3;intk++){
      for(int id=0;id<40;id++){
        kscores[intk] += a[id*3 + intk];
      }
    }

    // pick the lowest score
    if(kscores[0] <= kscores[1] && kscores[0] <= kscores[2]) *result = 0.1;
    if(kscores[1] <= kscores[0] && kscores[1] <= kscores[2]) *result = 0.2;
    if(kscores[2] <= kscores[0] && kscores[2] <= kscores[1]) *result = 0.3;

    //printf("%f    %f     %f\n", kscores[0], kscores[1], kscores[2]);
}


template <typename Dtype>
void l1(const Dtype * a, int n, Dtype * scratch, Dtype * result){
  _l1<Dtype><<<128, 1>>>(a, n, scratch);
  _sum<Dtype><<<1,1>>>(scratch, 128, result);
}

template <typename Dtype>
void max(const Dtype * a, int n, Dtype * scratch, Dtype * result){
  _max_elem<Dtype><<<128, 1>>>(a, n, scratch);
  _max<Dtype><<<1,1>>>(scratch, 128, result);
}

template <typename Dtype>
void min(const Dtype * a, int n, Dtype * scratch, Dtype * result){
  _min_elem<Dtype><<<128, 1>>>(a, n, scratch);
  _min<Dtype><<<1,1>>>(scratch, 128, result);
}

template <typename Dtype>
void zipml_search_5level_heuristics(const Dtype * a, int n, Dtype * qmin, Dtype * qmax, Dtype * scratch, Dtype * result){
  _zipml_search_elem_5level_heuristics<Dtype><<<40, 1>>>(a, n, qmin, qmax, scratch);
  _zipml_search_5level_heuristics<Dtype><<<1,1>>>(scratch, 128, result);
}

template <typename Dtype>
__global__ void zipml(const Dtype * src, Dtype * dst, int n, Dtype * k, Dtype * qmin, Dtype * qmax, bool verbose){ // src is original, dst is quantized
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < n){
    Dtype largest = fabs(*qmax) > fabs(*qmin) ? fabs(*qmax) : fabs(*qmin);

    const int NMARKERS = 5;
    Dtype markers[5];
    markers[0] = -3* *k; markers[1] = - *k;
    markers[2] = 0;
    markers[3] = *k; markers[4] = 3* *k;
    
    Dtype data = src[id] / largest;
    Dtype newval;

    if(data < markers[0]){
      newval = markers[0];
    }else if(data >= markers[NMARKERS - 1]){
      newval = markers[NMARKERS - 1];
    }else{
      for(int j=1;j<NMARKERS;j++){
        if(markers[j-1] <= data && data < markers[j]){
          if(data - markers[j-1] < markers[j] - data){
            newval = markers[j-1];
          }else{
            newval = markers[j];
          }
        }
      }
    }
    dst[id] = newval * largest;
  }

  if(verbose){
    if(id == 0) printf("ZIPML 5LEVELS %f -> %f \n", src[id], dst[id]);
    if(id == 10) printf("ZIPML 5LEVELS %f -> %f \n", src[id], dst[id]);
  }

  // // Sanity check k function
  // printf("k = %f\n", *k);

}


template <typename Dtype>
__global__ void bnn(const Dtype * src, Dtype * dst, int n, Dtype * l1, bool verbose){ // src is original, dst is quantized
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < n){
    dst[id] = src[id] >=0 ? *l1/n : - *l1/n;
  }

  if(verbose){
    if(id == 0) printf("BNN %f -> %f \n", src[id], dst[id]);
    if(id == 10) printf("BNN %f -> %f \n", src[id], dst[id]);
  }

  // Sanity check l1 function
  //Dtype real_l1 = 0.0;
  //for(int i=0;i<n;i++){
  //  real_l1 += fabs(src[i]);
  //}
  //printf("%f vs. %f\n", real_l1, *l1);
}

template <typename Dtype>
__global__ void bnn_multibits(const Dtype * src, Dtype * dst, int n, int nlevels, Dtype * qmin, Dtype * qmax, bool verbose){ // src is original, dst is quantized
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  // Follows XNOR-Net paper: k-bit Quantization
  if(id < n){
    Dtype largest = fabs(*qmax) > fabs(*qmin) ? fabs(*qmax) : fabs(*qmin);
    Dtype normalized = src[id] / largest;
    normalized = (normalized+1)/2 * (nlevels - 1);
    Dtype upper = ceil(normalized);
    Dtype lower = floor(normalized);
    Dtype rounded;
    if(normalized - lower <= upper - normalized){
      rounded = lower;
    }else{
      rounded = upper;
    }
    Dtype newval = 2 * (1.0 * rounded / (nlevels - 1) - 0.5);
    dst[id] = newval * largest;
  }

  if(verbose){
    if(id == 0) printf("UNIFORM-%d LEVEL %f -> %f \n", nlevels, src[id], dst[id]);
    if(id == 10) printf("UNIFORM-%d LEVEL %f -> %f \n", nlevels, src[id], dst[id]);
  }

  /**
  // THIS DOES NOT WORK WELL
  if(id < n){
    Dtype normalized = (src[id] - *qmin) / (*qmax - *qmin);
    Dtype upper = ceil(normalized * (nlevels - 1));
    Dtype lower = floor(normalized * (nlevels - 1));
    Dtype rounded;
    if(normalized - lower < upper - normalized){
      rounded = lower;
    }else{
      rounded = upper;
    }
    Dtype newval = 1.0 * rounded / (nlevels - 1);
    dst[id] = newval * (*qmax - *qmin) + *qmin;
  }

  if(verbose){
    if(id == 0) printf("UNIFORM-%d LEVEL %f -> %f \n", nlevels, src[id], dst[id]);
    if(id == 10) printf("UNIFORM-%d LEVEL %f -> %f \n", nlevels, src[id], dst[id]);
  }
  */

  // // Sanity check max function
  //Dtype real_max = -1000000;
  //Dtype real_min = 10000000;
  //for(int i=0;i<n;i++){
  //  if(src[i] > real_max) real_max = src[i];
  //  if(src[i] < real_min) real_min = src[i];
  //}
  //if(id == 0){
  //  printf("max %f vs. %f\n", real_max, *qmax);
  //  printf("min %f vs. %f\n", real_min, *qmin);
  //}
}


template <typename Dtype>
void quantize(const Dtype * src, Dtype * dst, int n, int nlevels, bool is_optimal, int usage_counter, Dtype * scratch,  Dtype * ql1,  Dtype * qmax,  Dtype * qmin, Dtype * qzipml){

  bool verbose = usage_counter % 100 == 0;

  // every 100 iter reminds us what you are doing.
  if(verbose){
    printf("QUANTIZE NLEVELS %d  ISOPTIMAL %d\n", nlevels, is_optimal);
  }

  // first, copy original weight to quantized version
  cudaMemcpy(dst, src, n*sizeof(Dtype), cudaMemcpyDeviceToDevice);
  
  if(nlevels == 2){
    // BNN
    // First, get l1
    if(verbose) printf("BNN!\n");
    l1<Dtype>(src, n, scratch, ql1);
    bnn<Dtype><<<128,n/128>>>(src, dst, n, ql1, verbose);
  }

  if(nlevels > 2 && is_optimal == false){
    // UNIFORM
    // First, get max and min
    if(verbose) printf("UNIFORM %d LEVELS!\n", nlevels);
    max<Dtype>(src, n, scratch, qmax);
    min<Dtype>(src, n, scratch, qmin);
    bnn_multibits<Dtype><<<128,n/128>>>(src, dst, n, nlevels, qmin, qmax, verbose);
  }

  if(is_optimal == true){
    if(nlevels == 5){
    // ZIPML
    // First, search grid with heuristics
    if(verbose) printf("ZIP %d LEVELS!\n", nlevels);
    max<Dtype>(src, n, scratch, qmax);
    min<Dtype>(src, n, scratch, qmin);
    zipml_search_5level_heuristics<Dtype>(src, n, qmin, qmax, scratch, qzipml);
    zipml<Dtype><<<128,n/128>>>(src, dst, n, qzipml, qmin, qmax, verbose);
    }else{
      printf("ONLY SUPPORT LEVEL 5 OPTIMAL HEURISTICS NOW!");
      assert(false);
    }

  }

}



template <typename Dtype>
__global__ void cleanup_bnn_gradient(const Dtype * weight, Dtype * gradient, int n){ 
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id < n){
    if(weight[id] < -1 || weight[id] > 1){
      gradient[id] = 0;
    }
  }

}


template <typename Dtype>
void postprocess_gradient(const Dtype * weight, Dtype * gradient, int n, int nlevels, bool is_optimal, int usage_counter){

  if(nlevels == 2){
    // BNN
    cleanup_bnn_gradient<Dtype><<<128,n/128>>>(weight, gradient, n);
  }



}




}  // namespace caffe

#endif  // CAFFE_QUANTIZE_H_


























