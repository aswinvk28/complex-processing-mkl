#include <mkl.h>
#include <hbwmalloc.h>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <iterator>
#include <complex>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <limits>
#include <cstring>

#define _USE_MATH_DEFINES

using namespace std;

struct FrameParams
{
    short int fft_factor;
};

FrameParams * frameParams = new FrameParams();

void twiddle_internal(vector<MKL_Complex8*>& td)
{
    const double trigconst = M_PI;

    #pragma omp parallel for
    for(int i = 1; i <= frameParams->fft_factor-1; i++) {
        #pragma omp simd
        for(int j = 1; j <= frameParams->fft_factor-1; j++) {
            std::complex<double> t = polar(1.0, -1.0*j*M_PI/(frameParams->fft_factor/2)*i);
            td[j-1][i-1].real = t.real();
            td[j-1][i-1].imag = t.imag();
        }
    }
}

void twiddle_external(const int n, const size_t fft_size, const size_t fft_size1, MKL_Complex8* td)
{
    const double trigconst = -2.0f * M_PI / fft_size;

    #pragma omp simd
    for(size_t i = 0; i < n*fft_size1; i+=n) {
        std::complex<double> t = polar(1.0, i*trigconst);
        td[i/n].real = t.real();
        td[i/n].imag = t.imag();
    }
}

void copy_from_buffer_array_item(MKL_Complex8* buffer, const size_t fft_size1, 
vector<MKL_Complex8*>& td, MKL_Complex8 *data, const int k)
{
    #pragma omp for
    for(int i = 1; i <= frameParams->fft_factor-2; i++) {
        MKL_Complex8* buffer_0 = (MKL_Complex8*) mkl_malloc(sizeof(MKL_Complex8)*fft_size1, 64);
        cblas_ccopy(fft_size1, &buffer[(i+1)*fft_size1], 1, buffer_0, 1);
        cblas_cscal(fft_size1, &td[i][k], buffer_0, 1);
        
        #pragma vector nontemporal
        #pragma omp simd
        for(size_t j = 0; j < fft_size1; j++) {
            data[frameParams->fft_factor*j].real += buffer_0[j].real;
            data[frameParams->fft_factor*j].imag += buffer_0[j].imag;
        }

        mkl_free(buffer_0);
    }
}

void create_handle(DFTI_DESCRIPTOR_HANDLE* handle, const size_t handle_size, const long number_transforms, const long input_distance, const MKL_LONG* stride)
{
    DftiCreateDescriptor(handle, DFTI_SINGLE, DFTI_COMPLEX, 1, handle_size);
    MKL_LONG status;
    if(number_transforms != NULL) {
        status = DftiSetValue(*handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) number_transforms);
        if(status && !DftiErrorClass(status,DFTI_NO_ERROR)) {
            printf("Error: %s\n", DftiErrorMessage(status));
        }
    }
    if(input_distance != 0) {
        status = DftiSetValue(*handle, DFTI_INPUT_DISTANCE, (MKL_LONG) input_distance);
        if(status && !DftiErrorClass(status,DFTI_NO_ERROR)) {
            printf("Error: %s\n", DftiErrorMessage(status));
        }
    }
    if(stride != NULL) {
        status = DftiSetValue(*handle, DFTI_INPUT_STRIDES, stride);
        if(status && !DftiErrorClass(status,DFTI_NO_ERROR)) {
            printf("Error: %s\n", DftiErrorMessage(status));
        }
    }
    DftiCommitDescriptor(*handle);
}

int main(int argc, char * argv[]) {

    

}