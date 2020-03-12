## Fast Fourier Transform using Intel MKL

Create DTFT handle using:

```c++

DftiCreateDescriptor(handle, DFTI_SINGLE, DFTI_COMPLEX, 1, handle_size);

MKL_LONG status;

```

Set parameters using:

```c++

// n_fft
DftiSetValue(*handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) number_transforms);

// input distance
DftiSetValue(*handle, DFTI_INPUT_DISTANCE, (MKL_LONG) input_distance);

// stride
DftiSetValue(*handle, DFTI_INPUT_STRIDES, stride);

// commit descriptor handle
DftiCommitDescriptor(*handle);

```

## Data types available

```c++

MKL_Complex8 *buffer;

MKL_Complex16 *buffer;

```

