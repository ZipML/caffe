
### Low-precision Model

Currenlty, only stochastic rounding is implemented.

ZIPML:

    QUANTIZE=1 NLEVELS=5 OPTIMAL=1 ./build/tools/caffe train -gpu 0 -solver examples/cifar10/cifar10_quick_solver.prototxt

XONR-BINARY:

    QUANTIZE=1 NLEVELS=2 ./build/tools/caffe train -gpu 0 -solver examples/cifar10/cifar10_quick_solver.prototxt

UNIFORM 5 LEVELS (XONR-BINARY):

    QUANTIZE=1 NLEVELS=5 ./build/tools/caffe train -gpu 0 -solver examples/cifar10/cifar10_quick_solver.prototxt

ORIGINAL CAFFE:

    ./build/tools/caffe train -gpu 0 -solver examples/cifar10/cifar10_quick_solver.prototxt

### Expected Result for Determinstic Rounding

    ORIGINAL CAFFE > ZIPML 5 LEVELS > XONR-BINARY > UNIFORM 5 LEVELS (XONR-BINARY)