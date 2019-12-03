# (Neutron) CT-using-ASTRA-toolbox
Reconstruction of CT from a given set of projection images using projectors provided in ASTRA toolbox.

Tomography reconstruction from a given set of projection images. Includes window-selection, pre-processing, sub-sampling (for exploration of sparse view effects). Three (most cited) algorithms: 1. FBP, 2. SIRT, and 3. TV (FISTA implementation: https://github.com/dmpelt/pytvtomo). Choice of projector types (ASTRA toolbox: https://www.astra-toolbox.com/docs/algs/index.html#) and beam geometry specifications.

1. Palenstijn, W. J., Bédorf, J., Sijbers, J., & Batenburg, K. J. (2017). A distributed ASTRA toolbox. Advanced structural and chemical imaging, 2(1), 19.

2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
