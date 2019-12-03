# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:12:44 2019

@author: asus
"""

#import gzip
import h5py
#import imageio
#import time
import astra
import numpy as np
import matplotlib.pyplot as plt
import CT_recon_astra_cpu_gpu as CT_recon


''' Load image dataset '''
Image_data = h5py.File('direct_contribution_images_720projections_0p25degree_0_to_180_degrees_py3.h5','r')
Image_data['Projection_data'][:].shape
Image_data['flat_field']
Image_data['dark_field']

Image_data_type = 'mcnp'
P_level_Watts = 500    

Neutron_CT = CT_recon.Neutron_CT

Tomo_object = Neutron_CT(Image_data,
                         P_level_Watts,
                         Image_data_type = 'mcnp')    

Tomo_object.Select_projection_window(slice_first = 200,
                                     slice_last = 220,
                                     slice_width_start = 100,
                                     slice_width_end = 400)

sinogram = Tomo_object.Sinogram_current_corrected(subsamplefactor = 1,
                                                  pad_cor = 6,
                                                  pad_roi = 60,
                                                  sino_slice = 0,
                                                  pad_cor_shift_direction = 'left')

rec_fbp, FBP_cl_h_profile = Tomo_object.Algorithm_FBP(beam_geometry = 'fanflat',
                                                      projector_type = 'cuda', 
                                                      rec_type = 'FBP_CUDA',
                                                      filter_type = 'ram-lak', 
                                                      source_origin_cm = 300,
                                                      detector_origin_cm = 20)

rec_sirt, SIRT_cl_h_profile = Tomo_object.Algorithm_SIRT(beam_geometry = 'fanflat',
                                                         projector_type = 'cuda', 
                                                         rec_type = 'SIRT_CUDA',
                                                         iterations = 200,
                                                         use_minc = 2*np.min(rec_fbp),
                                                         source_origin_cm = 300,
                                                         detector_origin_cm = 20)

rec_tv, TV_cl_h_profile = Tomo_object.Algorithm_TV_regularized(beam_geometry = 'fanflat',
                                                               lam = 1.0e-6,
                                                               lower_bound = 2*np.min(rec_fbp),
                                                               upper_bound = np.inf,
                                                               projector_type = 'cuda', 
                                                               num_inner_iter = 200,
                                                               num_main_iter = 100,
                                                               source_origin_cm = 300,
                                                               detector_origin_cm = 20,
                                                               print_progress = True)

'''
Recon image quality (possible) matric:
    cv2 edge detection
    cv2 segementation 
'''

plt.figure()
plt.plot(FBP_cl_h_profile, 'C1', label = 'FBP')
plt.plot(SIRT_cl_h_profile, 'C2', label = 'SIRT')
plt.plot(TV_cl_h_profile, 'C3', label = 'TV-FISTA')
plt.legend()
plt.show()

astra.projector.clear()
