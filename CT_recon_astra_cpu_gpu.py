# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:09:30 2019

@author: asus
"""

#import gzip
#import h5py
#import imageio
import time
import astra
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tvtomo
#import tomopy

astra.astra.get_gpu_info(0)
astra.astra.set_gpu_index(0)


class Neutron_CT(object):
    
    def __init__(self, 
                 Image_data,
                 P_level_Watts,
                 Image_data_type = 'mcnp',
                 mu_bar = 2.53 ,
                 k_eff = 0.9876,
                 Q_value_MeV = 181, 
                 detector_length_cm = 20, 
                 exposure_time_sec = 300):
        
        '''
        Image data:
            [Projection_data, flat_field, dark_field] in h5
            When mcnp images, flat_field = None, dark_field = None
        '''
        self.Image_data = Image_data 
        self.Image_data_type = Image_data_type # 'mcnp' or 'experimental'
        self.P_level = P_level_Watts * 6.242e12 # [MeV / s]
        self.mu_bar = mu_bar
        self.k_eff = k_eff
        self.Q_value = Q_value_MeV
        self.detector_length = detector_length_cm
        self.pixel_area = (self.detector_length / 512) ** 2 # 512 x 512 F5 point detectors over 20cm x 20cm area
        self.pixel_area_experimental = (self.detector_length / 2048) ** 2 # Actual images collected experimentally
        
        '''
        To determine ~ number of neutron hitting pixel area of the detector:
            The area of F5 point detectors = 16 * experimental_pixel_area
            So dividing exposure time by 16 ensures that correct noise behavior is included
        
        The noise behavior is only for neutron counts - conversion to photon process not modeled
        '''
        self.area_exp_time_correction = self.pixel_area / self.pixel_area_experimental
        self.exp_time = exposure_time_sec / self.area_exp_time_correction 
        if self.Image_data_type == 'mcnp':
            self.flux_multiplier = ((self.P_level * self.mu_bar) / (self.Q_value * self.k_eff)) * (self.pixel_area * self.exp_time)
        elif self.Image_data_type == 'exprimental':
            self.flux_multiplier = 1.
        
        
        
    
    def Preprocessing(self):
        '''
        As the object is the one rotating in our setup, no need to take flat/dark
        for every projection and construct their corresponding singoram for correction
        
        Instead, apply the same flat_field and dark_field correction to all projection,
        Then use the corrected projection images for constructing the singoram for each slice
        '''
        
        Projection_data = self.Image_data['Projection_data'][:]
        flat_field = self.Image_data['flat_field'][:]
        dark_field = self.Image_data['dark_field'][:]
        
        # Plor samples before correction
        tot_num_proj = Projection_data.shape[0] 
        plt.figure(figsize=(20, 20))
        idx = np.arange(0, tot_num_proj, 1)
        np.random.shuffle(idx)
        rand_idx = np.sort(idx[:15])
        n_rows = 3
        n_columns = 5
        for j in range(15):
            plt.subplot(n_rows, n_columns, j+1)
            plt.title('Before bf_df corr: projection {0} / {1}'.format(str(rand_idx[j]), tot_num_proj))
            plt.imshow(Projection_data[rand_idx[j]], 
                       aspect = 'auto',
                       interpolation="nearest", 
                       cmap="gray")
            plt.xlabel('Detector pixel')
            plt.ylabel('Detector pixel')
            plt.colorbar()
        
        ''' Check Formula: proj_corr = (bf - proj) / (bf - df) '''
        Projection_data *= -1
        Projection_data += flat_field
        Projection_data /= np.subtract(flat_field, dark_field)
        self.Projection_data_field_corrected = Projection_data
        
        # Plot samples after correction 
        plt.figure(figsize=(20, 20))
        for j in range(15):
            plt.subplot(n_rows, n_columns, j+1)
            plt.title('After bf_df corr: projection {0} / {1}'.format(str(rand_idx[j]), tot_num_proj))
            plt.imshow(self.Projection_data_field_corrected[rand_idx[j]], 
                       aspect = 'auto',
                       interpolation="nearest", 
                       cmap="gray")
            plt.xlabel('Detector pixel')
            plt.ylabel('Detector pixel')
            plt.colorbar()
        
        return self.Projection_data_field_corrected
    
    
    def Select_projection_window(self,
                                slice_first = 200,
                                slice_last = 220,
                                slice_width_start = 150,
                                slice_width_end = 350):
        
        self.slice_first = slice_first
        self.slice_last = slice_last
        self.slice_width_start = slice_width_start
        self.slice_width_end = slice_width_end
        
        
    def Sinogram_generator(self):
        
        # Which rows of the projection images to use to generate the sinogram
        rows = np.arange(self.slice_first, self.slice_last)
        # Vertical lines to use as boundary of region of interest in proj imgs
        row_width = self.slice_width_end - self.slice_width_start
        
        if self.Image_data_type == 'experimental':
            # Collect the projection imgs with bf and df correction applied
            projection_imgs = self.Preprocessing 
        
        elif self.Image_data_type == 'mcnp':
            # Collect the projection imgs with flux_multiplier applied
            projection_imgs = self.Image_data['Projection_data'][:]
            projection_imgs *= self.flux_multiplier
            
            # Plot samples before including noise based on poisson distribution of neutron counts
            plt.figure(figsize=(20, 20))
            idx = np.arange(0, projection_imgs.shape[0], 1)
            np.random.shuffle(idx)
            rand_idx = np.sort(idx[:15])
            n_rows = 3
            n_columns = 5
            for j in range(15):
                plt.subplot(n_rows, n_columns, j+1)
                plt.title('Projection: ' + str(rand_idx[j]))
                plt.imshow(projection_imgs[rand_idx[j]], 
                           aspect = 'auto',
                           interpolation="nearest", 
                           cmap="gray")
                plt.xlabel('Detector pixel')
                plt.ylabel('Detector pixel')
                plt.colorbar()
            
            # Include noise behavior - replace projection images by noised version
            for i in range(projection_imgs.shape[0]):
                projection_imgs[i] = np.random.poisson(projection_imgs[i])
                
            # Plot samples after including noise based on poisson distribution of neutron counts
            plt.figure(figsize=(20, 20))
            idx = np.arange(0, projection_imgs.shape[0], 1)
            np.random.shuffle(idx)
            rand_idx = np.sort(idx[:15])
            n_rows = 3
            n_columns = 5
            for j in range(15):
                plt.subplot(n_rows, n_columns, j+1)
                plt.title('Projection: ' + str(rand_idx[j]))
                plt.imshow(projection_imgs[rand_idx[j]], 
                           aspect = 'auto',
                           interpolation = "nearest", 
                           cmap = "gray")
                plt.xlabel('Detector pixel')
                plt.ylabel('Detector pixel')
                plt.colorbar()
        
        num_projections = len(projection_imgs)
        self.sinograms_all = np.zeros((num_projections, len(rows), row_width))
            
        for i in range(len(rows)):
            sinogram_single = np.zeros((row_width, num_projections))
            for j in np.arange(0, sinogram_single.shape[1]):
                pixel_line = projection_imgs[j][rows[i], self.slice_width_start:self.slice_width_end]
                sinogram_single[:, j] = pixel_line
            self.sinograms_all[:, i, :] = np.rot90(sinogram_single)
        
        
        # Plot sample sinograms - 15 random slices 
        plt.figure(figsize=(20, 20))
        idx = np.arange(0, self.slice_last - self.slice_first, 1)
        np.random.shuffle(idx)
        rand_idx = np.sort(idx[:15])
        n_rows = 3
        n_columns = 5
        for j in range(15):
            plt.subplot(n_rows, n_columns, j+1)
            plt.title('Sinogram: slice ' + str(rand_idx[j]))
            plt.imshow(self.sinograms_all[:, rand_idx[j], :], 
                       extent=[0, row_width, np.pi, 0], 
                       aspect = 'auto',
                       interpolation = "nearest", 
                       cmap = "gray")
            plt.xlabel('Detector pixel')
            plt.ylabel('Projection angle (radians)')
            plt.colorbar()
        
        return self.sinograms_all
    
    
    def Sinogram_subsampled(self, sino_slice = 0,
                             subsamplefactor = 1):
        
        '''
        Subsampling factor:
            If 10: every 10th projection is kept
            So final number of projection = tot_proj_original / 10 '''
        
        # First, run Sinogram_generator to collect all sinograms with bf and df correction
        sinograms_all = self.Sinogram_generator()
        
        # Then, select one slice of sinogram to work with (Scope of project == 2D tomography)
        sinogram_current = sinograms_all[:, sino_slice, :]
        self.sino_subsamplefactor = subsamplefactor
        self.sinogram_current = sinogram_current[::subsamplefactor, ]
        
        return self.sinogram_current
    
    
    def Sinogram_current_corrected(self,
                                   subsamplefactor,
                                   pad_cor,
                                   pad_roi,
                                   sino_slice = 0,
                                   pad_cor_shift_direction = 'left'):
        
        '''    
        Two main correction necessary after generating the sinogram:
            1. Center of rotation: pad_cor (pad one side)
            2. Region of interest: pad_roi (pad both sides)
            
        Note when current singoram is selected: its shape = (num_anlges, original_num_detectors)
            This sets:
                1. The number of angles
                2. The reconstruction area dimension = original_num_detectors
        '''
        
        # Extract current sinogram of a given selection of slice
        # This includes already (If using experimental images)
        #   1. bf and df correction because Preprocessing is called (through Sinogram_generator)
        #   2. subsampling factor since Sinogram_subsampled is called
        
        self.sinogram_current = self.Sinogram_subsampled(sino_slice,
                                                         subsamplefactor)
        
        # Use sinogram shape to set the number of angles and reconstruction dimension 
        self.num_angles, self.num_detectors_orig = self.sinogram_current.shape
        self.angles = np.linspace(0, np.pi, self.num_angles)


        # COR correction
        num_angles = self.num_angles
        N = self.num_detectors_orig
        self.sino_pad_cor = np.zeros((num_angles, N + pad_cor))
        if pad_cor_shift_direction == 'left':
            self.sino_pad_cor[:, :N] = self.sinogram_current
            self.sino_pad_cor[:, N:] = np.tile(self.sinogram_current[:, -1].reshape(num_angles, 1), (1,pad_cor))
        elif pad_cor_shift_direction == 'right':
            self.sino_pad_cor[:, :pad_cor] = np.tile(self.sinogram_current[:, 0].reshape(num_angles, 1), (1,pad_cor))
            self.sino_pad_cor[:, pad_cor:] = self.sinogram_current
        
        # ROI correction
        
        self.sino_pad_roi = np.zeros((self.num_angles, N + pad_cor + 2 * pad_roi))
        self.sino_pad_roi[:, pad_roi:N+pad_cor+pad_roi] = self.sino_pad_cor
        self.sino_pad_roi[:, :pad_roi] = np.tile(self.sino_pad_cor[:, 0].reshape(num_angles, 1), (1, pad_roi))
        self.sino_pad_roi[:, N+pad_cor+pad_roi:] = np.tile(self.sino_pad_cor[:, -1].reshape(num_angles, 1), (1, pad_roi))

        # Plot singrams (all in one): 
        # 1. Original
        # 2. After COR
        # 3. After ROI corrections
        plt.figure(figsize=(10, 5))
        n_rows = 1
        n_columns = 3
        
        row_width = self.slice_width_end - self.slice_width_start
        plt.subplot(n_rows, n_columns, 1)
        plt.title('Sinogram: original')
        plt.imshow(self.sinogram_current,
                   extent=[0, row_width, np.pi, 0],
                   aspect = 'auto',
                   interpolation = "nearest",
                   cmap="gray")
        plt.xlabel('Detector pixel')
        plt.ylabel('Projection angle (radians)')
        plt.colorbar()
        
        plt.subplot(n_rows, n_columns, 2)
        plt.title('Sinogram: after COR correction')
        plt.imshow(self.sino_pad_cor,
                   extent=[0, row_width, np.pi, 0],
                   aspect = 'auto',
                   interpolation = "nearest",
                   cmap="gray")
        plt.xlabel('Detector pixel')
        plt.ylabel('Projection angle (radians)')
        plt.colorbar()
        
        plt.subplot(n_rows, n_columns, 3)
        plt.title('Sinogram: after ROI correction')
        plt.imshow(self.sino_pad_roi,
                   extent=[0, row_width, np.pi, 0],
                   aspect = 'auto',
                   interpolation = "nearest",
                   cmap="gray")
        plt.xlabel('Detector pixel')
        plt.ylabel('Projection angle (radians)')
        plt.colorbar()
        
        return self.sinogram_current


    def create_projector_id(self, 
                            sinogram,
                            beam_geometry,
                            projector_type = 'cuda', 
                            recon_dimension = None,
                            source_origin_cm = None,
                            detector_origin_cm = None):
        
        '''
        ASTRA projection geometry entry (https://www.astra-toolbox.com/apiref/creators.html):
            astra.create_proj_geom('parallel', detector_spacing, det_count, angles)
            astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
                
        ASTRA projector entry:
            astra.create_projector(proj_type, 
                                   proj_geom, 
                                   recon_geom)
            proj_type: 
                For CPU operation, use: line or strip or linear 
                For GPU operation, use: cuda
            
            Reconstruction geometry:
                default = (N, N) where N = number of original detectors in projection window
        '''
        proj_type = projector_type
        
        num_detector_pixels = sinogram.shape[1]
        if beam_geometry == 'parallel':
            rel_detector_size = 1.0 # No magnification in parallel beam geometry
            proj_geom = astra.create_proj_geom('parallel', 
                                               rel_detector_size,
                                               num_detector_pixels,
                                               self.angles)
        elif beam_geometry == 'fanflat':
            
            if self.Image_data_type == 'mcnp':
                cm_per_pixel = self.detector_length / 512 # 512 F5 point dets spreadout on a 20cm image plane
            elif self.Image_data_type == 'experimental':
                cm_per_pixel = self.detector_length / 2048 # 2048 pixels spreadout on a 20cm image plane
            
            if source_origin_cm:
                source_origin_cm = source_origin_cm # distance b/n source and COR  [cm]
                detector_origin_cm = detector_origin_cm # distance b/n COR and detector[cm]

                source_origin = source_origin_cm / cm_per_pixel # Source-COR distance [pixels]
                detector_origin = detector_origin_cm / cm_per_pixel # COR-detector distance [pixels]

                magnification = (source_origin + detector_origin) / source_origin
                detector_pixel_size = magnification
                
                proj_geom = astra.create_proj_geom('fanflat',
                                                   detector_pixel_size,
                                                   num_detector_pixels,
                                                   self.angles,
                                                   source_origin,
                                                   detector_origin)
            else:
                print("Fanbeam geometry selected but source_origin_distance not given")
            
            
        if not recon_dimension:
            rec_geometry = self.num_detectors_orig
        else:
            rec_geometry = recon_dimension
        
        vol_geom = astra.create_vol_geom(rec_geometry, rec_geometry)
        # Create the actual projector 
        proj_id = astra.create_projector(proj_type, 
                                         proj_geom, 
                                         vol_geom)
            
#        print('proj_id create: ', proj_id)
            
        return proj_id
        
    
    def Algorithm_FBP(self,
                      beam_geometry,
                      projector_type = 'cuda', 
                      rec_type = 'FBP_CUDA',
                      filter_type = 'ram-lak', 
                      recon_dimension = None, 
                      source_origin_cm = None,
                      detector_origin_cm = None):
        
        sinogram = self.sino_pad_roi
        source_origin_cm = source_origin_cm
        detector_origin_cm = detector_origin_cm
        
        proj_id = self.create_projector_id(sinogram,
                                           beam_geometry,
                                           projector_type,
                                           recon_dimension,
                                           source_origin_cm,
                                           detector_origin_cm)
        start = time.time()
        rec_fbp_id, rec_fbp = astra.create_reconstruction(rec_type,
                                                          proj_id,
                                                          sinogram, 
                                                          filterType = filter_type)
        end = time.time()
        print('FBP: ' + beam_geometry + ', ' + projector_type + ', elapsed time: ' + str(end - start) + ' seconds')
        
        # Extract the horizontal pixel profile from centerline region
        FBP_cl_h_profile = rec_fbp[int(rec_fbp.shape[0] / 2), :]
        
        # Plot reconstructed image
        plt.figure()
        plt.imshow(rec_fbp)
        plt.title('Reconstruction: FBP ' + beam_geometry)
        plt.colorbar()
        plt.gray()
        
        return rec_fbp, FBP_cl_h_profile
        
    
    def Algorithm_SIRT(self,
                       beam_geometry,
                       projector_type = 'cuda', 
                       rec_type = 'SIRT_CUDA',
                       iterations = 100,
                       use_minc = 'yes',
                       recon_dimension = None,
                       source_origin_cm = None,
                       detector_origin_cm = None):
        
        '''
        From initial trials, the ROI correction on sinogram (padding both sides by pad_roi provided)
        makes the SIRT reconstruction converge when still all gray image 
        
        This kind of makes sense since initial iterations can match the padded regions easily (repeated values)
        and this can lead to easier convergence, but convergence to clearly wrong solution
        
        As a result, sinogram with just COR correction used for SIRT reconstruction,
        Then bright circle (which comes as a result of lack of ROI correction) can be fixed later 
        '''
        
        sinogram = self.sino_pad_cor
        source_origin_cm = source_origin_cm
        detector_origin_cm = detector_origin_cm
        
        proj_id = self.create_projector_id(sinogram,
                                           beam_geometry,
                                           projector_type,
                                           recon_dimension,
                                           source_origin_cm,
                                           detector_origin_cm)
        
        start = time.time()
        rec_sirt_id, rec_sirt = astra.create_reconstruction(rec_type,
                                                            proj_id,
                                                            sinogram,
                                                            iterations = iterations,
                                                            use_minc = use_minc)
        end = time.time()
        print('SIRT: ' + beam_geometry + ', ' + projector_type + ', ' + str(iterations) + ' iterations, ' + 'elapsed time: ' + str(end - start) + ' seconds')
        
        # Extract the horizontal pixel profile from centerline region
        SIRT_cl_h_profile = rec_sirt[int(rec_sirt.shape[0] / 2), :]
        
        # Plot reconstructed image
        plt.figure()
        plt.imshow(rec_sirt)
        plt.title('Reconstruction: SIRT ' + beam_geometry)
        plt.colorbar()
        plt.gray()
        
        return rec_sirt, SIRT_cl_h_profile
        
    
    def Algorithm_TV_regularized(self, 
                                 beam_geometry,
                                 lam,
                                 lower_bound,
                                 upper_bound,
                                 projector_type = 'cuda', 
                                 num_inner_iter = 100,
                                 num_main_iter = 100,
                                 recon_dimension = None,
                                 source_origin_cm = None,
                                 detector_origin_cm = None,
                                 print_progress = True):
        
        '''
        Inner iterations:
            ---
        Main iterations:
            ---
        '''
        
        sinogram = self.sino_pad_roi
        source_origin_cm = source_origin_cm
        detector_origin_cm = detector_origin_cm
        
        proj_id = self.create_projector_id(sinogram,
                                           beam_geometry,
                                           projector_type,
                                           recon_dimension,
                                           source_origin_cm,
                                           detector_origin_cm)
        p = astra.OpTomo(proj_id)
        f = tvtomo.FISTA(p,
                         lam,
                         num_inner_iter,
                         bmin = lower_bound,
                         bmax = upper_bound)
        
        start = time.time()
        rec_tv = f.reconstruct(self.sino_pad_roi,
                               num_main_iter,
                               progress = print_progress)
        
        end = time.time()
        print('TV-FISTA: ' + beam_geometry + ', ' + projector_type + ', ' + str(num_main_iter) + ' iterations, ' + 'elapsed time: ' + str(end - start) + ' seconds')
        
        # Extract the horizontal pixel profile from centerline region
        TV_cl_h_profile = rec_tv[int(rec_tv.shape[0] / 2), :]
        
        # Plot reconstructed image
        plt.figure()
        plt.imshow(rec_tv)
        plt.title('Reconstruction: TV FISTA ' + beam_geometry)
        plt.colorbar()
        plt.gray()
        
        return rec_tv, TV_cl_h_profile
        
    
    
    def Recover_DC_value(self):
        
        '''
        Recover the DC value based on Volume conservation theorem 
            Page 7/47 of in https://web.eecs.umich.edu/~fessler/course/516/l/c-tomo.pdf
        '''


#''' Load image dataset '''
#Image_data = h5py.File('direct_contribution_images_720projections_0p25degree_0_to_180_degrees_py3.h5','r')
#Image_data['Projection_data'][:].shape
#Image_data['flat_field']
#Image_data['dark_field']
#
#Image_data_type = 'mcnp'
#P_level_Watts = 500    
#    
#Tomo_object = Neutron_CT(Image_data,
#                         P_level_Watts,
#                         Image_data_type = 'mcnp')    
#
#Tomo_object.Select_projection_window()
#
#sinogram = Tomo_object.Sinogram_current_corrected(subsamplefactor = 8,
#                                                  pad_cor = 5,
#                                                  pad_roi = 50,
#                                                  sino_slice = 0,
#                                                  pad_cor_shift_direction = 'left')
#
#rec_fbp, FBP_cl_h_profile = Tomo_object.Algorithm_FBP(beam_geometry = 'fanflat',
#                                                      projector_type = 'cuda', 
#                                                      rec_type = 'FBP_CUDA',
#                                                      filter_type = 'ram-lak', 
#                                                      source_origin_cm = 300,
#                                                      detector_origin_cm = 20)
#
#rec_sirt, SIRT_cl_h_profile = Tomo_object.Algorithm_SIRT(beam_geometry = 'fanflat',
#                                                         projector_type = 'cuda', 
#                                                         rec_type = 'SIRT_CUDA',
#                                                         iterations = 100,
#                                                         use_minc = 2*np.min(rec_fbp),
#                                                         source_origin_cm = 300,
#                                                         detector_origin_cm = 20)
#
#rec_tv, TV_cl_h_profile = Tomo_object.Algorithm_TV_regularized(beam_geometry = 'fanflat',
#                                                               lam = 1e-6,
#                                                               lower_bound = 2*np.min(rec_fbp),
#                                                               upper_bound = np.inf,
#                                                               projector_type = 'cuda', 
#                                                               num_inner_iter = 100,
#                                                               num_main_iter = 100,
#                                                               source_origin_cm = 300,
#                                                               detector_origin_cm = 20,
#                                                               print_progress = False)
#                    
#astra.projector.clear()
