from pathlib import Path
import os

from astropy.nddata import CCDData
from astropy.nddata.utils import block_reduce
from astropy.nddata.utils import Cutout2D
from astropy.visualization import hist
from astropy import visualization as aviz
from astropy.stats import mad_std
import astropy.units as u
import ccdproc as ccdp
import matplotlib.pyplot as plt
import numpy as np

DARK_IMAGETYP='dark'
BIAS_IMAGETYP='bias'
FLAT_IMAGETYP='flat'

def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10),
               cmap='viridis', log=False,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.
    Parameters
    ----------
    image
        The image to show
    percl : number
        The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
    percu : number or None
        The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """
    if percu is None:
        percu = percl
        percl = 100 - percl

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" '
                         'if you provide one of them')
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)


    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
    fig_size_pix = fig.get_size_inches() * fig.dpi

    ratio = (image.shape // fig_size_pix).max()

    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio

    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in the
        # reduced image. We do *not* want to do this for images which are
        # masks, since their values should be zero or one.
         reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    norm = aviz.ImageNormalize(reduced_data,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               stretch=stretch)

    if is_mask:
        # The image is a mask in which pixels should be zero or one.
        # block_reduce may have changed some of the values, so reset here.
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)

    if show_colorbar:
        # I haven't a clue why the fraction and pad arguments below work to make
        # the colorbar the same height as the image, but they do....unless the image
        # is wider than it is tall. Sticking with this for now anyway...
        # Thanks: https://stackoverflow.com/a/26720422/3486425
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # In case someone in the future wants to improve this:
        # https://joseph-long.com/writing/colorbars/
        # https://stackoverflow.com/a/33505522/3486425
        # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

def find_nearest_dark_exposure(
        science_frame : CCDData,
        dark_exposure_times,
        tolerance=0.5
    ) -> float:
    """
    Find the nearest exposure time of a dark frame to the exposure time of the image,
    raising an error if the difference in exposure time is more than tolerance.
    
    Parameters
    ----------
    
    image : astropy.nddata.CCDData
        Image for which a matching dark is needed.
    
    dark_exposure_times : list
        Exposure times for which there are darks.
    
    tolerance : float or ``None``, optional
        Maximum difference, in seconds, between the image and the closest dark. Set
        to ``None`` to skip the tolerance test.
    
    Returns
    -------
    
    float
        Closest dark exposure time to the image.
    """

    dark_exposures = np.array(list(dark_exposure_times))
    idx = np.argmin(np.abs(dark_exposures - science_frame.header['exptime']))
    closest_dark_exposure = dark_exposures[idx]

    if (tolerance is not None and 
        np.abs(science_frame.header['exptime'] - closest_dark_exposure) > tolerance):
        
        raise RuntimeError('Closest dark exposure time is {} for flat of exposure '
                           'time {}.'.format(closest_dark_exposure, science_frame.header['exptime']))
        
    
    return closest_dark_exposure

def select_dark_frame(
        science_frame : CCDData,
        dark_frames : ccdp.ImageFileCollection
    ) -> CCDData:
    """
    """
    keyed_frames = {ccd.header['exposure']: ccd for ccd in dark_frames.ccds(imagetyp=DARK_IMAGETYP, combined=True)}
    closest_dark = find_nearest_dark_exposure(science_frame=science_frame, dark_exposure_times=keyed_frames.keys(), tolerance=None)

    return keyed_frames[closest_dark]

def select_flat_frame(
        science_frame : CCDData,
        flat_frames : ccdp.ImageFileCollection
    ) -> CCDData:
    """
    """

    print(flat_frames) 
    combined_flats = { ccd.header['filter']: ccd for ccd in flat_frames.ccds(imagetyp=FLAT_IMAGETYP, combined=True) }

    flat_frame = combined_flats.get(science_frame.header['filter'], None)

    return flat_frame

def subtract_bias(
        frames : ccdp.ImageFileCollection,
        master_bias : CCDData,
        output_path : Path
    ) -> ccdp.ImageFileCollection:
    """
    Subtract bias from one of more frames

    Parameters
    ----------
    frames : ccdproc.ImageFileCollection
        Frames from which to subtract bias
    combined_bias : astropy.nddata.CCDData
        Combined/Calibrated bias data
    output_path : pathlib.Path
        Output directory to save CCD data
    

    Returns
    -------
    ccdproc.ImageFileCollection
        Calibrated frames

    """
    if(output_path is not None):
        output_path.mkdir(exist_ok=True)
    
    for ccd, file_name in frames.ccds(return_fname=True, save_location=output_path, unit='adu'):
        ccd = ccdp.subtract_bias(ccd, master_bias, unit='adu')
        ccd.write()
        
    return frames

def subtract_dark(
        frames : ccdp.ImageFileCollection,
        master_dark : CCDData,
        output_path: Path
    ) -> ccdp.ImageFileCollection:
    """
    Subtract dark from one of more frames

    Parameters
    ----------
    frames : ccdproc.ImageFileCollection
        Frames from which to subtract dark
    combined_bias : astropy.nddata.CCDData
        Combined/Calibrated bias data
    output_path : pathlib.Path
        Output directory to save CCD data
    

    Returns
    -------
    ccdproc.ImageFileCollection
        Calibrated frames

    """
    if(output_path is not None):
        output_path.mkdir(exist_ok=True)
    
    for ccd, file_name in frames.ccds(return_fname=True, save_location=output_path, ccd_kwargs={'unit': 'adu'}):
        ccd = ccdp.subtract_dark(ccd, master_dark, exposure_time='exptime', exposure_unit=u.second, scale=True)
        ccd.write()
    
    return frames
        
def calibrate_bias(
        science_frame : CCDData,
        bias_path : Path
    ) -> CCDData:
    """
    Calibrate and combine bias frames to create a master bias frame

    Parameters
    ----------
    science_frame : astropy.nddata.CCDData
        Science frame
    bias_path : pathlib.Path
        directory to search for bias frames

    Returns
    -------
    astropy.nddata.CCDData
        Calibrated master bias frame

    """
    fits_files = ccdp.ImageFileCollection(location=bias_path, find_fits_by_reading=False, glob_include='*.fits')
    bias_frames = fits_files.files_filtered(imagetyp=BIAS_IMAGETYP, include_path=True)
    
    combined_bias = ccdp.combine(
            bias_frames,
            method='average',
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=350e6,
            unit='adu'
    )

    combined_bias.meta['combined'] = True
    
    return combined_bias
    
def calibrate_dark(
        science_frame : CCDData,
        bias_path : Path,
        dark_path : Path,
        output_path : Path,
    ) -> ccdp.ImageFileCollection:
    """
    Calibrate and combine dark frames to create master dark frame(s)

    Parameters
    ----------
    science_frame : astropy.nddata.CCDData
        Science frame
    dark_path : pathlib.Path
        directory to search for dark frames
    master_bias : astropy.nddata.CCDData


    Returns
    -------
    ccdproc.ImageFileCollection
        Calibrated master dark frames

    """   
    filenames = []
    master_bias = generate_master_bias(science_frame, bias_path)
    fits_files = ccdp.ImageFileCollection(location=dark_path, find_fits_by_reading=False, glob_include='*.fits')
    
    if master_bias is not None:
        for ccd, fname in fits_files.ccds(
                    imagetyp=DARK_IMAGETYP,
                    return_fname=True,
                    save_location=output_path,
                    ccd_kwargs={'unit': 'adu'}):
            ccd = ccdp.subtract_bias(ccd, master_bias)
            ccd.write(output_path / fname, overwrite=False)
    
    dark_times = set(fits_files.summary['exptime'])
    
    for exp_time in sorted(dark_times):
        calibrated_darks = fits_files.files_filtered(imagetyp=DARK_IMAGETYP, exptime=exp_time, include_path=True)
        
        if len(calibrated_darks) != 0:
            combined_dark = ccdp.combine(
                    calibrated_darks,
                    method='average',
                    sigma_clip=True,
                    sigma_clip_low_thresh=5,
                    sigma_clip_high_thresh=5,
                    sigma_clip_func=np.ma.median,
                    sigma_clip_dev_func=mad_std,
                    mem_limit=350e6,
                    unit='adu'
            )

            combined_dark.meta['combined'] = True

            dark_file_name = output_path / 'master_dark_{:6.3f}.fits'.format(exp_time)
            filenames.append(dark_file_name)

            combined_dark.write(dark_file_name, overwrite=True)
        
    return ccdp.ImageFileCollection(location=output_path, filenames=filenames, find_fits_by_reading=False)
        
def calibrate_flat(
        science_frame : CCDData,
        flat_path : Path,
        bias_path : Path,
        dark_path : Path,
        output_path : Path
    ) -> ccdp.ImageFileCollection:

    filenames = []
    master_bias = generate_master_bias(science_frame, bias_path)
    master_dark = generate_master_dark(science_frame, bias_path=bias_path, dark_path=dark_path)
    
    fits_files = ccdp.ImageFileCollection(location=flat_path, find_fits_by_reading=False, glob_include='*.fits')
    
    if master_bias is not None:
        for ccd, fname in fits_files.ccds(
                    imagetyp=FLAT_IMAGETYP,
                    return_fname=True,
                    save_location=output_path,
                    ccd_kwargs={'unit': 'adu'}):
            ccd = ccdp.subtract_bias(ccd, master_bias)
            ccd.write(output_path / fname, overwrite=True)
            
    if master_dark is not None:
        for ccd, fname in fits_files.ccds(
                    imagetyp=FLAT_IMAGETYP,
                    return_fname=True,
                    save_location=output_path,
                    ccd_kwargs={'unit': 'adu'}):
            ccd = ccdp.subtract_dark(ccd, master_dark, exposure_time='exptime', exposure_unit=u.second, scale=True)
            ccd.write(output_path / fname, overwrite=True)
    
    flat_filters = set(h['filter'] for h in fits_files.headers(imagetyp=FLAT_IMAGETYP))
    
    print(flat_filters)
    
    for flat_filter in sorted(flat_filters):
        calibrated_flats = fits_files.files_filtered(
                imagetyp=FLAT_IMAGETYP,
                filter=flat_filter,
                include_path=True)
        
        if len(calibrated_flats) != 0:           
            combined_flat = ccdp.combine(
                    calibrated_flats,
                    method='average',
                    scale=lambda a: 1.0 / np.median(a),
                    sigma_clip=True,
                    sigma_clip_low_thresh=5,
                    sigma_clip_high_thresh=5,
                    sigma_clip_func=np.ma.median,
                    sigma_clip_dev_func=mad_std,
                    mem_limit=350e6,
                    unit='adu'
            )

            combined_flat.meta['combined'] = True

            flat_file_name = output_path / 'master_flat_{}.fits'.format(flat_filter.replace("''", "p"))
            filenames.append(flat_file_name)

            combined_flat.write(flat_file_name, overwrite=True)
        
    return ccdp.ImageFileCollection(location=output_path, filenames=filenames, find_fits_by_reading=False)

def generate_cache_path(
        science_frame : CCDData,
        base_path : Path
    ) -> Path:
    """
    """
    xdim = science_frame.header['naxis1']
    ydim = science_frame.header['naxis2']
    
    cache_path = base_path / '.cache' / '{0}x{1}'.format(xdim, ydim)
    
    return cache_path

def generate_master_bias(
        science_frame : CCDData,
        bias_path : Path,
        use_cache : bool=True
    ) -> CCDData:
    """
    """
    cache_path = generate_cache_path(science_frame, bias_path) / 'bias'
    cache_file = cache_path / 'master.fits'

    if use_cache and cache_file.is_file():
        ccd = CCDData.read(cache_file)
            
        if ccd is not None:
            return ccd
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    ccd = calibrate_bias(science_frame, bias_path)
    
    if ccd is not None:
        ccd.write(cache_file)
        
    return ccd

def generate_master_dark(
        science_frame : CCDData,
        bias_path : Path,
        dark_path : Path,
        use_cache : bool=True
    ) -> CCDData:
    """
    """
    cache_path = generate_cache_path(science_frame, dark_path) / 'dark'
    
    if use_cache and cache_path.is_dir():
        dark_frames = ccdp.ImageFileCollection(location=cache_path)
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        dark_frames = calibrate_dark(science_frame=science_frame, bias_path=bias_path, dark_path=dark_path, output_path=cache_path)       
    
    ccd = select_dark_frame(science_frame=science_frame, dark_frames=dark_frames)
    
    return ccd
    
def generate_master_flat(
        science_frame : CCDData,
        bias_path : Path,
        dark_path : Path,
        flat_path : Path,
        use_cache : bool=True
    ) -> CCDData:
    """
    """
    cache_path = generate_cache_path(science_frame, flat_path) / 'flat'
    
    if use_cache and cache_path.is_dir():
        flat_frames = ccdp.ImageFileCollection(location=cache_path)
    else:
        cache_path.mkdir(parents=True, exist_ok=True)
        flat_frames = calibrate_flat(science_frame=science_frame, bias_path=bias_path, dark_path=dark_path, flat_path=flat_path, output_path=cache_path)
    
    ccd = select_flat_frame(science_frame=science_frame, flat_frames=flat_frames)
    
    return ccd
    
    
def calibrate_science(
        science_frame : CCDData,
        flat_path : Path,
        bias_path : Path,
        dark_path : Path,
        subtract_bias=True,
        subtract_dark=True,
        divide_flat=True
    ) -> CCDData:
    """
    """
    master_bias = generate_master_bias(science_frame, bias_path)
    master_dark = generate_master_dark(science_frame, bias_path, dark_path)
    master_flat = generate_master_flat(science_frame, bias_path, dark_path, flat_path)
    
    if subtract_bias and master_bias is not None:
        science_frame = ccdp.subtract_bias(science_frame, master_bias)
    if subtract_dark and master_dark is not None:
        science_frame = ccdp.subtract_dark(science_frame, master_dark, exposure_time='exptime', exposure_unit=u.second, scale=True)
    if divide_flat and master_flat is not None:
        science_frame = ccdp.flat_correct(science_frame, master_flat)
    
    return science_frame
   
