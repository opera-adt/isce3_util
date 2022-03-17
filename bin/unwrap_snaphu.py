import argparse
import os
import isce3
import isce3.unwrap.snaphu as snaphu
import yamale
import journal
import numpy as np
from ruamel.yaml import YAML

EXAMPLE = """example:
  unwrap_snaphu.py <custom_runconfig>   # Run with default and custom runconfigs
  unwrap_snaphu.py -h / --help          # help
"""

WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))

def command_line_parser():
    """
    Command line parser
    """
    parser = argparse.ArgumentParser(
        description="Unwrap interferograms using SNAPHU unwrapper",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('-r', '--custom_runconfig', type=str,
                        dest='run_config_path',
                        help='YAML custom runconfig for SNAPHU unwrapping. '
                             'Ignored if default runconfig snaphu.yaml is input.')
    return parser.parse_args()


def load_runconfig(cmd_opts):
    """
    Load SNAPHU runconfig. If cmd_opts.run_config_path
    is None, loads default runconfig. Otherwise, default
    runconfig is updated with user-defined options.
    Parameters
    ----------
    cmd_opts: NameSpace
       Command line option parser
    """
    error_channel = journal.error('unwrap_snaphu.load_runconfig')
    try:
        # Load schemas corresponding to SNAPHU and to validate against
        schema = yamale.make_schema(f'{WORKFLOW_SCRIPTS_DIR}/schemas/snaphu_schema.yaml',
                                    parser='ruamel')
    except:
        err_str = f'Unable to load the schema for SNAPHU unwrapper'
        error_channel.log(err_str)
        raise ValueError(err_str)

    runconfig_path = cmd_opts.run_config_path
    if os.path.isfile(runconfig_path):
        try:
            data = yamale.make_data(runconfig_path, parser='ruamel')
        except yamale.YamaleError as yamale_err:
            err_str = f'Yamale is unable to load the SNAPHU runconfig at {runconfig_path}'
            error_channel.log(err_str)
            raise yamale.YamaleError(err_str) from yamale_err
    else:
        err_str= f'Runconfig file at {runconfig_path} has not been found'
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    # Validate YAML file from command line
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as yamale_err:
        err_str = f'Schema validation failed for SNAPHU and runconfig at {runconfig_path}'
        error_channel.log(err_str)
        raise yamale.YamaleError(err_str) from yamale_err

    # Load user-provided runconfig
    parser = YAML(typ='safe')
    with open(runconfig_path, 'r') as f:
        runconfig = parser.load(f)
    return runconfig


def clean_none_cfg_keys(cfg):
    """
    Clean dict from None values
    Parameters
    ----------
    cfg: dict
      Dictionary to be cleaned
    Returns
    -------
    clean_cfg: dict
      Dictionary without None values
    """
    clean_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            nested = clean_none_cfg_keys(v)
            if len(nested.keys()) > 0:
                clean_cfg[k] = nested
        elif v is not None:
            clean_cfg[k] = v
    return clean_cfg


def check_dataset_group(cfg, igram_len, name):
    """
    Check ancillary raster count. Ancillary rasters
    are mask, power, unwrapped_phase_estimate. User
    can provide one raster or as many raster as the
    number of interferograms to unwrapped.
    cfg: dict
       Dictionary containing SNAPHU paramters
    igram_len: int
       Number of interferograms to unwrap
    name: str
       Name of dataset to check (e.g. 'mask')
    """
    error_channel = journal.error(
        'main.check_input_files.check_dataset_group')

    if (cfg.get(name) is not None):
        data_cfg = cfg[name]
        if len(data_cfg) > 1:
           if len(data_cfg) != igram_len:
               err_str = f'Assign one {name} or as many {name}s as number of wrapped igrams'
               error_channel.log(err_str)
           for file in data_cfg:
               if not os.path.exists(file):
                   err_str = f'{file} for dataset {name} does not exist'
                   error_channel.log(err_str)


def check_input_files(cfg):
    """
    Check input file group of SNAPHU runconfig
    Parameters
    ----------
    cfg: dict
      Dictionary containing SNAPHU parameters
    """
    error_channel = journal.error('main.check_input_files')
    # Check that input cfg is valid (not None)
    if cfg is None:
        err_str = 'SNAPHU parameter dictionary is not valid (None)'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Extract input & output files to check
    igram_paths = cfg['igram_raster']
    corr_paths = cfg['correlation_raster']
    ugram_paths = cfg['unwrapped_raster']
    conn_comp_paths = cfg['connected_components_raster']

    # Check consistency between number of interferograms and correlations
    if len(igram_paths) != len(corr_paths):
        err_str = "Number of interferograms must be consistent with number of correlations"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Check that user-defined interferograms and correlations exist
    for file_igram, file_corr in zip(igram_paths, corr_paths):
        if not os.path.exists(file_igram) or not os.path.exists(file_corr):
            err_str = f'{file_igram} or {file_corr} files do not exists'
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

    # If assigned, number of unwrapped/connected components file paths
    # must correspond to number of interferogram file paths
    if (ugram_paths is not None) and (len(ugram_paths) != len(igram_paths)):
        err_str = f'Number of unwrapped file paths must correspond to number of igram file paths'
        error_channel.log(err_str)
        raise ValueError(err_str)
    if (conn_comp_paths is not None) and (
            len(conn_comp_paths) != len(igram_paths)):
        err_str = f'Number of connected components file paths must correspond to number of igram file paths'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # For mask, power, and unwrapped phase estimate, use one raster for
    # all the interferograms or as many rasters as the number of interferograms
    check_dataset_group(cfg, len(igram_paths), 'mask')
    check_dataset_group(cfg, len(igram_paths), 'unwrapped_phase_estimate')
    check_dataset_group(cfg, len(igram_paths), 'power')


def set_topo_params(cost_cfg):
    """
    Set parameters for SNAPHU topo cost mode
    Parameters
    ----------
    cost_cfg: dict
      Dictionary containing SNAPHU cost parameters options
    Returns
    -------
    topo: isce3.unwrap.snaphu.TopoCostParams
       Topo cost parameter object with user-defined parameters
    """
    # Note: isce3.snaphu.unwrap.TopoCostParams is a frozen dataclass.
    # Therefore, attributes are not settable. We use dict.get to initialize object
    # dict.get() will assign the value stored in dict keyword if keyword is found
    # or a specified default value. More on assigned default values at:
    # https://github-fn.jpl.nasa.gov/isce-3/isce/blob/develop/python/packages/isce3/unwrap/snaphu.py

    # Note: bperp, near_range, dr, dz, range_res, az_red, wavelength, transmit_mode
    # altitude are required to initialize object. Schema will check their existence

    error_channel = journal.error('unwrap_snaphu.set_topo_params')

    if 'topo_parameters' in cost_cfg:
        cfg = cost_cfg['topo_parameters']
        cfg = clean_none_cfg_keys(cfg)
        topo = snaphu.TopoCostParams(bperp=cfg.get('bperp'),
                                     near_range=cfg.get('near_range'),
                                     dr=cfg.get('dr'), da=cfg.get('da'),
                                     range_res=cfg.get('range_res'),
                                     az_res=cfg.get('azimuth_res'),
                                     wavelength=cfg.get('wavelength'),
                                     transmit_mode=cfg.get('transmit_mode'),
                                     altitude=cfg.get('altitude'),
                                     earth_radius=cfg.get('earth_radius',
                                                          6378000.0),
                                     kds=cfg.get('kds', 0.02),
                                     specular_exp=cfg.get('specular_exp', 8.0),
                                     dzr_crit_factor=cfg.get('dzr_crit_factor',
                                                             2.0),
                                     shadow=cfg.get('shadow', False),
                                     dz_ei_min=cfg.get('dz_ei_min', -4.0),
                                     lay_width=cfg.get('layover_width', 16),
                                     lay_min_ei=cfg.get('layover_min_ei', 1.25),
                                     slope_ratio_factor=cfg.get(
                                         'slope_ratio_factor', 1.18),
                                     sigsq_ei=cfg.get('sigsq_ei', 100.0),
                                     drho=cfg.get('drho', 0.005),
                                     dz_lay_peak=cfg.get('dz_layover_peak',
                                                         -2.0),
                                     azdz_factor=cfg.get('azdz_factor', 0.99),
                                     dz_ei_factor=cfg.get('dz_ei_factor', 4.0),
                                     dz_ei_weight=cfg.get('dz_ei_weight', 0.5),
                                     dz_lay_factor=cfg.get('dz_layover_factor',
                                                           1.0),
                                     lay_const=cfg.get('layover_const', 0.9),
                                     lay_falloff_const=cfg.get(
                                         'layover_falloff_const', 2.0),
                                     sigsq_lay_factor=cfg.get(
                                         'sigsq_layover_factor', 0.1),
                                     krow_ei=cfg.get('krow_ei', 65),
                                     kcol_ei=cfg.get('kcol_ei', 257),
                                     init_dzr=cfg.get('init_dzr', 2048.0),
                                     init_dz_step=cfg.get('init_dz_step', 100),
                                     cost_scale_ambig_ht=cfg.get(
                                         'cost_scale_ambiguity_height', 80.0),
                                     dnom_inc_angle=cfg.get('dnom_inc_angle',
                                                            0.01),
                                     kpar_dpsi=cfg.get('kpar_dpsi', 7),
                                     kperp_dpsi=cfg.get('kperp_dpsi', 7))
    else:
        err_str = "bperp, near_range, dr, dz, range_res, az_red, wavelength, " \
                  "transmit_mode, altitude are required to initialize topo cost parameter object "
        error_channel.log(err_str)
        raise ValueError(err_str)
    return topo


def set_defo_params(cost_cfg):
    """
    Set parameters for SNAPHU deformation cost mode
    Parameters
    ----------
    cost_cfg: dict
      Dictionary containing SNAPHU cost parameters
    Returns
    -------
    defo: isce3.unwrap.snaphu.DefoCostParams
      Deformation cost parameter object with user-defined parameters
    """
    # If deformation parameter is not empty, extract it
    if cost_cfg.get('deformation_parameters') is not None:
        cfg = cost_cfg['deformation_parameters']
        cfg = clean_none_cfg_keys(cfg)
        defo = snaphu.DefoCostParams(azdz_factor=cfg.get('azdz_factor', 1.0),
                                     defo_max=cfg.get('defo_max', 1.2),
                                     sigsq_corr=cfg.get('sigsq_corr', 0.05),
                                     defo_const=cfg.get('defo_const', 0.9),
                                     lay_falloff_const=cfg.get(
                                         'layover_falloff_const', 2.0),
                                     kpar_dpsi=cfg.get('kpar_dpsi', 7),
                                     kperp_dpsi=cfg.get('kperp_dpsi', 7))
    else:
        defo = snaphu.DefoCostParams()
    return defo


def set_smooth_params(cost_cfg):
    """
    Set parameters for SNAPHU smooth cost mode
    Parameters
    ----------
    cost_cfg: dict
      Dictionary containing SNAPHU cost parametersoptions
    Returns
    -------
    smooth: isce3.unwrap.snaphu.SmoothCostParams
      Smooth cost parameter object with user-defined parameters
    """
    # If smooth parameters are present, extract smooth dict
    if cost_cfg.get('smooth_parameters') is not None:
        cfg = cost_cfg['smooth_parameters']
        cfg = clean_none_cfg_keys(cfg)
        smooth = snaphu.SmoothCostParams(kpar_dpsi=cfg.get('kpar_dpsi', 7),
                                         kperp_dpsi=cfg.get('kperp_dpsi', 7))
    else:
        # use all defaults values
        smooth = snaphu.SmoothCostParams()
    return smooth


def set_pnorm_params(cost_cfg):
    """
    Set parameters for SNAPHU P-Norm cost mode
    Parameters
    ----------
    cost_cfg: dict
      Dictionary containing SNAPHU cost parameter options
    Returns
    -------
    pnorm: isce3.unwrap.snaphu.PNormCostParams
      P-Norm cost parameter object with user-defined parameters
    """

    # If pnorm section of runconfig is not empty,
    # proceed to set user-defined pnorm parameters
    if cost_cfg.get('pnorm_parameters') is not None:
        cfg = cost_cfg['pnorm_parameters']
        cfg = clean_none_cfg_keys(cfg)
        pnorm = snaphu.PNormCostParams(p=cfg.get('lp_exp', 0.0),
                                       bidir=cfg.get('bidirection', True))
    else:
        pnorm = snaphu.PNormCostParams()
    return pnorm


def select_cost_options(cfg, cost_mode='defo'):
    """
    Select and set cost parameter object based
    on user-defined cost mode options
    Parameters
    ----------
    cfg: dict
      Dictionary containing SNAPHU parameter options
    cost_mode: str
      Snaphu cost mode. Default: defo
    Returns
    -------
    cost_params_obj: object
      Object corresponding to selected cost mode.
      E.g. if cost_mode is "defo", cost_params_obj
      is an instance of isce3.unwrap.snaphu.DefoCostParams()
    """
    error_channel = journal.error('unwrap_snaphu.select_cost_options')

    # If 'cost_mode_parameters' does not exist, create empty dictionary
    if cfg.get('cost_mode_parameters') is None:
        cfg['cost_mode_parameters'] = {}

    if cost_mode == 'topo':
        cost_params_obj = set_topo_params(cfg['cost_mode_parameters'])
    elif cost_mode == 'defo':
        cost_params_obj = set_defo_params(cfg['cost_mode_parameters'])
    elif cost_mode == 'smooth':
        cost_params_obj = set_smooth_params(cfg['cost_mode_parameters'])
    elif cost_mode == 'p-norm':
        cost_params_obj = set_pnorm_params(cfg['cost_mode_parameters'])
    else:
        err_str = f"{cost_mode} is not a valid cost mode option"
        error_channel.log(err_str)
        raise ValueError(err_str)
    return cost_params_obj


def get_raster(cfg, name, iter=0):
    """
    Select ancillary raster (e.g. mask) based
    on dataset name and number.
    Parameters:
    ----------
    cfg: dict
      Dictionary containing SNAPHU parameter options
    name: str
      Name of the ancillary raster to extract
    iter: int
      Iterator to extract correct dataset
    """
    if cfg.get(name) is not None:
        # If dataset exist, check number of allocated rasters
        if len(cfg[name]) == 1:
            path = cfg[name]
        # If more than one, use iter to extract correct raster
        else:
            path = cfg[name][iter]
        raster = isce3.io.gdal.Raster(path)
    else:
        raster = None
    return raster


def set_tile_params(snaphu_cfg):
    """
    Set user-defined tiling parameter options
    Parameters
    ----------
    snaphu_cfg: dict
       Dictionary containing SNAPHU options
    Returns:
    -------
    tile: isce3.unwrap.snaphu.TilingParams() or None
       Object containing tiling parameters options or None
    """
    # If 'tiling_parameters' is in snaphu_cfg, inspect setted
    # options. If None found, assign default
    if snaphu_cfg.get('tiling_parameters') is not None:
        cfg = snaphu_cfg['tiling_parameters']
        cfg = clean_none_cfg_keys(cfg)
        tile = snaphu.TilingParams(nproc=cfg.get('nproc', 1),
                                   tile_nrows=cfg.get('tile_nrows', 1),
                                   tile_ncols=cfg.get('tile_ncols', 1),
                                   row_overlap=cfg.get('row_overlap', 0),
                                   col_overlap=cfg.get('col_overlap', 0),
                                   tile_cost_thresh=cfg.get('tile_cost_thresh',
                                                            500),
                                   min_region_size=cfg.get('min_region_size',
                                                           100),
                                   tile_edge_weight=cfg.get('tile_edge_weight',
                                                            2.5),
                                   secondary_arc_flow_max=cfg.get(
                                       'secondary_arc_flow_max', 8),
                                   single_tile_reoptimize=cfg.get(
                                       'single_tile_reoptimize', False))
    else:
        tile = None
    return tile


def set_solver_params(snaphu_cfg):
    """
    Set user-defined solver parameter options
    Parameters
    ----------
    snaphu_cfg: dict
      Dictionary containing SNAPHU options
    Returns:
    -------
    solver: isce3.unwrap.snaphu.SolverParams() or None
      Object containing solver parameters options or None
    """
    # If 'solver_parameters' is in snaphu_cfg, inspect setted
    # options. If None found, assign default
    if snaphu_cfg.get('solver_parameters') is not None:
        cfg = snaphu_cfg['solver_parameters']
        cfg = clean_none_cfg_keys(cfg)
        solver = snaphu.SolverParams(max_flow_inc=cfg.get('max_flow_inc', 4),
                                     init_max_flow=cfg.get('initial_max_flow',
                                                           9999),
                                     arc_max_flow_const=cfg.get(
                                         'arc_max_flow_const', 3),
                                     threshold=cfg.get('threshold', 0.001),
                                     max_cost=cfg.get('max_cost', 1000.0),
                                     cost_scale=cfg.get('cost_scale', 100.0),
                                     n_cycle=cfg.get('n_cycle', 200),
                                     max_new_node_const=cfg.get(
                                         'max_new_node_const', 0.0008),
                                     max_n_flow_cycles=cfg.get(
                                         'max_n_flow_cycles', None),
                                     max_cycle_frac=cfg.get('max_cycle_frac',
                                                            0.00001),
                                     n_conn_node_min=cfg.get('n_conn_node_min',
                                                             0),
                                     n_major_prune=cfg.get('n_major_prune',
                                                           2000000000),
                                     prune_cost_thresh=cfg.get(
                                         'prune_cost_threshold', 2000000000))
    else:
        solver = None
    return solver


def set_connected_components_params(snaphu_cfg):
    """
    Set user-defined connected components parameter options
    Parameters
    ----------
    snaphu_cfg: dict
      Dictionary containing SNAPHU options
    Returns:
    -------
    conn: isce3.unwrap.snaphu.ConnCompParams() or None
      Object containing connected components parameters options or None
    """
    # If 'connected_components_parameters' is in snaphu_cfg,
    # inspect setted options. If None found, assign default

    if snaphu_cfg.get('connected_components_parameters') is not None:
        cfg = snaphu_cfg['connected_components_parameters']
        cfg = clean_none_cfg_keys(cfg)
        conn = snaphu.ConnCompParams(
            min_frac_area=cfg.get('min_frac_area', 0.01),
            cost_thresh=cfg.get('cost_threshold', 300),
            max_ncomps=cfg.get('max_ncomps', 32))
    else:
        conn = None
    return conn


def set_corr_bias_params(snaphu_cfg):
    """
    Set user-defined correlation bias parameter options
    Parameters
    ----------
    snaphu_cfg: dict
      Dictionary containing SNAPHU options
    Returns:
    -------
    corr: isce3.unwrap.snaphu.CorrBiasModelParams or None
      Object containing correlation bias model parameters options or None
    """
    if snaphu_cfg.get('correlation_bias_parameters') is not None:
        cfg = snaphu_cfg['correlation_bias_parameters']
        cfg = clean_none_cfg_keys(cfg)
        corr = snaphu.CorrBiasModelParams(c1=cfg.get('c1', 1.3),
                                          c2=cfg.get('c2', 0.14),
                                          min_corr_factor=cfg.get(
                                              'min_corr_factor', 1.25))
    else:
        corr = None
    return corr


def set_phase_std_params(snaphu_cfg):
    """
    Set user-defined phase standard deviation
    model parameter options
    Parameters
    ----------
    snaphu_cfg: dict
      Dictionary containing SNAPHU options
     Returns:
     -------
    std: isce3.unwrap.snaphu.PhaseStddevModelParams or None
      Object containing phase std model parameters options or None
    """
    if snaphu_cfg.get('phase_std_parameters') is not None:
        cfg = snaphu_cfg['phase_std_parameters']
        cfg = clean_none_cfg_keys(cfg)
        std = snaphu.PhaseStddevModelParams(c1=cfg.get('c1', 0.4),
                                            c2=cfg.get('c2', 0.35),
                                            c3=cfg.get('c3', 0.06),
                                            sigsq_min=cfg.get('sigsq_min', 1))
    else:
        std = None
    return std


def compute_nlooks(rg_igram_step, az_igram_step,
                   rg_res, az_res):
    '''
    Compute effective number of looks
    Parameters
    ----------
    rg_igram_step: float
      Interferogram spacing along slant range
    az_igram_step: float
      Interferogram spacing along azimuth
    rg_res : float
      Slant range resolution
    az_res: float
      Azimuth resolution
    Returns:
    -------
    nlooks: float
       Effective number of looks
    '''

    nlooks = (rg_igram_step * az_igram_step) / (rg_res * az_res)
    return nlooks


def unwrap_snaphu(snaphu_cfg, ugram, conn_comp, igram, corr,
                  nlooks, scratchdir=None, power=None, mask=None, unw_est=None):
    """
    Unwrap igram using SNAPHU and user-defined parameters
    and ancillary rasters
    Parameters:
    snaphu_cfg: dict
      Dictionary containing SNAPHU parameter options
    ugram: isce3.io.gdal.Raster
      Unwrapped interferogram raster (gdal.GDT_Float32)
    conn_comp: isce3.io.gdal.Raster
      Connected component raster (gdal.GDT_Byte)
    igram: isce3.io.gdal.Raster
      Complex wrapped interferogram raster (gdal.GDT_CFloat32)
    corr: isce3.io.gdal.Raster
      Normalized interferometric correlation amplitude (gdal.GDT_Float32)
    nlooks: float
      Number of effective looks
    scratchdir: str
      Path to scratch directory
    power: isce3.io.gdal.Raster (or None)
      Slc power raster in the reference SLC geometry
    mask: isce3.io.gdal.Raster (or None)
      Binary mask to apply during unwrapping (gdal.GDT_Byte)
    unw_est: isce3.io.gdal.Raster (or None)
      Coarse unwrapped phase estimate (gdal.GDT_Float32)
    """
    # Get cost mode and corresponding cost parameter object
    # assign deformation if not found
    snaphu_cfg = clean_none_cfg_keys(snaphu_cfg)
    cost_mode = snaphu_cfg.get('cost_mode', 'defo')
    cost_params = select_cost_options(snaphu_cfg, cost_mode)

    # Get initialization method. If not found, assign MCF
    init_method = snaphu_cfg.get('initialization_method', 'mcf')

    # Get tiling parameters
    tile_params = set_tile_params(snaphu_cfg)
    # Get solver parameters
    solver_params = set_solver_params(snaphu_cfg)
    # Get connected components parameters
    conn_comp_params = set_connected_components_params(snaphu_cfg)
    # Get correlation bias model parameters
    corr_model_params = set_corr_bias_params(snaphu_cfg)
    # Get phase standard deviation model parameters
    phase_std_params = set_phase_std_params(snaphu_cfg)

    verbose = snaphu_cfg['verbose'] if snaphu_cfg.get(
        'verbose') is not None else False
    debug = snaphu_cfg['debug'] if snaphu_cfg.get(
        'debug') is not None else False

    # All parameters are set. Ready to unwrap
    snaphu.unwrap(ugram, conn_comp, igram, corr,
                  nlooks, cost_mode, cost_params, init_method, power,
                  mask, unw_est, tile_params, solver_params,
                  conn_comp_params, corr_model_params, phase_std_params,
                  scratchdir, verbose, debug)


def main(cfg):
    """
    Main function. cfg: snaphu dictionary
    """

    # Check that input files are valid
    check_input_files(cfg)

    # Check output directory
    if (cfg.get('scratchdir') is not None) and (
            not os.path.exists(cfg.get('scratchdir'))):
        os.mkdir(cfg['scratchdir'], exist_ok=True)

    # Loop over interferograms to unwrap
    for k in range(len(cfg['igram_raster'])):
        # Put interferogram and correlation in a ISCE3 raster
        igram_raster = isce3.io.gdal.Raster(cfg['igram_raster'][k])
        corr_raster = isce3.io.gdal.Raster(cfg['correlation_raster'][k])

        # If ugram_raster and connected components are None use igram basename
        if cfg['unwrapped_raster'] is None:
            basename = os.path.basename(cfg['igram_raster'][k])
            ugram_name = f'{basename}.ugram'
            conn_comp_name = ugram_name.replace('ugram', 'conn_comp')
        else:
            ugram_name = cfg['unwrapped_raster'][k]
            conn_comp_name = cfg['connected_components_raster'][k]

        # Create unwrapped and connected components rasters.
        ugram_raster = isce3.io.gdal.Raster(ugram_name, igram_raster.width,
                                            igram_raster.length, np.float32,
                                            "ENVI")
        conn_comp_raster = isce3.io.gdal.Raster(conn_comp_name,
                                                igram_raster.width,
                                                igram_raster.length, np.uint32,
                                                "ENVI")

        # Check allocation of power, mask, unwrapped phase estimate
        power_raster = get_raster(cfg, 'power', k)
        mask_raster = get_raster(cfg, 'mask', k)
        unw_est_raster = get_raster(cfg, 'unwrapped_phase_estimate', k)

        # Proceed to unwrapping
        unwrap_snaphu(cfg, ugram_raster, conn_comp_raster, igram_raster,
                      corr_raster, cfg.get('nlooks'), cfg.get('scratchdir'),
                      power_raster, mask_raster, unw_est_raster)


if __name__ == '__main__':
    # Get command line parser
    opts = command_line_parser()

    # Load runconfig
    runconfig = load_runconfig(opts)

    # Main function
    main(runconfig['snaphu'])
