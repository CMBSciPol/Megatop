import logging
import os
import time
import warnings

import healpy as hp
import numpy as np
import yaml


class BBmeta:
    """
    Metadata manager for the BBmaster pipeline.
    The purpose of this class is to provide
    a single interface to all the parameters and products
    that will be used from different stages of the pipeline.
    """

    def __init__(self, fname_config):
        """
        Initialize the pipeline manager from a yaml file.

        Parameters
        ----------
        fname_config : str
            Path to the yaml file with the configuration.
        """

        # Load the configuration file
        with open(fname_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Set the high-level parameters as attributes
        for key in self.config:
            setattr(self, key, self.config[key])

        # Set all the `_directory` attributes
        self._set_directory_attributes()

        # Set the general attributes (nside, lmax, etc...)
        self._set_general_attributes()

        # Copy the configuration file to output directory
        with open(f"{self.output_dirs['root']}/config.yaml", "w") as f:
            yaml.dump(self.config, f)

        # Basic sanity checks
        if self.lmax > 3 * self.nside - 1:
            raise ValueError(f"lmax should be lower or equal to 3*nside-1 = {3 * self.nside - 1}")

        # Initialize method to parse map_sets metadata
        map_sets_attributes = list(self.map_sets[next(iter(self.map_sets))].keys())
        for map_sets_attribute in map_sets_attributes:
            self._init_getter_from_map_set(map_sets_attribute)

        # frequency from map_set
        self.frequencies = [self.freq_tag_from_map_set(m) for m in self.map_sets]

        # A list of the maps used in the analysis
        self.maps_list = self._get_map_list()

        # Masks
        self._init_masks_params()

        # Simulation
        if self.map_sim_pars is not None:
            self._init_simulation_params()

        # Initialize a timer
        self.timer = Timer()

        # Initialize a logger
        debug = False  # TODO fixed for now
        logging.basicConfig(
            format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.DEBUG if debug else logging.INFO,
        )
        self.logger = logging.getLogger()

    def _set_directory_attributes(self):
        """
        Set the directory attributes that are listed
        in the paramfiles
        """
        for label, path in self.data_dirs.items():
            if label == "root":
                self.data_dir = self.data_dirs["root"]
            else:
                # full_path = f"{self.data_dirs['root']}/{path}"
                # Using os.path.join can handle cases where the path is already
                # a full path. 'root' is then overrulled.
                full_path = os.path.join(self.data_dirs["root"], path)
                setattr(self, label, full_path)
                setattr(self, f"{label}_rel", path)
                try:
                    os.makedirs(full_path, exist_ok=True)
                except PermissionError:
                    print(f"PermissionError: Could not create {full_path}")
                    print("We continue without creating the directory")

        for label, path in self.output_dirs.items():
            if label == "root":
                self.output_dir = self.output_dirs["root"]
            else:
                # full_path = f"{self.output_dirs['root']}/{path}"
                full_path = os.path.join(self.output_dirs["root"], path)
                setattr(self, label, full_path)
                setattr(self, f"{label}_rel", path)
                os.makedirs(full_path, exist_ok=True)

    def _set_general_attributes(self):
        """ """
        for key, value in self.general_pars.items():
            setattr(self, key, value)

    def _get_map_list(self):
        """
        List the different maps (including splits).
        Constructor for the map_list attribute.
        """
        out_list = list(self.map_sets.keys())
        return out_list

    def _init_getter_from_map_set(self, map_set_attribute):
        """
        Initialize a getter for a map_set attribute.

        Parameters
        ----------
        map_set_attribute : str
            Should a key of the map_set dictionnary.
        """
        setattr(
            self,
            f"{map_set_attribute}_from_map_set",
            lambda map_set: self.map_sets[map_set][map_set_attribute],
        )

    def _get_galactic_mask_name(self):
        """
        Get the name of the galactic mask.
        """
        fname = f"{self.masks['galactic_mask_root']}_{self.masks['galactic_mask_mode']}.fits"
        return os.path.join(self.mask_directory, fname)

    def _get_binary_mask_name(self):
        """
        Get the name of the binary or survey mask.
        """
        return os.path.join(self.mask_directory, self.masks["binary_mask"])

    def _get_point_source_mask_name(self):
        """
        Get the name of the point source mask.
        """
        return os.path.join(self.mask_directory, self.masks["point_source_mask"])

    def _get_analysis_mask_name(self):
        """
        Get the name of the final analysis mask.
        """
        return os.path.join(self.mask_directory, self.masks["analysis_mask"])

    def _get_nhits_map_name(self):
        """
        Get the name of the hits counts map.
        """
        if not self.use_input_nhits:
            # Not using custom nhits map
            return os.path.join(self.mask_directory, self.masks["nhits_map"])
        else:
            # Using custom nhits map
            return self.masks["input_nhits_path"]

    def idx_from_list(self, frequencies):
        if not (set(self.frequencies) <= set(frequencies)):
            raise Exception(
                f"Some frequencies are not part of {frequencies} (can't compute noise for them). Check your yaml !"
            )
        return [frequencies.index(fr) for fr in self.frequencies]

    def read_mask(self, mask_type):
        """
        Read the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        """
        return hp.ud_grade(
            hp.read_map(getattr(self, f"{mask_type}_mask_name")), nside_out=self.nside
        )

    def save_mask(self, mask_type, mask, overwrite=False):
        """
        Save the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        mask : array-like
            Mask to save.
        overwrite : bool, optional
            Overwrite the mask if it already exists.
        """
        return hp.write_map(
            getattr(self, f"{mask_type}_mask_name"), mask, overwrite=overwrite, dtype=np.float32
        )

    def read_hitmap(self):
        """
        Read the hitmap. For now, we assume that all tags
        share the same hitmap.
        """
        hitmap = hp.read_map(self.nhits_map_name)
        return hp.ud_grade(hitmap, self.nside, power=-2)

    def save_hitmap(self, map, overwrite=True):
        """
        Save the hitmap to disk.

        Parameters
        ----------
        map : array-like
            Mask to save.
        """
        hp.write_map(
            os.path.join(self.mask_directory, self.masks["nhits_map"]),
            map,
            dtype=np.float32,
            overwrite=overwrite,
        )

    def read_nmt_binning(self):
        """
        Read the binning file and return the corresponding NmtBin object.
        """
        import pymaster as nmt

        binning = np.load(self.path_to_binning)
        return nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)

    def get_n_bandpowers(self):
        """
        Read the binning file and return the number of ell-bins.
        """
        binner = self.read_nmt_binning()
        return binner.get_n_bands()

    def get_effective_ells(self):
        """
        Read the binning file and return the number of ell-bins.
        """
        binner = self.read_nmt_binning()
        return binner.get_effective_ells()

    def read_beam(self, map_set):
        """ """
        file_root = self.file_root_from_map_set(map_set)
        beam_file = f"{self.beam_directory}/beam_{file_root}.dat"
        l, bl = np.loadtxt(beam_file, unpack=True)  # noqa: E741
        return l, bl

    def _init_masks_params(self):
        """ " """
        keys = [
            "analysis_mask",
            "nhits_map",
            "binary_mask",
            "point_source_mask",
            "binary_mask_zero_threshold",
            "include_in_mask",
            "apod_radius",
            "apod_type",
        ]

        # Determine if input hit counts map exists
        self.use_input_nhits = self.masks["input_nhits_path"] is not None

        # Determine if input point source maps exists or is needed
        self.use_input_point_source = (
            "input_point_source_path" in self.masks
            and "point_source" in self.masks["include_in_mask"]
        )

        # Initialize masks file_names
        for mask_type in [
            "binary_mask",
            "analysis_mask",
            "nhits_map",
        ]:
            setattr(self, f"{mask_type}_name", getattr(self, f"_get_{mask_type}_name")())

        if "galactic" in self.masks["include_in_mask"]:
            keys += ["galactic_mask_root", "galactic_mask_mode"]
            setattr(self, "galactic_mask_name", getattr(self, "_get_galactic_mask_name")())
        if "point_source" in self.masks["include_in_mask"]:
            keys += ["apod_radius_point_source"]
            setattr(self, "point_source_mask_name", getattr(self, "_get_point_source_mask_name")())
            if not self.use_input_point_source:
                keys += ["mock_nsrcs", "mock_srcs_hole_radius"]

        missing_keys = [key for key in keys if key not in self.masks]
        if missing_keys:
            raise KeyError(f"Missing keys in masks: {missing_keys}")

    def _init_simulation_params(self):
        """
        Loop over the simulation parameters and set them as attributes.
        """
        if not hasattr(self, "noise_sim_pars"):
            raise AttributeError("The 'noise_sim_pars' field is missing from the config file.")

        # Check for inconsistent CMB simulation settings
        self.sky_model = self.map_sim_pars["sky_model"]
        if self.map_sim_pars["cmb_sim_no_pysm"]:
            for cmb in ["c1", "c2", "c3", "c4"]:
                if cmb in self.sky_model:
                    warnings.warn(
                        "You specified a PySM CMB model while setting 'cmb_sim_no_pysm' to False. Dropping the PySM CMB model."
                    )
                    self.sky_model.remove(cmb)
            if not hasattr(self, "fiducial_cmb"):
                raise AttributeError("The 'fiducial_cmb' field is missing from the config file.")
            keys = ["r_input", "A_lens"]
            missing_keys = [key for key in keys if key not in self.map_sim_pars]
            if missing_keys:
                raise KeyError(f"Missing keys in map_sim_pars: {missing_keys}")

        # noise checks
        if self.noise_sim_pars["noise_option"] not in [
            "white_noise",
            "no_noise",
            "noise_spectra",
            "MSS2",
        ]:
            raise KeyError(
                "Only no_noise, white_noise, noise_spectra and MSS2 noise options are supported for now ..."
            )
        if self.noise_sim_pars["experiment"] not in ["SO", "MSS2"]:
            raise KeyError("Only SO simulations supported for now ")
        if self.noise_sim_pars["experiment"] == "SO":
            keys = ["sensitivity_mode", "SAC_yrs_LF"]
            missing_keys = [key for key in keys if key not in self.noise_sim_pars]
            if missing_keys:
                raise KeyError(f"Missing keys in noise_sim_pars: {missing_keys}")

    def save_fiducial_cl(self, ell, cl_dict, cl_type):
        """
        Save a fiducial power spectra dictionary to disk and return file name.

        Parameters
        ----------
        ell : array-like
            Multipole values.
        cl_dict : dict
            Dictionnary with the power spectra.
        cl_type : str
            Type of power spectra.
            Can be "cosmo", "tf_est" or "tf_val".
        """
        fname = getattr(self, f"{cl_type}_cls_file")
        np.savez(fname, l=ell, **cl_dict)
        return fname

    def load_fiducial_cl(self, cl_type):
        """
        Load a fiducial power spectra dictionary from disk.

        Parameters
        ----------
        cl_type : str
            Type of power spectra.
            Can be "cosmo", "tf_est" or "tf_val".
        """
        fname = getattr(self, f"{cl_type}_cls_file")
        return np.load(fname)

    def plot_dir_from_output_dir(self, out_dir):
        """ """
        root = self.output_dir

        if root in out_dir:
            path_to_plots = out_dir.replace(f"{root}/", f"{root}/plots/")
        else:
            path_to_plots = f"{root}/plots/{out_dir}"

        os.makedirs(path_to_plots, exist_ok=True)

        return path_to_plots

    def get_fname_mask(self, map_type="analysis"):
        """
        Get the full filepath to a mask of predefined type.

        Parameters
        ----------
        map_type : str
            Choose between 'analysis', 'binary', 'point_source'.
            Defaults to 'analysis'.
        """
        base_dir = self.mask_directory
        if map_type == "analysis":
            fname = os.path.join(base_dir, self.masks["analysis_mask"])
        elif map_type == "binary":
            fname = os.path.join(base_dir, self.masks["binary_mask"])
        elif map_type == "point_source":
            fname = os.path.join(base_dir, self.masks["point_source_mask"])
        else:
            raise ValueError(
                "The map_type chosen does not exits. "
                "Choose between 'analysis', 'binary', "
                "'point_source'."
            )
        return fname

    def get_fname_cls_fiducial_cmb(self, cl_type="lensed"):
        """
        Get the full filepath to a fiducial CMB power spectra of predefined type.

        Parameters
        ----------
        cl_type : str
            Choose between 'lensed', 'unlensed_scalar_tensor_r1'.
            Defaults to 'lensed'.
        """
        base_dir = self.fiducial_cmb["cls_cmb_directory"]
        if cl_type == "lensed":
            fname = os.path.join(base_dir, self.fiducial_cmb["lensed"])
        elif cl_type == "unlensed_scalar_tensor_r1":
            fname = os.path.join(base_dir, self.fiducial_cmb["unlensed_scalar_tensor_r1"])
        else:
            raise ValueError(
                "The cl_type chosen does not exits. "
                "Choose between 'lensed', 'unlensed_scalar_tensor_r1'."
            )
        return fname

    def get_map_filename(self, map_set, id_split, id_sim=None):
        """
        Get the path to file for a given `map_set` and split index.
        Can also get the path to a given simulation if `id_sim` is provided.

        Path to standard map:
            {map_directory}/{map_set_root}_split_{id_split}.fits
        Path to sim map: e.g.
            {sims_directory}/0000/{map_set_root}_split_{id_split}.fits

        Parameters
        ----------
        map_set : str
            Name of the map set.
        id_split : int
            Index of the split.
        id_sim : int, optional
            Index of the simulation.
            If None, return the path to the data map.
        """
        map_set_root = self.file_root_from_map_set(map_set)
        if id_sim is not None:
            path_to_maps = os.path.join(self.sims_directory, f"{id_sim:04d}")
            os.makedirs(path_to_maps, exist_ok=True)
        else:
            path_to_maps = self.map_directory

        if id_split is None:
            return os.path.join(path_to_maps, f"{map_set_root}.fits")
        else:
            return os.path.join(path_to_maps, f"{map_set_root}_split_{id_split}.fits")

    def read_map(self, map_set, id_split, id_sim=None, pol_only=False):
        """
        Read a map given a map set and split index.
        Can also read a given covariance simulation if `id_sim` is provided.

        Parameters
        ----------
        map_set : str
            Name of the map set.
        id_split : int
            Index of the split.
        id_sim : int, optional
            Index of the simulation.
            If None, return the data map.
        pol_only : bool, optional
            Return only the polarization maps.
        """
        field = [1, 2] if pol_only else [0, 1, 2]
        fname = self.get_map_filename(map_set, id_split, id_sim)
        return hp.read_map(fname, field=field)

    def get_noise_map_filename(self, map_set):
        """
        Returns the noise map filename.

        Path to standard map:
            {noise_map_directory}/{map_set_noise}.fits

        Args:
            map_set (str): The map set.

        Returns:
            str: The noise map filename.
        """
        try:
            path_to_maps = self.noise_maps_directory
        except AttributeError:
            path_to_maps = self.mock_directory

        map_set_root = self.noise_root_from_map_set(map_set)
        return os.path.join(path_to_maps, map_set_root + ".fits")

    def get_nhits_map_filename(self, map_set):
        """
        Returns the nhits map filename.

        Path to standard map:
            {nhits_map_directory}/{map_set_nhits}.fits

        Args:
            map_set (str): The map set.

        Returns:
            str: The nhits map filename.
        """
        path_to_maps = self.nhits_directory
        map_set_root = self.nhits_map_path_from_map_set(map_set)
        return os.path.join(path_to_maps, map_set_root + ".fits")


class Timer:
    """
    Basic timer class to time different
    parts of pipeline stages.
    """

    def __init__(self):
        """
        Initialize the timers with an empty dict
        """
        self.timers = {}

    def start(self, timer_label):
        """
        Start the timer with a given label. It allows
        to time multiple nested loops using different labels.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        """
        if timer_label in self.timers:
            raise ValueError(f"Timer {timer_label} already exists.")
        self.timers[timer_label] = time.time()

    def stop(self, timer_label, logger, text_to_output=None):
        """
        Stop the timer with a given label.
        Allows to output a custom text different
        from the label.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        text_to_output : str, optional
            Text to output instead of the timer label.
            Defaults to None.
        verbose : bool, optional
            Print the output text.
            Defaults to True.
        """
        if timer_label not in self.timers:
            raise ValueError(f"Timer {timer_label} does not exist.")

        dt = time.time() - self.timers[timer_label]
        self.timers.pop(timer_label)
        prefix = f"[{text_to_output}]" if text_to_output else f"[{timer_label}]"
        logger.info(f"{prefix} took {dt:.02f}s to process.")
