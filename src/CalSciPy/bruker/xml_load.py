from __future__ import annotations
import os
from xml.etree import ElementTree
from .bruker_meta_objects import BrukerElementFactory
from .mark_points import PhotostimulationMeta
from .configuration_values import DEFAULT_PRAIRIEVIEW_VERSION


"""
Library of functions that extract useful information from PrairieView XML files
"""


def get_voltage_output(xml_file):
    """
    Compile the important metadata that pertains to the VoltageOutput in PrairieView.

    Parameters
    ----------
    xml_file : str
        Path to XML file for voltage output.

    Returns
    -------
    None.

    """
    # Import dependencies
    from xml.etree import ElementTree as et
    import os
    
    if (os.path.isdir(xml_file)):
        raise Exception("Path must point to a FILE, not a DIRECTORY!")
    
    if "VoltageOutput" in xml_file:
        # Read the xml file
        tree=et.parse(xml_file)
        root=tree.getroot()
    else:
        raise Exception("Incorrect function, call appropriate function for this type of XML")
    
    
    # Get the names of the waveforms used in the recording
    waveform_names = {}
    for waveform in root.findall("Waveform//"): # // Selects all subelements, on all levels beneath the current element
        for name in waveform.findall("Name"):
            waveform_names[name] = name.text
    waveform_names = list(waveform_names.values())
            
    # Get the channel names in the recording
    channel_names = {}
    for channel in root.findall("Waveform"):
        for name in channel.findall("Name"):
            channel_names[name] = name.text
    channel_names = list(channel_names.values())
            
    
    # Get the units of each channel
    channel_units = {}
    for channel in root.findall("Waveform"):
        for unit in channel.findall("Units"):
            channel_units[unit] = unit.text
    channel_units = list(channel_units.values())
    
    # Get the channels that are actually in use
    channel_in_use = {}
    for channel in root.findall("Waveform"):
        for usage in channel.findall("Enabled"):
            channel_in_use[usage] = usage.text
    channel_in_use = list(channel_in_use.values())
    
    def return_used_channels(channel_names, channel_units, channel_in_use):
        """
        Determine which channels are used and returns only those channels.

        Parameters
        ----------
        channel_names : string
            Display name of available channels.
        channel_units : string
            Units of available channels.
        channel_in_use : string
            Truly a boolean of which channels are enabled ("true") or disabled ("false")

        Returns
        -------
        channel_names : string
            Names of channels present in the recording.
        channel_units : string
            Units of the channels present in the recording.

        """
        # Find the indexes of the unused channels
        index_to_remove = []
        for index in range(len(channel_in_use)):
            if channel_in_use[index] == "false":
                index_to_remove.append(index)
        # Remove the unused channels
        for index in sorted(index_to_remove, reverse = True): # need to reverse, or index is messed up
            del channel_names[index]
            del channel_units[index]
        
        # Return the cleaned channel information
        return channel_names, channel_units
        
    
    # Get the sampling rate of the digitizer
    sampling_rate = root.findtext("UpdateRate")
    return sampling_rate, channel_names, channel_units


def get_voltage_recording(xml_file):
    """
    Extract time and number of samples from VoltageRecording XML file.

    Parameters
    ----------
    xml_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Import dependencies
    from xml.etree import ElementTree as et
    import os
    
    # Check for appropriate file to process
    if (os.path.isdir(xml_file)):
        raise Exception("Path must point to a FILE, not a DIRECTORY!")
    
    if "VoltageRecording_" in xml_file:
        # Read the xml file
        tree=et.parse(xml_file)
        root=tree.getroot()
    else:
        raise Exception("Incorrect function, call appropriate function for this type of XML")

    # Read the xml file
    tree=et.parse(xml_file)
    root=tree.getroot()
    
    # Get the number of samples recorded
    num_samples = root.findtext("SamplesAcquired")
    recording_date = root.findtext("DateTime")
    recording_duration = root.findtext("AcquisitionTime") # in milliseconds
    
    return num_samples, recording_date, recording_duration


def get_image_xml(xml_file):
    # Import dependencies
    from xml.etree import ElementTree as et
    import os
    import glob
    import numpy as np


    # Check for appropriate file to process
    if (os.path.isdir(xml_file)):
        raise Exception("Path must point to a FILE, not a DIRECTORY!")
    
    # Check that there are tif files in the directory
    parent_dir = os.path.basename(xml_file)
    tifs = glob.glob('*.tif')
    if tifs > 0:
        # Read the xml file
        tree=et.parse(xml_file)
        root=tree.getroot()
    else:
        raise Exception("Incorrect function, call appropriate function for this type of XML")
    
    # Check version of PrarieView used to generate the XML
    # XML structure varies from version to version so this needs to be accounted for.
    version = root.get("version")
    if version != "5.7.64.200":
        raise Exception("Function not configured for this version. Versions supported: 5.7.64.200 ")
    
    # Experiment notes
    try:
        notes = root.get("notes")
    except:
        pass
    
    # Get number of frames in file, use this to determine how time should be shaped
    try:
        frame_index = []
        for item in root.iter("Frame"):
            value = item.attrib['index']
            frame_index.append(value)
        frame_index = np.array(frame_index)
    except:
        pass
    
    try:
        relative_time = []
        for item in root.iter("Frame"):
           value = item.attrib['relativeTime']
           relative_time.append(value)
        relative_time = np.array(relative_time)
    except:
        pass   
    
    try: 
        absolute_time = []
        for item in root.iter("Frame"):
           value = item.attrib['absoluteTime']
           absolute_time.append(value)
        absolute_time = np.array(absolute_time)
    except:
        pass
    
    # Channel info
    try:
        count = 0 # enables iteration of iter method to a limited range
        channel_names = [] #File channel="1" channelName="Ch1 Red" filename="patch_voltage-003_Cycle00001_Ch1_000001.ome.tif" />
        for item in root.iter("File"):
            value = item.get('channelName')
            channel_names.append(value)
            count += 1
            # Change count limit if more than 2 channels are used for imaging
            if count >= 2:
                break
    except:
        pass
    try:
        count = 0 
        channel_nums = [] #File channel="1" channelName="Ch1 Red" filename="patch_voltage-003_Cycle00001_Ch1_000001.ome.tif" />
        for item in root.iter("File"):
            value = item.get("channel")
            channel_nums.append(value)
            count += 1
            # Change count limit if more than 2 channels are used for imaging
            if count >= 2:
                break
    except:
        pass
    
    # Laser information
    try:
        laser_power = []
        key_val = root.find(".//PVStateValue[@key='laserPower']")
        for item in key_val.iter("IndexedValue"):
            value = item.get("value")
            laser_power.append(value)
    except:
        pass
    try:
        laser_names = []
        key_val = root.find(".//PVStateValue[@key='laserPower']")
        for item in key_val.iter("IndexedValue"):
            value = item.get("description")
            laser_names.append(value)
    except:
        pass
    
    # Bit depth
    try:
        bit_depth = [] #bitDepth
        key_val = root.find(".//PVStateValue[@key='bitDepth']")
        bit_depth = key_val.get("value")
    except:
        pass
    
    
    # Pixel dwell time
    # Find units
    try:
        dwell_time = [] # PVStateValue #dwellTime
        key_val = root.find(".//PVStateValue[@key='dwellTime']")
        dwell_time = key_val.get("value")
    except:
        pass
    
    
    frame_period = {} # PVStateValue "framePeriod"
    objective_name = {} # PVStateValue "objectiveLens"
    objective_mag = {} # PVStateValue "objectiveLensMag"
    objective_na = {} # PVStateValue "objectiveLensNA"
    height_pixels = {} # PVStateValue "linesPerFrame"
    width_pixels = {} # PVStateValue "pixelsPerLine"
    microns_per_pixel = {} # PVStateValue "XAxis" "YAxis" "ZAxis" "value"
    optical_zoom = {} # PVStateValue "opticalZoom"
    pmt_gain = {} # PVStateValue "pmtGain", "index", "value", "description"
    scan_method = {} # PVStateValue "ResonantGalvo"
    
    #VoltageRecording name="Ephys" triggerMode="Start with next scan (PFI0)" cycle="1" index="1" configurationFile="patch_voltage-003_Cycle00001_VoltageRecording_001.xml" dataFile="patch_voltage-003_Cycle00001_VoltageRecording_001.csv"

    
    return notes, relative_time, absolute_time, frame_index, channel_names


def read_mark_points_xml(file_path: str, version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> PhotostimulationMeta:
    """

    :param file_path: path to xml file
    :type file_path: str or pathlib.Path
    :param version: version of prairieview
    :type version: str
    :return: photostimulation metadata
    :rtype: PhotostimulationMeta
    """

    # We generally expected the file to be structured such that
    # File/
    # ├── MarkPointSeriesElements
    # │   └──MarkPointElement
    # │      ├──GalvoPointElement
    # │      |      └──Point
    #        └──GalvoPointElement (Empty)
    # However, by using configurable mappings we do have some wiggle room

    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    # if it's imaging we can grab the version directly
    if "version" in root.attrib:
        version = root.attrib.get("version")

    bruker_element_factory = BrukerElementFactory(version)

    return PhotostimulationMeta(root, factory=bruker_element_factory)


if __name__ == "__main__":
    path = "C:\\Users\\YUSTE\\PycharmProjects\\CalSciPy.bruker\\scratchpad\\1000ms_50spirals_meta\\EM0275_03_13_23-002_Cycle00001_MarkPoints.xml"
    meta = read_mark_points_xml(path)
    print(f"{meta.sequence}")
    print(f"{meta.points[0]}")
