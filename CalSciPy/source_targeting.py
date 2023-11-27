from CalSciPy.optogenetics import randomize_targets
from CalSciPy.roi_tools import Suite2PHandler
from visualize_optogenetics import view_target_overlay
from CalSciPy.bruker.protocols.mark_points import generate_marked_points_protocol

path_1 = Path("Y:\\EM0564_PRE_SCREEN_11_1_23-001")

path_2 = Path("C:\\Users\\YUSTE\\Desktop\\EM0566_SET")

photostim = Photostimulation.import_rois(Suite2PHandler, path_2.joinpath("suite2p").joinpath("plane0"))

sub_targets = np.random.choice(237, size=25, replace=False)

targets = randomize_targets(sub_targets, targets_per_group=25, trials=15)

targets_file = {key: group[0] for key, group in enumerate(targets)}
np.save("C:\\Users\\YUSTE\\Desktop\\EM0566_SET\\targets_file.npy", targets_file)

for key, group in enumerate(targets):
    photostim.add_photostimulation_group(ordered_index=group[0], delay=5000.0, point_interval=4900.0,
                                         name=f"Trial {key}")

generate_marked_points_protocol(photostim,
                                targets_only=True,
                                file_path=str(path_2),
                                name="TRIALS_15_D_EM0566",
                                z_offset=21.44)

view_target_overlay(photostim, targets=photostim.stimulated_neurons)
