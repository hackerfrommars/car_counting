import os
import logging
import logging.handlers
import random

import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

import utils
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
IMAGE_DIR = "./out"
# VIDEO_SOURCE = "input.mp4"
VIDEO_SOURCE = "prime_test_video.mp4"
SHAPE = (720, 1280)  # HxW
# SHAPE = (540, 960)  # HxW
EXIT_PTS = np.array([
    # [[270, 138], [724, 191], [724, 253], [282, 302], [259, 258]]
    [[360, 180], [984, 252], [984, 330], [390, 398], [344, 344]]
])
# ============================================================================


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    # processing pipline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR
                         # , min_contour_width=100, min_contour_height=100),
                         ),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        # VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        VehicleCounter(exit_masks=[exit_mask], x_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        # real frame number
        _frame_number += 1

        # skip every 2nd frame to speed up processing
        if _frame_number % 2 != 0:
            continue

        # frame number that will be passed to pipline
        # this needed to make video from cutted frames
        frame_number += 1

        # plt.imshow(frame)
        # plt.show()
        # return

        # imS = cv2.resize(frame, (640, 360))
        # cv2.imshow('frame', imS)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()

# ============================================================================

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
