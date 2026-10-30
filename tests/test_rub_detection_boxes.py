import unittest
import cv2
import numpy as np

from continuous_testing import RubReference
from rub_preparation import draw_acid_roi
from test_continuous_testing import Photos,QUAD,scene


class BoxPhotos(Photos):
    def mark_observation(self,frame):
        return self.marks(frame),[(120,100,150,150)]


class RubDetectionBoxTests(unittest.TestCase):
    def test_actual_detection_box_is_drawn_instead_of_change_mask_outline(self):
        models=BoxPhotos();reference=RubReference(models,scene(),1,QUAD)
        frame=scene(mark=True)
        result=reference.after(models,frame,2,QUAD)
        self.assertEqual('NEW_RUB',result['status'])
        self.assertEqual('DETECTION_RECTANGLE',result['rub_display_kind'])
        self.assertEqual([[120,100,150,150]],result['rub_detection_boxes'])
        photo=result['photo']
        # The source box extends past the changed stripe. It must be drawn,
        # while neither the stripe edge nor its interior is painted magenta.
        np.testing.assert_array_equal([255,0,255],photo[140,120])
        np.testing.assert_array_equal(frame[120,130],photo[120,130])
        np.testing.assert_array_equal(frame[120,135],photo[120,135])
        self.assertGreater(np.count_nonzero(result['display_mask_reference']),
                           np.count_nonzero(result['new_mask']))

    def test_unchanged_detection_still_cannot_become_a_new_rub(self):
        models=BoxPhotos();reference=RubReference(models,scene(mark=True),1,QUAD)
        result=reference.after(models,scene(mark=True),2,QUAD)
        self.assertNotEqual('NEW_RUB',result['status'])

    def test_fragmented_internal_region_renders_as_a_rectangle_without_fill(self):
        frame=np.zeros((180,180,3),np.uint8)
        mask=np.zeros((180,180),np.uint8)
        mask[70:130,60:65]=255;mask[70:75,60:100]=255;mask[125:130,60:100]=255
        display=draw_acid_roi(frame.copy(),mask)
        np.testing.assert_array_equal([255,0,255],display[100,99])
        np.testing.assert_array_equal([0,0,0],display[100,80])


if __name__=='__main__':unittest.main()
