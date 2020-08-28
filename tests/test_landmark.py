import unittest
import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.spatial_transformations import augment_mirroring


class LandmarkTransform2D(unittest.TestCase):
    def setUp(self):
        img_size = (59, 60)
        self.center_pixel = np.array([20, 30])
        self.data_2D = np.zeros(img_size)
        self.data_2D[self.center_pixel[0] - 5:self.center_pixel[0] + 5,
                     self.center_pixel[1] - 5:self.center_pixel[1] + 5] = 1
        self.data_2D = np.expand_dims(np.expand_dims(self.data_2D, 0), 0)
        xaxis = np.arange(0, img_size[0])
        yaxis = np.arange(0, img_size[1])

        self.lm = np.zeros((1, 1, 2))
        self.lm[0, 0, 0] = self.center_pixel[0]
        self.lm[0, 0, 1] = self.center_pixel[1]
        sigma_squared = 10
        self.patch_size = img_size
        self.seg_2D = np.exp(
            -np.linalg.norm(np.array([xaxis[:, None], yaxis[None, :]]) - self.center_pixel) ** 2 / sigma_squared)
        self.seg_2D = np.expand_dims(np.expand_dims(self.seg_2D, 0), 0)
        self.data_2D_input = np.copy(self.data_2D)

    def test_heatmap_index(self):
        index = np.unravel_index(np.argmax(self.seg_2D[0, 0]), self.seg_2D[0, 0].shape)
        np.testing.assert_array_equal(index, self.center_pixel, err_msg="heatmap center pixel not correct.")

    def test_2D_no_transform(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 0), do_scale=False,
                                                    random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_almost_equal(self.data_2D_input, data_out, decimal=6,
                                             err_msg="Input and Output data not the same.")
        np.testing.assert_array_equal(np.round(lm_out[0, 0]), np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")

    def test_2D_rot(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=True, angle_x=(np.pi, np.pi),
                                                    do_scale=False, random_crop=False, order_seg=3)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_mirror_axis0(self):

        did_mirror = False
        while did_mirror is False:
            data_out, seg_out, lm_out = augment_mirroring(sample_data=np.copy(self.data_2D[0]),
                                                          sample_seg=np.copy(self.seg_2D[0]),
                                                          lm=np.copy(self.lm[0]), axes=[0])

            if not np.array_equal(self.seg_2D[0], seg_out):
                did_mirror = True

        index_out = np.unravel_index(np.argmax(seg_out[0]), seg_out[0].shape)

        np.testing.assert_array_equal(np.round(lm_out), np.expand_dims(index_out, axis=0),
                                      err_msg="Heatmap segmentation maximum and mirrored"
                                              "landmark along axis 0 are not equal.")

    def test_2D_mirror_axis1(self):

        did_mirror = False
        while did_mirror is False:
            data_out, seg_out, lm_out = augment_mirroring(sample_data=np.copy(self.data_2D[0]),
                                                          sample_seg=np.copy(self.seg_2D[0]),
                                                          lm=np.copy(self.lm[0]), axes=[1])

            if not np.array_equal(self.seg_2D[0], seg_out):
                did_mirror = True

        index_out = np.unravel_index(np.argmax(seg_out[0]), seg_out[0].shape)

        np.testing.assert_array_equal(np.round(lm_out), np.expand_dims(index_out, axis=0),
                                      err_msg="Heatmap segmentation maximum and mirrored"
                                              "landmark along axis 1 are not equal.")

    def test_2D_scale(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=False, angle_x=(0, 2 * np.pi),
                                                    do_scale=True, random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_crop(self):
        patch_size = (40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=patch_size, lm=self.lm, do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_elastic_deform(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=True,
                                                    do_rotation=False, angle_x=(0, 0), do_scale=False,
                                                    random_crop=False, alpha=(0., 1000.), sigma=(40., 60.))

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(np.round(lm_out[0, 0]), np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")


class LandmarkTransform3D(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        img_size = (59, 60, 70)
        self.center_pixel = np.array([20, 30, 40])
        self.data_3D = np.zeros(img_size)
        self.data_3D[self.center_pixel[0] - 5:self.center_pixel[0] + 5,
                     self.center_pixel[1] - 5:self.center_pixel[1] + 5,
                     self.center_pixel[2]-5:self.center_pixel[2]+5] = 1
        self.data_3D = np.expand_dims(np.expand_dims(self.data_3D, 0), 0)
        xaxis = np.arange(0, img_size[0])
        yaxis = np.arange(0, img_size[1])
        zaxis = np.arange(0, img_size[2])

        self.lm = np.zeros((1, 1, 3))
        self.lm[0, 0, 0] = self.center_pixel[0]
        self.lm[0, 0, 1] = self.center_pixel[1]
        self.lm[0, 0, 2] = self.center_pixel[2]
        sigma_squared = 10
        self.patch_size = img_size
        self.seg_3D = np.exp(
            -np.linalg.norm(np.array([xaxis[:, None, None], yaxis[None, :, None], zaxis[None, None, :]])
                            - self.center_pixel) ** 2 / sigma_squared)
        self.seg_3D = np.expand_dims(np.expand_dims(self.seg_3D, 0), 0)
        self.data_3D_input = np.copy(self.data_3D)

    def test_heatmap_index(self):
        index = np.unravel_index(np.argmax(self.seg_3D[0, 0]), self.seg_3D[0, 0].shape)
        np.testing.assert_array_equal(index, self.center_pixel, err_msg="heatmap center pixel not correct.")

    def test_3D_no_transform(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 0), do_scale=False,
                                                    random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_almost_equal(self.data_3D_input, data_out, decimal=6,
                                             err_msg="Input and Output data not the same.")
        np.testing.assert_array_equal(np.round(lm_out[0, 0]), np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")

    def test_3D_elastic_deform(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=True,
                                                    do_rotation=False, angle_x=(0, 0), do_scale=False,
                                                    random_crop=False, alpha=(0., 900.), sigma=(20., 30.))

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(np.round(lm_out[0, 0]), np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")

    def test_2D_mirror_axis0(self):

        did_mirror = False
        while did_mirror is False:
            data_out, seg_out, lm_out = augment_mirroring(sample_data=np.copy(self.data_3D[0]),
                                                          sample_seg=np.copy(self.seg_3D[0]),
                                                          lm=np.copy(self.lm[0]), axes=[0])

            if not np.array_equal(self.seg_3D[0], seg_out):
                did_mirror = True

        index_out = np.unravel_index(np.argmax(seg_out[0]), seg_out[0].shape)

        np.testing.assert_array_equal(np.round(lm_out), np.expand_dims(index_out, axis=0),
                                      err_msg="Heatmap segmentation maximum and mirrored landmark are not equal.")

    def test_2D_mirror_axis1(self):

        did_mirror = False
        while did_mirror is False:
            data_out, seg_out, lm_out = augment_mirroring(sample_data=np.copy(self.data_3D[0]),
                                                          sample_seg=np.copy(self.seg_3D[0]),
                                                          lm=np.copy(self.lm[0]), axes=[1])

            if not np.array_equal(self.seg_3D[0], seg_out):
                did_mirror = True

        index_out = np.unravel_index(np.argmax(seg_out[0]), seg_out[0].shape)

        np.testing.assert_array_equal(np.round(lm_out), np.expand_dims(index_out, axis=0),
                                      err_msg="Heatmap segmentation maximum and mirrored landmark are not equal.")

    def test_2D_mirror_axis2(self):

        did_mirror = False
        while did_mirror is False:
            data_out, seg_out, lm_out = augment_mirroring(sample_data=np.copy(self.data_3D[0]),
                                                          sample_seg=np.copy(self.seg_3D[0]),
                                                          lm=np.copy(self.lm[0]), axes=[2])

            if not np.array_equal(self.seg_3D[0], seg_out):
                did_mirror = True

        index_out = np.unravel_index(np.argmax(seg_out[0]), seg_out[0].shape)

        np.testing.assert_array_equal(np.round(lm_out), np.expand_dims(index_out, axis=0),
                                      err_msg="Heatmap segmentation maximum and mirrored landmark are not equal.")

    def test_3D_rot(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=True, angle_x=(0, 2*np.pi),
                                                    angle_y=(0, 2*np.pi), angle_z=(0, 2*np.pi),
                                                    do_scale=False, random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_scale(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=False, angle_x=(0, 2 * np.pi),
                                                    do_scale=True, random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_crop(self):
        patch_size = (40, 40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=patch_size, lm=self.lm, do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=False)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, np.round(lm_out[0, 0]),
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")


if __name__ == '__main__':
    unittest.main()