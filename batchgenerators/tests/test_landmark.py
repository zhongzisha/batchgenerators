import unittest
import numpy as np
from matplotlib import pyplot as plt

from batchgenerators.augmentations.spatial_transformations import augment_spatial
from batchgenerators.augmentations.crop_and_pad_augmentations import constrained_random_crop


class LandmarkTransform_2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
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
        np.testing.assert_array_equal(lm_out[0, 0], np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")

    # def test_2D_elastic(self):
    #     data_out, seg_out, lm_out = augment_spatial(data=self.data_2D, seg=self.seg_2D, patch_size=self.patch_size,
    #                                                 lm=self.lm,
    #                                                 do_elastic_deform=True, do_rotation=False, angle_x=(0, 2 * np.pi),
    #                                                 do_scale=False,
    #                                                 random_crop=False, seed=0)
    #
    #     index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)
    #     print("index from segmentation: ", index_out)
    #     print("landmark index: ", lm_out[0, 0])
    #     plt.imshow(self.data_2D_input[0, 0])
    #     plt.scatter(self.center_pixel[1], self.center_pixel[0])
    #     plt.show()
    #
    #     plt.imshow(data_out[0, 0])
    #     plt.scatter(index_out[1], index_out[0])
    #     plt.scatter(lm_out[0, 0, 1], lm_out[0, 0, 0])
    #     plt.show()
    #     np.testing.assert_array_equal(index_out, lm_out[0, 0],
    #                                   err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_rot(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=True, angle_x=(np.pi, np.pi),
                                                    do_scale=False, random_crop=False, seed=0, order_seg=3)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_scale(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=False, angle_x=(0, 2 * np.pi),
                                                    do_scale=True, random_crop=False, seed=0)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_crop(self):
        patch_size = (40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=patch_size, lm=self.lm, do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=False, seed=0)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_2D_random_crop(self):
        patch_size = (40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                    patch_size=patch_size, lm=np.copy(self.lm), do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=True, patch_center_dist_from_border=0, seed=None)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_constrained_random_crop(self):
        data_out, seg_out, lm_out = constrained_random_crop(data=np.copy(self.data_2D), seg=np.copy(self.seg_2D),
                                                            crop_size=40, anchor=self.center_pixel, lm=np.copy(self.lm),
                                                            margins=(10, 10))
        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")


class LandmarkTransform_3D(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        img_size = (59, 60, 70)
        self.center_pixel = np.array([20, 30, 40])
        self.data_3D = np.zeros(img_size)
        self.data_3D[self.center_pixel[0] - 5:self.center_pixel[0] + 5,
        self.center_pixel[1] - 5:self.center_pixel[1] + 5, self.center_pixel[2]-5:self.center_pixel[2]+5] = 1
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
            -np.linalg.norm(np.array([xaxis[:, None, None], yaxis[None, :, None], zaxis[None, None, :]]) - self.center_pixel) ** 2 / sigma_squared)
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
        np.testing.assert_array_equal(lm_out[0, 0], np.array(index_out),
                                      err_msg="Segmentation index and Landmark not the same.")

    # def test_2D_elastic(self):
    #     data_out, seg_out, lm_out = augment_spatial(data=self.data_2D, seg=self.seg_2D, patch_size=self.patch_size,
    #                                                 lm=self.lm,
    #                                                 do_elastic_deform=True, do_rotation=False, angle_x=(0, 2 * np.pi),
    #                                                 do_scale=False,
    #                                                 random_crop=False, seed=0)
    #
    #     index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)
    #     print("index from segmentation: ", index_out)
    #     print("landmark index: ", lm_out[0, 0])
    #     plt.imshow(self.data_2D_input[0, 0])
    #     plt.scatter(self.center_pixel[1], self.center_pixel[0])
    #     plt.show()
    #
    #     plt.imshow(data_out[0, 0])
    #     plt.scatter(index_out[1], index_out[0])
    #     plt.scatter(lm_out[0, 0, 1], lm_out[0, 0, 0])
    #     plt.show()
    #     np.testing.assert_array_equal(index_out, lm_out[0, 0],
    #                                   err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_rot(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=True, angle_x=(0, 2*np.pi),
                                                    angle_y=(0, 2*np.pi), angle_z=(0, 2*np.pi),
                                                    do_scale=False, random_crop=False, seed=0)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_scale(self):
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=self.patch_size, lm=np.copy(self.lm),
                                                    do_elastic_deform=False, do_rotation=False, angle_x=(0, 2 * np.pi),
                                                    do_scale=True, random_crop=False, seed=0)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_crop(self):
        patch_size = (40, 40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=patch_size, lm=self.lm, do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=False, seed=0)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_3D_random_crop(self):
        patch_size = (40, 40, 40)
        data_out, seg_out, lm_out = augment_spatial(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                    patch_size=patch_size, lm=np.copy(self.lm), do_elastic_deform=False,
                                                    do_rotation=False, angle_x=(0, 2 * np.pi), do_scale=False,
                                                    random_crop=True, patch_center_dist_from_border=0, seed=None)

        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")

    def test_constrained_random_crop(self):
        data_out, seg_out, lm_out = constrained_random_crop(data=np.copy(self.data_3D), seg=np.copy(self.seg_3D),
                                                            crop_size=40, anchor=self.center_pixel, lm=np.copy(self.lm),
                                                            margins=(10, 10, 10))
        index_out = np.unravel_index(np.argmax(seg_out[0, 0]), seg_out[0, 0].shape)

        np.testing.assert_array_equal(index_out, lm_out[0, 0],
                                      err_msg="Heatmap segmentation maximum and transformed landmark are not equal.")


if __name__ == '__main__':
    unittest.main()