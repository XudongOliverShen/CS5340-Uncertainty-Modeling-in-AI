import cv2
import numpy as np

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        self.out_height = out_height
        self.out_width = out_width

        # read in image
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)
        # some large number
        self.constant = 10000000

        # start
        self.seams_carving()


    def seams_carving(self):
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)
        #print('delta_row ', delta_row)
        #print('delta_col ', delta_col)

        if delta_col < 0:
            self.seams_removal(delta_col * -1)


    def seams_removal(self, num_pixel):
        for dummy in range(num_pixel): #1 for debug
            energy_map = self.calc_energy_map() # (111, 164)
            #print('energy_map shape ', energy_map.shape)
            seam_idx = self.viterbi(energy_map)

            self.delete_seam(seam_idx)


    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image) # split image into R,G,B channels (111, 164, 3)
        #print('self.out_image shape ', self.out_image.shape)
        #print('g ', g)
        #print('g.shape ', g.shape) #(111, 164)
        #print('r ', r)
        #print('r.shape ', r.shape)
        #print('b ', b)
        #print('b.shape ', b.shape)

        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        #print('b_energy ', b_energy.shape, 'g_energy ', g_energy.shape,'r_energy ', r_energy.shape) # (111, 164)
        return b_energy + g_energy + r_energy



    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3)) # (111,163,3)
        for row in range(m):
            col = seam_idx[row] 
            #Note that output[row, :]: (163,3)
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col]) # (163,), self.out_image[row, :, 0] (164,)
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col]) # numpy.delete(a, index) returns the array after deleting
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))

    def viterbi(self,energy_map):
        # For each row of the image, we delete a pixel, and the pixel has to be connected to pixels from its above row to form a "seam"
        # That is to say, a pixel to be deleted has to come from one of its upper, upper-left, or upper-right pixels
        # This is exactly the transition matrix of seamcarving
        sh = energy_map.shape
        path_energy = np.copy(energy_map)

        # Take log for Viterbi scoring computation
        for i in range(sh[0]):
            for j in range(sh[1]):
                # if pixel energy is 0, replace log with 0 - self.constant
                if path_energy[i][j] == 0:
                    path_energy[i][j] = 0-self.constant
                else:
                    path_energy[i][j] = np.log(path_energy[i][j])

        path = []
        for i in range(1, sh[0]):
            for j in range(sh[1]):
                if j == 0: # the left-most column does not have a connected pixel at its upper-left position
                    # log(p(transition)) = log(1) = 0, thus ignored in summation
                    prev_minpath_idx = np.argmin(path_energy[i-1, j:j+2])
                    path_energy[i][j] += path_energy[i-1][prev_minpath_idx] 
                # elif j == sh[1]-1: # the right-most column does not have a connected pixel at its upper-right position
                # numpy will take care of this case TODO: check
                else:
                    prev_minpath_idx = np.argmin(path_energy[i-1,j-1:j+2]) + j-1
                    path_energy[i][j] += path_energy[i-1][prev_minpath_idx]
        # Reconstruct the path (backtracking)
        path_end = np.argmin(path_energy[sh[0]-1,:])
        #print('path end ', path_end)
        #print('min path energy: ', path_energy[sh[0]-1, path_end])
        #print('min path energy alternative: ', np.min(path_energy[sh[0]-1,:]))
        path.append(path_end)
        start_idx = path_end
        offset_mapping_0 = {0:0, 1:1}
        offset_mapping = {0:-1, 1:0, 2:1}

        # hard code for debug
        #start_idx = 160

        for i in range(sh[0]-2, -1, -1):
            # need break ties or not??? TODO
            if start_idx == 0: # left-most column
                offset = np.argmin(path_energy[i, start_idx:start_idx+2]) # this returns a relative position to the current pixel
                start_idx += offset_mapping_0[offset]
            else:
                offset = np.argmin(path_energy[i, start_idx-1:start_idx+2])
                start_idx += offset_mapping[offset]
            path.append(start_idx)
        reversed_path = list(reversed(path))
        print('backtrack path ', reversed_path)
        print('len path ', len(reversed_path))
        return reversed_path

        




if __name__ == '__main__':
    filename_input = 'image_input.jpg'
    filename_output = 'image_output.jpg'


    height_input, width_input = cv2.imread(filename_input).astype(np.float64).shape[: 2]

    height_output = height_input
    width_output = width_input - 30 # TODO : DEBUG, the person disappeared... when -2, the person is still there
    print('Original image size: ', height_input,width_input)

    obj = SeamCarver(filename_input, height_output, width_output)
    obj.save_result(filename_output)







