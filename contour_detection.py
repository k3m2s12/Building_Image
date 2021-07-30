import cv2
from collections.abc import Sequence
import numpy as np
from random import random


def n_max(list, N):
    final_list = []

    for i in range(0, N):
        max = 0

        for j in range(len(list)):
            if list[j] > max:
                max = list[j]

        print(max)
        list.remove(max)
        final_list.append(max)
    print(final_list)


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.
    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
    """
    if not isinstance(lst, Sequence):
        # base case
        return shape
    # peek ahead and assure all lists in the next depth have the same length

    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    # recurse
    shape += (len(lst),)
    shape = get_shape(lst[0], shape)

    return shape


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def check_before_or_after(diff_contours, image_before, image_after):
    '''
    function which check contours are for before or for after
    :param diff contours and two images
    :return match_list
    '''
    # Convert the image to grayscale
    before = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(image_after, cv2.COLOR_BGR2GRAY)

    # Threshold to make it binary
    ret_before, binary_before = cv2.threshold(before, 100, 255, cv2.THRESH_OTSU)
    ret_after, binary_after = cv2.threshold(after, 100, 255, cv2.THRESH_OTSU)

    inverted_before = ~binary_before
    inverted_after = ~binary_after

    contours_before, hierarchy_before = cv2.findContours(inverted_before, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_after, hierarchy_after = cv2.findContours(inverted_after, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    match_list = []

    for contour in diff_contours:
        before_match = 0
        after_match = 0
        for contour_before in contours_before:
            if (contour[0][0][0] == contour_before[0][0][0]) & (contour[0][0][1] == contour_before[0][0][1]):
                # print(contour[0][0][0], contour_before[0][0][0], contour[0][0][1], contour_before[0][0][1])

                before_match = 1

        for contour_after in contours_after:
            if (contour[0][0][0] == contour_after[0][0][0]) & (contour[0][0][1] == contour_after[0][0][1]):
                # print(contour[0][0][0], contour_after[0][0][0], contour[0][0][1], contour_after[0][0][1])

                after_match = 1

        match = [before_match, after_match]
        '''
        if match == [1, 0]:
            print("The building is demolished.")
        elif match == [1, 0]:
            print("The new building is constructed.")
        else:
            print("Anomaly.")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        '''
        match_list.append(match)

    return match_list


def contour_rectangle_generation(contour_list, draw_contour, threshold):
    for c in contour_list:
        x, y, w, h = cv2.boundingRect(c)

        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > threshold:
            cv2.rectangle(draw_contour, (x, y), (x + w, y + h), (150, 255, 150), 2)


def area_thresholding(contours, threshold):
    area_contours = np.array(get_contour_areas(contours))
    index_for_area = np.where(area_contours > threshold)
    indexes = index_for_area[0]
    threshold_contours = [contours[x] for x in indexes]
    # print("* Index which above threshold: ", indexes)
    # print("* Size which above threshold: ", indexes.size)

    return threshold_contours


def draw_specific_contour(contours, image, num):
    # Draw just the first contour
    # The 0 means to draw the first contour
    first_contour = cv2.drawContours(image, contours[148], -1, (255, 0, 255), 1)
    cv2.imshow('Specifically detected contour', first_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contour_rectangle_generation(contours[num], first_contour, 0)
    cv2.imshow('Specific contour with bounding box', first_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def centroid_test(centroid_list, contours_before, contours_after):
    for centroid in centroid_list:
        for contour_before in contours_before:
            cent_test_before = cv2.pointPolygonTest(contour_before, centroid, False)

        for contour_after in contours_after:
            cent_test_after = cv2.pointPolygonTest(contours_after, centroid, False)

    match_centroid = [cent_test_before, cent_test_after]
    return match_centroid


def get_area_list(contours):
    area_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_list.append(area)
    return area_list


def label_contour(contours, image):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, text=str(i + 1), org=(cx, cy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)


def contour_red_check(contours, diff_image, percent):
    red_index_list = []
    for contour in contours:
        for i in range(0, len(contour)):
            x = contour[i][0][1]
            y = contour[i][0][0]
            red_num = 0
            pd_num = 5

            for pd in range(1, pd_num):

                if (x < (2000 - pd)) & (y < (2000 - pd)):
                    check_list = [diff_image[x - pd, y - pd], diff_image[x - pd, y], diff_image[x - pd, y + pd],
                                  diff_image[x, y - pd], diff_image[x, y + pd],
                                  diff_image[x + pd, y - pd], diff_image[x + pd, y], diff_image[x + pd, y + pd]]

                    for check in check_list:
                        # If there is ENOUGH red, say YES
                        if (check[0] == 0) and (check[1] == 0):
                            red_num += 1

        red_percent = red_num / (len(contour) * 8)
        # print(red_percent)

        if red_percent > percent:
            red_index_list.append(contour)

    print("Detected Red Contours: ", len(red_index_list))

    return red_index_list


def combined_contours_extraction(contours, true_contour_list):
    combined_contour_list = []
    print("* Extraction from ", len(contours), " to ", len(true_contour_list))

    for contour in contours:
        contour_check = 0

        for i in range(0, len(true_contour_list)):
            if type(contour[0][0]) != type(np.uint8(0)):
                if (true_contour_list[i][0][0][0] != contour[0][0][0]) or \
                        (true_contour_list[i][0][0][1] != contour[0][0][1]):
                    contour_check += 1
            else:
                if (true_contour_list[i][0][0][0] != contour[0][0]) or \
                        (true_contour_list[i][0][0][1] != contour[0][1]):
                    contour_check += 1

        if contour_check == len(true_contour_list):
            combined_contour_list.append(contour)
    return combined_contour_list


def best_grid_generator(contours, trial_time, grid_size):
    contour_count_grid = []
    centroid_list = get_centroid_list(contours)
    top_tile = 400

    for i in range(trial_time):
        contours_in_grid = 0
        x = int(random() * (2000 - grid_size))
        y = int(random() * (2000 - grid_size))

        for centroid in centroid_list:
            centroid_check = [0, 0]
            if (x <= centroid[0]) & (centroid[0] <= x + grid_size):
                centroid_check[0] = 1
            if (y <= centroid[1]) & (centroid[1] <= y + grid_size):
                centroid_check[1] = 1
            if centroid_check == [1, 1]:
                contours_in_grid += 1

        contour_count_grid.append(contours_in_grid)

    n = int(len(contours) * (top_tile / 100))
    n_max(contour_count_grid, n)

    return n_max


def get_centroid_list(contours):
    centroid_list = []
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid_list.append([cx, cy])
    return centroid_list


########################################################################################
########################################################################################
if __name__ == "__main__":
    # Read Images - before - diff - after
    image_before = cv2.imread("2016.jpg")
    image_after = cv2.imread("2021.jpg")
    image_diff = cv2.imread("diff_gray.jpg")
    image_diff_rb = cv2.imread("diff_rb.jpg")

    # Convert the image to grayscale
    image_diff_gray = cv2.cvtColor(image_diff_rb, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(image_diff_gray, 100, 255, cv2.THRESH_OTSU)

    # Display the binary image and invert the image(i.e. 255 - pixel value)
    cv2.imshow('Binary image', binary)
    cv2.imwrite("binary_image.png", binary)
    cv2.waitKey(0)  # Wait for keypress to continue
    cv2.destroyAllWindows()  # Close windows
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("* Total Contour detected: ", len(contours))

    # Thresholding
    contours = area_thresholding(contours, 10)
    print("* Threshold-ed Contour detected: ", len(contours))

    # Contours for image_before & image_after
    image_before_gray = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    image_after_gray = cv2.cvtColor(image_after, cv2.COLOR_BGR2GRAY)

    ret_before, binary_before = cv2.threshold(image_before_gray, 100, 255, cv2.THRESH_OTSU)
    ret_after, binary_after = cv2.threshold(image_after_gray, 100, 255, cv2.THRESH_OTSU)

    inverted_before = ~binary_before
    inverted_after = ~binary_after

    contours_before, hierarchy_before = cv2.findContours(inverted_before, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_after, hierarchy_after = cv2.findContours(inverted_after, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    before_contours = cv2.drawContours(image_before, contours_before, -1, (0, 0, 255), 1)
    cv2.imwrite("before_contour.png", before_contours)

    after_contours = cv2.drawContours(image_after, contours_after, -1, (0, 0, 255), 1)
    # Contour Detections for difference image
    with_contours = cv2.drawContours(image_diff, contours, -1, (0, 0, 255), 1)
    cv2.imshow('Detected_contours_total', with_contours)
    cv2.imwrite("contour.png", with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_diff_rb = cv2.imread("diff_rb.jpg")

    # Red_check_before
    image_before = cv2.imread("2016.jpg")
    red_before_contours_list = contour_red_check(contours_before, with_contours, 0)
    red_before_contours = cv2.drawContours(image_before, red_before_contours_list, -1, (0, 0, 255), 1)
    cv2.imshow('red_before_contours_total', red_before_contours)
    cv2.imwrite("red_before_contour.png", red_before_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Red_check_before
    image_after = cv2.imread("2021.jpg")
    red_after_contours_list = contour_red_check(contours_after, with_contours, 0)
    red_after_contours = cv2.drawContours(image_after, red_after_contours_list, -1, (0, 0, 255), 1)
    cv2.imshow('red_after_contours_total', red_after_contours)
    cv2.imwrite("red_after_contour.png", red_after_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TRUE INDEX LIST EXTRACTION
    true_index_list = []
    contour_count = 0
    pd = 1

    for contour in contours:
        true_norm = 0

        for point in contour:
            x = point[0][1]
            y = point[0][0]

            if (x < 1999) & (y < 1999):
                check_list = [image_diff_rb[x - pd, y - pd], image_diff_rb[x - pd, y], image_diff_rb[x - pd, y + pd],
                              image_diff_rb[x, y - pd], image_diff_rb[x, y + pd],
                              image_diff_rb[x + pd, y - pd], image_diff_rb[x + pd, y], image_diff_rb[x + pd, y + pd]]

                for check in check_list:
                    # If there is red, say no
                    if check[0] < 220:
                        true_norm = 1
        # [240~255, 113~150, 0~100] ORANGE, [180~255, 180~ 255, 230~255] Sky-Blue, [245~255, 245~255, 245~255] White

        if true_norm == 0:
            true_index_list.append(contour_count)

        contour_count += 1

    true_contour_list = [contours[i] for i in true_index_list]

    # Draw the contours (in red) on the original image, color nn BGR (blue, green, red), -1 = all contours
    with_true_contours = cv2.drawContours(image_diff_rb, true_contour_list, -1, (0, 0, 255), 1)
    cv2.imshow('Detected_true_contours', with_true_contours)
    cv2.imwrite("true_con.png", with_true_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show the total number of contours that were detected
    print('* Total number of true_contours detected: ' + str(len(true_contour_list)))

    # Draw a bounding box around the first contour
    # x,y starting coordinate, w:width, h: height
    contour_rectangle_generation(true_contour_list, with_true_contours, 20)
    cv2.imshow('True contours with Bounding Box', with_true_contours)
    cv2.imwrite("true_con_BB.png", with_true_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check modification type.
    true_contour_match = check_before_or_after(true_contour_list, image_before, image_after)

    # Combined Contours (Have to reset with new cv2.imread)
    image_diff_rb = cv2.imread("diff_rb.jpg")
    combined_contour_list = combined_contours_extraction(contours, true_contour_list)
    combined_contours = cv2.drawContours(image_diff_rb, combined_contour_list, -1, (0, 0, 255), 1)
    cv2.imshow('Combined_contours', combined_contours)
    cv2.imwrite("combined_con.png", combined_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Contour rectangle generation
    contour_rectangle_generation(combined_contour_list, combined_contours, 30)
    print("* Combined Contour Detected: ", len(combined_contour_list))
    cv2.imshow('Combined_contours with Bounding Box', combined_contours)
    cv2.imwrite("combined_con_BB.png", combined_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combined contours for Before
    image_before = cv2.imread("2016.jpg")
    image_after = cv2.imread("2021.jpg")

    contours_before = area_thresholding(contours_before, 10)
    contours_after = area_thresholding(contours_after, 10)

    true_contour_before_list = combined_contours_extraction(red_before_contours_list, true_contour_list)
    true_contour_after_list = combined_contours_extraction(red_after_contours_list, true_contour_list)

    temp_true_contour_before_list = true_contour_before_list.copy()
    temp_true_contour_after_list = true_contour_after_list.copy()

    true_contour_before_list = combined_contours_extraction(true_contour_before_list, temp_true_contour_after_list)
    true_contour_after_list = combined_contours_extraction(true_contour_after_list, temp_true_contour_before_list)

    before_combined_contours = cv2.drawContours(image_before, true_contour_before_list, -1, (0, 0, 255), 1)
    cv2.fillPoly(image_before, pts=true_contour_before_list, color=(0, 0, 255))
    print("* Combined Contour Before: ", len(true_contour_before_list))
    cv2.imshow('Combined_contours_before', before_combined_contours)
    cv2.imwrite("Collections of buildings which is being changed.png.png", before_combined_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combined contours for After
    after_combined_contours = cv2.drawContours(image_after, true_contour_after_list, -1, (0, 0, 255), 1)
    cv2.fillPoly(image_after, pts=true_contour_after_list, color=(0, 0, 255))
    print("* Combined Contour After: ", len(true_contour_after_list))
    cv2.imshow('Combined_contours_after', after_combined_contours)
    cv2.imwrite("Collections of possibly modified buildings.png", after_combined_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Use centroid and get clustering to detect massive change
    grid_centroid_before = best_grid_generator(true_contour_before_list, 1000, 10)
    grid_centroid_after = best_grid_generator(true_contour_after_list, 1000, 10)
