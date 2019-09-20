def localizaFolhaTestePalo(self,adaptative_thres):
        sobelX = cv2.Sobel(adaptative_thres, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(adaptative_thres, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        thresholded_edge_sobel = cv2.bitwise_or(sobelX, sobelY)
        cv2.imwrite("tmp/thresholded_edge_sobel.png", thresholded_edge_sobel)

        #thresholded_edge_2 = cv2.Canny(adaptative_thres, 0, 255)
        # thresholded_edge_2=imutils.auto_canny(adaptative_thres)
        #cv2.imwrite("tmp/thresholded_bordas.png", thresholded_edge_2)
        im2, cnts, hierarchy = cv2.findContours(thresholded_edge_sobel.copy(), 
        cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

        # keep only 5 the largest ones
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        our_cnt = None

        for c in cnts:

            area = cv2.contourArea(c)

            if area > 260000:
                print(area)
                # print(area)
                peri = cv2.arcLength(c, True)
                epsilion = peri * 0.05
                approx = cv2.approxPolyDP(c, epsilion, True)
                if len(approx) == 4:
                    our_cnt = approx

                    break


