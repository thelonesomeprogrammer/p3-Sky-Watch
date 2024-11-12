import cv2
import numpy as np

img = cv2.imread("./Billed Data/06341.jpg")

deres_cam = np.array([[3.37557838e+03, 0.00000000e+00, 1.90060064e+03], [0.00000000e+00, 3.37627765e+03, 1.02106476e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
deres_dist = np.array([[-5.68718021e-02, -3.77364195e+00, -2.33949181e-03,  9.64380021e-03,   1.04120413e+01]])
vores_cam = np.array([[3400, 0, 1929], [0, 3400, 1091], [0, 0, 1]], dtype=np.float32)
vores_dist = np.zeros(4)

h, w = img.shape[:2]

dncm,  droi  = cv2.getOptimalNewCameraMatrix(deres_cam,deres_dist,(w,h),1,(w,h))
vncm,  vroi  = cv2.getOptimalNewCameraMatrix(vores_cam,vores_dist,(w,h),1,(w,h))
vdncm, vdroi = cv2.getOptimalNewCameraMatrix(vores_cam,deres_dist,(w,h),1,(w,h))
dvncm, dvroi = cv2.getOptimalNewCameraMatrix(deres_cam,vores_dist,(w,h),1,(w,h))
print(vdncm)
dx, dy, dw,dh = droi 
ddx, ddy, ddw,ddh = vroi 
dddx, dddy, dddw,dddh = vdroi 
ddddx, ddddy, ddddw,ddddh = dvroi 






ddst = cv2.undistort(img,deres_cam,deres_dist,None,dncm)
vdst = cv2.undistort(img,vores_cam,vores_dist,None,vncm)
vddst = cv2.undistort(img,vores_cam,deres_dist,None,vdncm)
dvdst = cv2.undistort(img,deres_cam,vores_dist,None,dvncm)

dst=ddst[dy:dy+dh,dx:dx+dw]
dest=vdst[ddy:ddy+ddh,ddx:ddx+ddw]
drst=vddst[dddy:dddy+dddh,dddx:dddx+dddw]
dtst=dvdst[ddddy:ddddy+ddddh,ddddx:ddddx+ddddw]



ddst = cv2.resize(dst,(1920,1080))
vdst = cv2.resize(dest,(1920,1080))
vddst = cv2.resize(drst,(1920,1080))
dvdst = cv2.resize(dtst,(1920,1080))

cv2.imwrite("ddst.jpg",ddst)
cv2.imshow("ddst",ddst)
cv2.waitKey(0)
cv2.imshow("vdst",vdst)
cv2.waitKey(0)
cv2.imshow("vddst",vddst)
cv2.waitKey(0)
cv2.imshow("dvdst",dvdst)
cv2.waitKey(0)
