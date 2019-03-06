import cv2
import numpy as np
import sys
from collections import Counter
from math import floor

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# Declaring Essentials
maxL = float('-inf')
minL = float('inf')
maxY = float('-inf')
minY = float('inf')
Xw = 0.95
Yw = 1.0
Zw = 1.09
uw = (4 * Xw)/(Xw + (15 * Yw) + (3 * Zw))
vw = (9 * Yw)/(Xw + (15 * Yw) + (3 * Zw))
# print("uw, vw ", uw, vw)

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
# inputImage = np.zeros([4, 4, 3], dtype=np.uint8)

if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# inputImage[0] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
# inputImage[1] = [[0, 0, 255],[0, 0, 255],[0, 0, 255],[0, 0, 255]]
# inputImage[2] = [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]]
# inputImage[3] = [[100, 100, 0], [100, 100, 0], [100, 100, 0], [100, 100, 0]]

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

# tmp = np.copy(inputImage)
#
# for i in range(H1, H2+1) :
#     for j in range(W1, W2+1) :
#         b, g, r = inputImage[i, j]
#         gray = round(0.3*r + 0.6*g + 0.1*b + 0.5)
#         tmp[i, j] = [gray, gray, gray]
#
# cv2.imshow('tmp', tmp)
# cv2.imwrite("gray_"+name_output, tmp)

# end of example of going over window

def invgamma(x):
    if x < 0.03928:
        x = x / 12.92
    else:
        x = ((x + 0.055) / 1.055) ** 2.4
    return x

def rgbToxyz(r, g, b):
    a1 = np. array([r,g, b])
    a2 = np. array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
    # print(np.dot(a2, a1))
    a3 = np.matmul(a2, a1)
    return a3[0], a3[1], a3[2]

def XYZToxyY(X, Y, Z):
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    return x, y, Y

def xyYToXYZ(x, y, Y):
    return x*Y/y, Y, (1-x-y)*Y/y

def XYZToLuv(X, Y, Z):
    t = Y/Yw
    if t > 0.008856:
        L = (116 * (t**(1/3))) - 16
    else:
        L = 903.3 * t
    if L < 0 :
        L = 0
    if L > 100:
        L = 100
    d = X + (15 * Y) + (3 * Z)
    if d!=0:
        uPrime = 4 * X/d
        vPrime = 9 * Y/d
    else:
        uPrime = 0
        vPrime = 0
    u = (13 * L) * (uPrime - uw)
    v = (13 * L) * (vPrime - vw)
    return L, u, v

def linearScaleHundred(x, max_value, min_value):
    # print("In Linear Scaling")
    A = 0
    B = 100
    a = min_value
    b = max_value
    # print(A, B, a, b)
    x = (((x-a)*(B-A))/(b-a)) + A
    # print("X value", x)
    return x

def linearScaleOne(x, max_value, min_value):
    # print("In Linear Scaling")
    A = 0
    B = 1
    a = min_value
    b = max_value
    # print(A, B, a, b)
    x = (((x-a)*(B-A))/(b-a)) + A
    # print("X value", x)
    return x

def LuvToXYZ(L, u, v):
    if L != 0:
        uPrime = (u + (13 * uw * L)) / (13 * L)
        vPrime = (v + (13 * vw * L)) / (13 * L)
    else:
        uPrime = 0
        vPrime = 0
    # print("uPrime, vPrime ", uPrime, vPrime)
    if L > 7.9996:
        Y = (((L + 16) / (116)) ** 3) * Yw
    else:
        Y = (L / 903.3) * Yw
    if vPrime == 0:
        X = 0
        Z = 0
    else:
        X = Y * 2.25 * (uPrime / vPrime)
        Z = (Y * (3 - (0.75 * uPrime) - (5 * vPrime))) / vPrime
    return X, Y, Z

def XYZtoRGB(X, Y, Z):
    a1 = np.array([X, Y, Z])
    a2 = np.array([[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648, -0.204043, 1.057311]])
    a3 = np.matmul(a2, a1)
    # print("a3", a3)
    for i in range(len(a3)):
        # print("a3 ",i , a3[i])
        if a3[i] > 1:
            a3[i] = 1
        if a3[i] < 0:
            a3[i] = 0
        if a3[i] < 0.00304:
            a3[i] = a3[i] * 12.92
        else:
            a3[i] = (1.055 * (a3[i] ** (1/2.4))) - 0.055
        # print("a3 ", i, a3[i])
    R, G, B = a3[0] * 255, a3[1] * 255, a3[2] * 255
    return R, G, B

# First Program
outputImage = np.zeros([rows, cols, bands], dtype=float)
finalOutputProgramOneImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows) :
    for j in range(0, cols) :
        if i in range(H1, H2+1) and j in range(W1, W2+1):
            b, g, r = inputImage[i, j]

            # print()
            # print("RGB", r, g, b)

            # Converting to Non Linear RGB
            r, g, b = r/255, g/255, b/255
            # print("Non Linear RGB ", r, g, b)

            # Converting to Linear RGB
            r, g, b = invgamma(r), invgamma(g), invgamma(b)
            # print("Linear RGB ",r, g, b)

            # Converting Color Image to XYZ
            X, Y, Z = rgbToxyz(r, g, b)
            # print("X Y Z ", X, Y, Z)

            # Converting to Luv
            L, u, v = XYZToLuv(X, Y, Z)
            # print("L, u, v ", L, u, v)

            minL = min(L, minL)
            maxL = max(L, maxL)
            outputImage[i,j] = [L, u, v]
        else:
            outputImage[i, j] = inputImage[i, j]

# Getting Lprime then converting it to XYZ then To Linear RGB and then to Non Linear RGB
for i in range(0, rows):
    for j in range(0, cols):
        if i in range(H1, H2 + 1) and j in range(W1, W2 + 1):
            L, u, v = outputImage[i, j]
            # print("Before scaling L u v ", L, u, v)

            # Scaling L to L Prime
            L = linearScaleHundred(L, maxL, minL)
            # print("L' u v ", L, u, v )

            # Convertingg Luv to XYZ
            X, Y, Z = LuvToXYZ(L, u, v)
            # print("X Y Z", X, Y, Z)

            # Converting XYZ to RGB
            R, G, B = XYZtoRGB(X, Y, Z)
            # print("R G B", R, G, B)
            finalOutputProgramOneImage[i, j] = [B, G, R]
        else:
            finalOutputProgramOneImage[i, j] = outputImage[i, j]

cv2.imshow("Program 1 : ", finalOutputProgramOneImage)
cv2.imwrite("Program 1_"+name_output, finalOutputProgramOneImage);

# Second Program
outputImage = np.zeros([rows, cols, bands], dtype=float)
finalOutputProgramTwoImage = np.zeros([rows, cols, bands], dtype=np.uint8)

list_of_L = []

for i in range(0, rows) :
    for j in range(0, cols) :
        if i in range(H1, H2+1) and j in range(W1, W2+1):
            b, g, r = inputImage[i, j]

            # print()
            # print("RGB", r, g, b)

            # Converting to Non Linear RGB
            r, g, b = r/255, g/255, b/255
            # print("Non Linear RGB ", r, g, b)

            # Converting to Linear RGB
            r, g, b = invgamma(r), invgamma(g), invgamma(b)
            # print("Linear RGB ",r, g, b)

            # Converting Color Image to XYZ
            X, Y, Z = rgbToxyz(r, g, b)
            # print("X Y Z ", X, Y, Z)

            # Converting to Luv
            L, u, v = XYZToLuv(X, Y, Z)
            # print("L, u, v ", L, u, v)

            list_of_L.append(floor(L))
            outputImage[i,j] = [L, u, v]
        else:
            outputImage[i, j] = inputImage[i, j]

h_of_L = Counter(list_of_L)
sorted_list_of_L = sorted(list(Counter(list_of_L).keys()))
f_of_L = {}
counter = 0
for L in sorted_list_of_L:
    counter+=h_of_L[L]
    f_of_L[L] = counter
# print("h of L", h_of_L)
# print("f of L", f_of_L)
new_L = {}
f_i_minus_one = 0
number_of_pixels = (H2-H1) * (W2-W1)
for L in sorted_list_of_L:
    new_L[L] = floor(((f_i_minus_one + f_of_L[L])/2) * (101/number_of_pixels))
    f_i_minus_one = f_of_L[L]
# print("new L ", new_L)

for i in range(0, rows):
    for j in range(0, cols):
        if i in range(H1, H2 + 1) and j in range(W1, W2 + 1):
            L, u, v = outputImage[i, j]
            L = new_L[floor(L)]


            # Convertingg Luv to XYZ
            X, Y, Z = LuvToXYZ(L, u, v)
            # print("X Y Z", X, Y, Z)

            # Converting XYZ to RGB
            R, G, B = XYZtoRGB(X, Y, Z)
            # print("R G B", R, G, B)
            finalOutputProgramTwoImage[i, j] = [B, G, R]
        else:
            finalOutputProgramTwoImage[i, j] = outputImage[i, j]

cv2.imshow("Program 2:", finalOutputProgramTwoImage)
cv2.imwrite("Program 2_"+name_output, finalOutputProgramTwoImage);

# Third Program
outputImage = np.zeros([rows, cols, bands], dtype=float)
finalOutputProgramThreeImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows) :
    for j in range(0, cols) :
        if i in range(H1, H2+1) and j in range(W1, W2+1):
            b, g, r = inputImage[i, j]

            # print()
            # print("RGB", r, g, b)

            # Converting to Non Linear RGB
            r, g, b = r/255, g/255, b/255
            # print("Non Linear RGB ", r, g, b)

            # Converting to Linear RGB
            r, g, b = invgamma(r), invgamma(g), invgamma(b)
            # print("Linear RGB ",r, g, b)

            # Converting Color Image to XYZ
            X, Y, Z = rgbToxyz(r, g, b)
            # print("X Y Z ", X, Y, Z)

            x, y, Y = XYZToxyY(X, Y, Z)

            minY = min(Y, minY)
            maxY = max(Y, maxY)
            outputImage[i, j] = [x, y, Y]
        else:
            outputImage[i, j] = inputImage[i, j]

for i in range(0, rows):
    for j in range(0, cols):
        if i in range(H1, H2 + 1) and j in range(W1, W2 + 1):
            x, y, Y = outputImage[i, j]
            # print("Before scaling L u v ", L, u, v)

            # Scaling L to L Prime
            Y = linearScaleOne(Y, maxY, minY)
            # print("L' u v ", L, u, v )

            # Convertingg Luv to XYZ
            X, Y, Z = xyYToXYZ(x, y, Y)
            # print("X Y Z", X, Y, Z)

            # Converting XYZ to RGB
            R, G, B = XYZtoRGB(X, Y, Z)
            # print("R G B", R, G, B)
            finalOutputProgramThreeImage[i, j] = [B, G, R]
        else:
            finalOutputProgramThreeImage[i, j] = outputImage[i, j]


cv2.imshow("Program 3:", finalOutputProgramThreeImage)
cv2.imwrite("Program 3_"+name_output, finalOutputProgramThreeImage);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()