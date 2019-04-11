# BestProjectInTheWorld (!)



The training set (train_x.csv, test_x.csv) consists of 50,000 8bit-greyscale
images of size 64x64. There is a unique label for each of the examples, which
is the result of the application of operator on the two digits in the image (
result for "a"/"A" is the sum of two digits and result for "m"/"M" is the
product of two digits ). The total num of unique classes are 40 . ( [0, 1, 2,
3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27,
28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] )

The test set (test_x.csv) consists of 10000 images of the same format.

The .csv file contains the raw uncompressed images pixel values in row major
format separated by ",". Each image is 64 x 64, so each row contains 4096 float
values, each containing the a float value which represents the 0-255 greyscale
intensity of each pixel.
