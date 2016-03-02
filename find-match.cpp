//
//
// Kshitij Agrawal
//
//
//

//*******************surf.cpp******************//

//********** SURF implementation in OpenCV*****//

//**loads image, computes SURF keypoints and descriptors **//


#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "affine2d.h"
#include "registration.hpp"

#include <iostream>
#define MAX_MATCHES 100

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define CONFIDENCE 0.99
#define DISTANCE 2

/** @function readme */
void readme()
{
  printf("Usage: ./SURF_detector <EO IMG> <IR IMG> <hessian>\n");
}

void cvt2point(const std::vector<cv::DMatch> matches,
               const std::vector<cv::KeyPoint> &keypoints1,
               const std::vector<cv::KeyPoint> &keypoints2,
               std::vector<cv::Point> &points1,
               std::vector<cv::Point> &points2)
{

  for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
    // Get the position of left keypoints
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point(x, y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point(x, y));
  }

}

// ransac test for stereo images
cv::Mat ransacTest(const std::vector<cv::DMatch> matches,
                const std::vector<cv::KeyPoint> &keypoints1,
                const std::vector<cv::KeyPoint> &keypoints2,
                std::vector<cv::DMatch> &goodMatches,
                double distance, double confidence,
                double minInlierRatio)
{
  goodMatches.clear();
  // Convert keypoints into Point
  std::vector<cv::Point> points1, points2;
  cvt2point(matches, keypoints1, keypoints2, points1, points2);
  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(), 0);
  // cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2),
  //                       inliers, CV_FM_RANSAC, distance, confidence);

  cv::Mat fundemental = cv::findHomography(cv::Mat(points1), cv::Mat(points2),
                        RANSAC, distance, inliers, 1000, confidence);

  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM = matches.begin();
  // for all matches
  for (; itIn != inliers.end(); ++itIn, ++itM) {
    if (*itIn) {
      // it is a valid match
      goodMatches.push_back(*itM);
    }
  }


  // The F matrix will be recomputed with
  // all accepted matches
  // Convert keypoints into Point
  // for final F computation
  points1.clear();
  points2.clear();
  cvt2point(goodMatches, keypoints1, keypoints2, points1, points2);

  // Compute 8-point F from all accepted matches
  fundemental = cv::findHomography(cv::Mat(points1), cv::Mat(points2), // matches
                  noArray(),0,2); // 8-point method
  // cout << fundemental << endl;

  return fundemental;


}

float *findAffineMat(
  const std::vector<cv::DMatch> &matches,
  const std::vector<cv::KeyPoint> &keypoints1,
  const std::vector<cv::KeyPoint> &keypoints2,
  float *X)
{

  // Calculate the Affine LS Homography

  int **final_matches;

  final_matches = (int **)malloc(sizeof(int *)*matches.size());
  for (int i = 0; i < matches.size(); i++) {
    final_matches[i] = (int *)malloc(sizeof(int) * 4);
  }

  for (int i = 0; i < matches.size(); i++) {

    final_matches[i][0] = keypoints1[matches[i].queryIdx].pt.x;
    final_matches[i][1] = keypoints1[matches[i].queryIdx].pt.y;
    final_matches[i][2] = keypoints2[matches[i].trainIdx].pt.x;
    final_matches[i][3] = keypoints2[matches[i].trainIdx].pt.y;
  }

  findAffine(matches.size(), final_matches, X);

}

int main(int argc, char **argv)
{
  if (argc != 4)
  { readme(); return -1; }

  Mat img_1 = imread(argv[1]);
  if (!img_1.data) {
    std::cout << " --(!) Error reading : %s" << argv[1] << std::endl;
    return -1;
  }
  Mat img_2 = imread(argv[2]);

  if (!img_2.data) {
    std::cout << " --(!) Error reading : %s" << argv[2] << std::endl;
    return -1;
  }
  // Convert to gray

  Mat gray_ref, gray_obj;

  cvtColor(img_1, gray_ref, CV_BGR2GRAY, 0);
  cvtColor(img_2, gray_obj, CV_BGR2GRAY, 0);

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = atoi(argv[3]);
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  detector->detectAndCompute(gray_ref, Mat(), keypoints_1, descriptors_1);

  minHessian = minHessian / 2;
  detector->setHessianThreshold(minHessian);
  detector->detectAndCompute(gray_obj, Mat(), keypoints_2, descriptors_2);

  #if 0
  // -- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1);
  imshow("Keypoints 2", img_keypoints_2);
  #endif

  // -- Perform Matching
  vector <DMatch> good_matches;
  flannKnn(descriptors_1, descriptors_2, good_matches);

  // -- Draw Matches
  Mat img_matches;

  // Mat mask = Mat::zeros(img_1.size(), CV_8U);  // type of mask is CV_8U
  // Mat roi(mask, cv::Rect(231,130,630,334));
  // roi = Scalar(255, 255, 255);
  // drawMatches(img_2, keypoints_2, img_1, keypoints_1, good_matches,
  //             img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
  //             vector< char >(), DrawMatchesFlags::DEFAULT);


  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::DEFAULT);

  printf("Found %d good matches\n", (int)good_matches.size());
  imshow("Good Matches & Object detection", img_matches);

  vector<DMatch> ransac_match;
  #if 1
  // Calculate the Affine LS Homography
  ransacTest(good_matches, keypoints_1, keypoints_2, ransac_match, DISTANCE, CONFIDENCE, 0.25);
  printf("Before RS %d After RS %d\n", good_matches.size(), ransac_match.size());

  drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac_match,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  imshow("After ransac", img_matches);

  float X[100];

  findAffineMat(ransac_match, keypoints_1, keypoints_2,X);
  
  cv::Mat affinemat = Mat(2, 3, CV_32FC1, X);
  
  cout << affinemat << endl;

  Mat timage;

  warpAffine(img_2, timage, affinemat, img_1.size(), INTER_LINEAR |
             WARP_INVERSE_MAP , BORDER_CONSTANT, Scalar());
  imshow("Transformed", timage);

  alphablending(img_1, timage);

  #else
  // -- Calculate Projective RANSAC

  Mat tmat = ransacTest(good_matches, keypoints_1, keypoints_2, ransac_match, DISTANCE, CONFIDENCE, 0.25);

  // cout << tmat << endl;

  drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac_match,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  imshow("After ransac", img_matches);


  Mat fimg;

  warpPerspective(img_2, fimg, tmat, img_1.size(), INTER_LINEAR |
    WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar());

  imshow("RANSAC", fimg);

  alphablending(img_1, fimg);

  // Mat affinemat = Mat::zeros(2, 3, CV_32FC1);
  // affinemat = findAffineMat(ransac_match, keypoints_1, keypoints_2);
  // cout << affinemat << endl;

  // warpAffine(img_2, fimg, affinemat, Size(720 , 576), INTER_LINEAR,
  //                 BORDER_CONSTANT, Scalar());

  // imshow("Affine", fimg);


  #endif

  waitKey(0);
  // free2d(&final_matches,ransac_match.size());
  return 0;
}

