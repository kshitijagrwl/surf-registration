//*******************surf.cpp******************//
//********** SURF implementation in OpenCV*****//
//**loads video from webcam, grabs frames computes SURF keypoints and descriptors**//  //** and marks them**//

//****author: achu_wilson@rediffmail.com****//

#include <stdio.h>
#include  <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/** @function readme */
void readme()
{
  printf("Usage: ./SURF_detector <EO IMG> <IR IMG> <hessian>\n");
}

int main(int argc, char **argv)
{
  if (argc != 4)
  { readme(); return -1; }

  Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
  Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

  if (!img_1.data || !img_2.data) {
    // std::cout << " --(!) Error reading images " << std::endl;
    printf("Error reading images\n");
    return -1;
  }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = atoi(argv[3]);
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

  //Hessian made twice for IR
  minHessian = minHessian;
  detector->setHessianThreshold(minHessian);
  detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

  #if 0
  // -- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1);
  imshow("Keypoints 2", img_keypoints_2);
  #else

  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match(descriptors_1, descriptors_2, matches);
  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2*min_dist, 0.02)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2,
              good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
              vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  //-- Show detected matches
  // imshow("Good Matches", img_matches);
  for (int i = 0; i < (int)good_matches.size(); i++) {
    printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
    obj.push_back( keypoints_2[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_1[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, RANSAC );

  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_2.cols, 0 );
  obj_corners[2] = cvPoint( img_2.cols, img_2.rows ); obj_corners[3] = cvPoint( 0, img_2.rows );
  std::vector<Point2f> scene_corners(4);
  perspectiveTransform( obj_corners, scene_corners, H);
  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  // line( img_matches, scene_corners[0] + Point2f( img_2.cols, 0), scene_corners[1] + Point2f( img_2.cols, 0), Scalar(0, 255, 0), 4 );
  // line( img_matches, scene_corners[1] + Point2f( img_2.cols, 0), scene_corners[2] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[2] + Point2f( img_2.cols, 0), scene_corners[3] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[3] + Point2f( img_2.cols, 0), scene_corners[0] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );

  #endif

  waitKey(0);

  return 0;
}

