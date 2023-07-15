
# Epipolar geometry

Implementation of several algorithms of epipolar geometry in python. The main purpose was to create understandable and well-commented code to let everyone know how this algorithms work

## Currently(15.07.23) implemented algorithms

* Fundamental matrix estimation - 8point, 7point, levenberg marquardt

* RANSAC - simplest RANSAC

## To be implemented

* Normalized 8point + 7point

* Essential matrix estimation 

* Essential matrix decomposition

* Probably will continue with implementing further steps of sparse reconstruction(2-view triangulation, P3P, bundle adjustment)

## Dependencies

numpy, scipy, matplotlib, openCV(for feature detection+matching and comparison purposes)

