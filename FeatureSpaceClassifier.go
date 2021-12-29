package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type FeatureSpaceClassifier struct {
	val *mat.VecDense
}

func NewFSC() *FeatureSpaceClassifier {
	fsc := new(FeatureSpaceClassifier)
	fsc.val = mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < fsc.val.Len(); i++ {
		fsc.val.SetVec(i, -1)
	}
	return fsc
}

func (fsc *FeatureSpaceClassifier) update(yTrue int, yPred int, fs *FeatureSpace, sample *Sample) {
	var cp = 0.0
	if yTrue == yPred {
		cp = 1.0
	} else {
		cp = 0.0
	}
	// fscExisting is staying the same

	lossParam := math.Min(params.C, -params.learningRate/mat.Norm(sample.x, 2))
	for i := 0; i < fsc.val.Len(); i++ {
		if fs.space.AtVec(i) == 1 {
			exponentInner := -1 * (fsc.val.AtVec(i) + fsc.val.AtVec(i))
			fsc.val.SetVec(i, fsc.val.AtVec(i)-lossParam*((math.Log(math.Exp(exponentInner)))/(1+math.Log(math.Exp(exponentInner))))*cp)
		}
		if fs.space.AtVec(i) == 2 {
			exponentInner := -1 * (fsc.val.AtVec(i) + fsc.val.AtVec(i))
			fsc.val.SetVec(i, -lossParam*((math.Log(math.Exp(exponentInner)))/(1+math.Log(math.Exp(exponentInner))))*cp)
		}
	}
}

func (fsc *FeatureSpaceClassifier) getProjectionsForGivenFeatureSpace(fs *FeatureSpace) (fscExisting *mat.VecDense, fscShared *mat.VecDense, fscNew *mat.VecDense) {
	resExisting := mat.NewVecDense(params.sizeFS, nil)
	resShared := mat.NewVecDense(params.sizeFS, nil)
	resNew := mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < fsc.val.Len(); i++ {
		switch fs.space.AtVec(i) {
		case -1:
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 0: // existing
			resExisting.SetVec(i, fsc.val.AtVec(i))
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 1: // shared
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, fsc.val.AtVec(i))
			resNew.SetVec(i, -1)
		case 2: // new
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, -1)
			resNew.SetVec(i, fsc.val.AtVec(i))
		}
	}
	return resExisting, resShared, resNew
}
