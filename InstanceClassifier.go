package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type InstanceClassifier struct {
	val *mat.VecDense
}

func NewIC() *InstanceClassifier {
	ic := new(InstanceClassifier)
	ic.val = mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < ic.val.Len(); i++ {
		ic.val.SetVec(i, -1)
	}
	return ic
}

func (ic *InstanceClassifier) getProjectionsForGivenFeatureSpace(fs *FeatureSpace) (icExisting *mat.VecDense, icShared *mat.VecDense, icNew *mat.VecDense) {
	resExisting := mat.NewVecDense(params.sizeFS, nil)
	resShared := mat.NewVecDense(params.sizeFS, nil)
	resNew := mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < ic.val.Len(); i++ {
		switch fs.space.AtVec(i) {
		case -1:
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 0: // existing
			resExisting.SetVec(i, ic.val.AtVec(i))
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 1: // shared
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, ic.val.AtVec(i))
			resNew.SetVec(i, -1)
		case 2: // new
			resExisting.SetVec(i, -1)
			resShared.SetVec(i, -1)
			resNew.SetVec(i, ic.val.AtVec(i))
		}
	}
	return resExisting, resShared, resNew
}

func (ic *InstanceClassifier) update(sample *Sample, yTruth int, projectionConfidenceW float64, projectionConfidenceX float64, fs *FeatureSpace) {
	//icExisting, icShared, icNew := ic.getProjectionsForGivenFeatureSpace(fs)

	// fscExisting is staying the same

	for i := 0; i < ic.val.Len(); i++ {
		if fs.space.AtVec(i) == 1 {
			val1 := params.C * projectionConfidenceW * float64(yTruth) * sample.x.AtVec(i)
			val2 := projectionConfidenceW * params.learningRate * float64(yTruth) * sample.x.AtVec(i) / math.Pow(mat.Norm(sample.x, 2), 2)
			ic.val.SetVec(i, ic.val.AtVec(i)+math.Min(val1, val2))
		}

		if fs.space.AtVec(i) == 2 {
			val1 := params.C * projectionConfidenceX * float64(yTruth) * sample.x.AtVec(i)
			val2 := projectionConfidenceX * params.learningRate * float64(yTruth) * sample.x.AtVec(i) / math.Pow(mat.Norm(sample.x, 2), 2)
			ic.val.SetVec(i, ic.val.AtVec(i)+math.Min(val1, val2))
		}
	}
}

func (ic *InstanceClassifier) regularization(fsc *FeatureSpaceClassifier) {
	for i := 0; i < ic.val.Len(); i++ {
		if ic.val.AtVec(i) != -1 {
			ic.val.SetVec(i, math.Min(1.0, params.lambda/(ic.val.AtVec(i)*fsc.val.AtVec(i)))*ic.val.AtVec(i))
		}
	}
}
