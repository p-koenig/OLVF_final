package main

import "gonum.org/v1/gonum/mat"

type Sample struct {
	x *mat.VecDense
	y int
}

func NewSample(x *mat.VecDense, y int) *Sample {
	s := new(Sample)
	s.x = x
	s.y = y
	return s
}

func (s *Sample) getLength() int {
	return s.x.Len()
}

func (s *Sample) getProjectionsForGivenFeatureSpace(fs *FeatureSpace) (fscShared *mat.VecDense, fscNew *mat.VecDense) {
	resShared := mat.NewVecDense(params.sizeFS, nil)
	resNew := mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < s.x.Len(); i++ {
		switch fs.space.AtVec(i) {
		case -1:
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 0: // existing
			resShared.SetVec(i, -1)
			resNew.SetVec(i, -1)
		case 1: // shared
			resShared.SetVec(i, s.x.AtVec(i))
			resNew.SetVec(i, -1)
		case 2: // new
			resShared.SetVec(i, -1)
			resNew.SetVec(i, s.x.AtVec(i))
		}
	}
	return resShared, resNew
}
