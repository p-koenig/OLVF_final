package main

import "gonum.org/v1/gonum/mat"

type FeatureSpace struct { // -1 = never looked at, 0 = existing, 1 = shared, 2 = new
	space *mat.VecDense
	len   int
}

func NewFeatureSpace() *FeatureSpace {
	f := new(FeatureSpace)
	f.space = mat.NewVecDense(params.sizeFS, nil)
	for i := 0; i < f.space.Len(); i++ {
		f.space.SetVec(i, -1)
	}
	f.len = 0
	return f
}

func (fs *FeatureSpace) update(sample *Sample) {
	for i := 0; i < sample.getLength(); i++ {
		if sample.x.AtVec(i) != -1 && fs.space.AtVec(i) > -1 { // existing feature, present in new sample => shared
			fs.space.SetVec(i, 1)
		} else if sample.x.AtVec(i) != -1 && fs.space.AtVec(i) == -1 { // new feature => new
			fs.space.SetVec(i, 2)
		} else if sample.x.AtVec(i) == -1 && fs.space.AtVec(i) > 0 { // existing feature, missing in new sample => existing
			fs.space.SetVec(i, 0)
		}
	}
	if fs.len < sample.getLength() {
		fs.len = sample.getLength()
	}
}
