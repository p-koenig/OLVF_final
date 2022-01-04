package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"sort"
)

type Param struct {
	C            float64
	lambda       float64
	B            float64 // (0, 1]
	numIter      int
	sizeFS       int
	learningRate float64
}

var params *Param = new(Param)

func predict(classifierWeights *mat.VecDense, current *mat.VecDense) int {
	mulres := mat.Dot(classifierWeights, current)
	if mulres > 0 {
		return 1
	} else {
		return -1
	}
}

func predictProjectionConfidences(fsc *FeatureSpaceClassifier, sample *FeatureSpace) (float64, float64) {
	// was ist sigma?
	return 0.5, 0.5 // dummy vals
}

func truncateFeatureSpace(fsc *FeatureSpaceClassifier, ic *InstanceClassifier, space *FeatureSpace) {
	pl := make(PairList, ic.val.Len())
	for i := 0; i < ic.val.Len(); i++ {
		pl[i] = Pair{i, ic.val.AtVec(i)}
	}
	sort.Sort(pl)

	for i := 0; i < pl.Len(); i++ {
		if float64(i)/float64(pl.Len()) > params.B {
			break
		} else {
			fsc.val.SetVec(pl[i].Key, -1)
			ic.val.SetVec(pl[i].Key, -1)
			space.space.SetVec(pl[i].Key, -1)
		}
	}
}

// Pair only helper
type Pair struct {
	Key   int
	Value float64
}

type PairList []Pair

func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func main() {
	params.C = 0.001
	params.lambda = 0.001
	params.B = 0.4
	params.numIter = 10
	params.sizeFS = 8
	params.learningRate = 0.1

	fmt.Println("Hello OLVF")

	df := newDataset("datasets/imdb.x.csv", "datasets/imdb.y.csv")
	df.printDataSet()

	// init classifiers
	instanceClassifier := NewIC()
	featureSpaceClassifier := NewFSC()

	// more inits
	featureSpace := NewFeatureSpace()
	fmt.Println("\n", "---- START ----", "\n")

	for t := 0; t < params.numIter; t++ {

		/// DEBUGGING
		fmt.Println("\n", "------------------------------------------------------------")
		fmt.Println("## Iteration ", t, "\n")
		fmt.Println("\n", "## FSC at Start: ")
		featureSpaceClassifier.print()
		fmt.Println("\n", "## IC at Start: ")
		instanceClassifier.print()
		// DEBUGGING END

		// 2
		sample := df.getNextSample()
		featureSpace.update(sample)

		// 3
		_, instanceClassifierSharedProjected, _ := instanceClassifier.getProjectionsForGivenFeatureSpace(featureSpace)

		// 4
		sampleSharedProjected, _ := sample.getProjectionsForGivenFeatureSpace(featureSpace)

		// 5
		yPred := predict(instanceClassifierSharedProjected, sampleSharedProjected)

		// 6
		yTruth := sample.y

		// 7
		featureSpaceClassifier.update(yTruth, yPred, featureSpace, sample)

		// 8 - 10
		projectionConfidenceW, projectionConfidenceX := predictProjectionConfidences(featureSpaceClassifier, featureSpace)

		// 11
		instanceClassifier.update(sample, yTruth, projectionConfidenceW, projectionConfidenceX, featureSpace)

		// 12
		instanceClassifier.regularization(featureSpaceClassifier)

		// 13
		truncateFeatureSpace(featureSpaceClassifier, instanceClassifier, featureSpace)
	}
}
