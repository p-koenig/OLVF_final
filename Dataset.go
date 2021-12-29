package main

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"os"
	"strconv"
)

type Dataset struct {
	X     *mat.Dense
	y     *mat.VecDense
	cIter int
}

func (d *Dataset) randomizeRowOrder() {
	// tbd
}

func (d *Dataset) getNextSample() *Sample {
	sample := d.X.RowView(d.cIter).(*mat.VecDense)
	tmp := mat.NewVecDense(sample.Len(), nil)
	tmp.CopyVec(sample)

	sample = removeRandomFeaturesFromSample(sample)

	d.incrementCiter()
	return NewSample(sample, d.getyTruthforcurrentsample())
}

func removeRandomFeaturesFromSample(sample *mat.VecDense) *mat.VecDense {
	// first removal
	len, _ := sample.Dims()
	sample.SetVec(rand.Intn(len-1), -1)
	return sample
}

func (d *Dataset) getyTruthforcurrentsample() int {
	return int(d.y.AtVec(d.cIter))
}

func (d *Dataset) incrementCiter() {
	len, _ := d.X.Dims()
	if d.cIter >= len-1 {
		fmt.Println("OVERFLOW Citer")
	} else {
		d.cIter++
	}
}

func (d *Dataset) preprocessing() {
	fmt.Print(d.X)
}

func newDataset(filepathX string, filepathY string) *Dataset {
	d := new(Dataset)

	csvfile, err := os.Open(filepathX)
	if err != nil {
		fmt.Print("Error reading file")
	}
	reader := csv.NewReader(csvfile)

	// X
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Print("Error")
	}

	_X := mat.NewDense(len(records), len(records[0]), nil)
	for i, record := range records {
		for j, ele := range record {
			floatVal, err := strconv.ParseFloat(ele, 64)
			// fmt.Println(ele + " at" + strconv.Itoa(i) + "," + strconv.Itoa(j))
			if err != nil {
				fmt.Print("intconversionError\n")
			} else {
				_X.Set(i, j, floatVal)
			}

		}
	}
	d.X = _X

	// Y
	csvfileY, errY := os.Open(filepathY)
	if errY != nil {
		fmt.Print("Error reading file")
	}
	readerY := csv.NewReader(csvfileY)
	recordsY, errY := readerY.ReadAll()
	if errY != nil {
		fmt.Print("Error")
	}

	_y := mat.NewDense(len(recordsY), len(recordsY[0]), nil)
	for i, recordY := range recordsY {
		for j, ele := range recordY {
			floatVal, err := strconv.ParseFloat(ele, 64)
			// fmt.Println(ele + " at" + strconv.Itoa(i) + "," + strconv.Itoa(j))
			if err != nil {
				fmt.Print("intconversionError\n")
			} else {
				_y.Set(i, j, floatVal)
			}
		}
	}
	d.y = _y.ColView(1).(*mat.VecDense)
	d.cIter = 0

	return d
}

func (d *Dataset) printDataSet() {
	r, c := d.X.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Print(d.X.At(i, j))
			if j != c-1 {
				fmt.Print(" | ")
			}
		}
		fmt.Println()
	}
}
