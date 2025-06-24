// Covariance Shrinkage
//
// Quadratic Inverse shrinkage discussed in http://www.ledoit.net/BEJ1911-021R1A0.pdf

package covarianceShrinkage

import (
	"errors"
	"math"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Returns the Quadratic-Inverse-Shrinkage covariance estimator.
//
// Arguments:
// X *mat.Dense : matrix of t observations by n random variables. The data used to create the covariance matrix.
// Should be differenced if the covariance matrix is the covariance of the differences.
//
// S *mat.SymDense: nxn covariance matrix. If nil, S will be the sample covariance matrix

// deameanData bool: if true then demean the data, otherwise no demeaning takes place
//
// Returns:
// F *mat.Dense: the Quadratic-Inverse-Shrinkage covariance estimator
// norm float64: maximum absolute difference between covariance and shrinkage matrix
// err error
func QuadraticInverseShrinkage(X *mat.Dense, S *mat.SymDense, demeanData bool) (F *mat.Dense, norm float64, err error) {

	rawX := X.RawMatrix().Data

	t := X.RawMatrix().Rows
	n := X.RawMatrix().Cols
	k := 1

	// Sanity Checks
	if t <= 1 || n <= 0 {
		return nil, 0.0, errors.New("QIS: t must be greater than 1 and n must be greater than 0")
	}
	if len(rawX) != t*n {
		return nil, 0.0, errors.New("QIS: len(X) != t * n")
	}

	Y := mat.NewDense(t, n, rawX)
	Yc := mat.DenseCopyOf(Y)

	var tEff int
	if demeanData {
		k = 1

		// Demean the matrix X
		tEff = t - k
		if tEff <= 0 {
			return nil, 0.0, errors.New("QIS: effective sample size t-k ≤ 0")
		}

		for j := 0; j < n; j++ {
			colMean := mat.Sum(Y.ColView(j)) / float64(t)
			for i := 0; i < t; i++ {
				Yc.Set(i, j, Yc.At(i, j)-colMean)
			}
		}

	}

	if S == nil {
		S = mat.NewSymDense(n, nil)

		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				var sum float64
				for k := 0; k < t; k++ {
					yi := Yc.At(k, i)
					yj := Yc.At(k, j)
					sum += yi * yj
				}
				S.SetSym(i, j, sum/float64(tEff))
			}
		}
	}

	// Eigen-decomposition
	var eig mat.EigenSym
	Ssym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			Ssym.SetSym(i, j, S.At(i, j))
		}
	}
	if ok := eig.Factorize(Ssym, true); !ok {
		return nil, 0.0, errors.New("QIS: eigendecomposition failed")
	}

	lambda := eig.Values(nil)
	const eps = 1e-12
	for i := range lambda {
		if lambda[i] < eps {
			lambda[i] = eps // Reset negative values to 0
		}
	}

	var U mat.Dense
	eig.VectorsTo(&U) // The columns of U correspond to eigenvalues in monotone order

	// Constants
	tF := float64(tEff)
	nF := float64(n)
	c := nF / tF
	h := math.Pow(math.Min(c*c, 1/(c*c)), 0.35) / math.Pow(nF, 0.35) // Smoothing parameter
	m := int(math.Min(nF, tF))                                       //number of non-null eigenvalues
	start := n - m
	invLambda := make([]float64, m) //inverse of (non-null) eigenvalues
	for i := 0; i < m; i++ {
		invLambda[i] = 1 / lambda[start+i]
	}

	// Compute Quadratic-Inverse Shrinkage estimator of the covariance matrix
	theta := make([]float64, m)  //smoothed stein shrinker
	Htheta := make([]float64, m) //its conjugate
	for j := 0; j < m; j++ {
		invLJ := invLambda[j]
		var sumT, sumH float64
		for i := 0; i < m; i++ {
			invLI := invLambda[i]
			diff := invLI - invLJ
			den := diff*diff + invLI*invLI*h*h
			sumT += invLI * diff / den
			sumH += invLI * invLI * h / den
		}
		theta[j] = sumT / float64(m)
		Htheta[j] = sumH / float64(m)
	}

	Atheta2 := make([]float64, m) //its squared amplitude
	for i := range Atheta2 {
		Atheta2[i] = theta[i]*theta[i] + Htheta[i]*Htheta[i]
	}

	// Shrink eigenvalues
	delta := make([]float64, n)
	if n <= tEff {
		for j := 0; j < m; j++ {
			invLJ := invLambda[j]
			tj := theta[j]
			aj := Atheta2[j]
			den := (1-c)*(1-c)*invLJ + 2*c*(1-c)*invLJ*tj + c*c*invLJ*aj
			delta[start+j] = 1 / den
		}
	} else {
		meanInv := floats.Sum(invLambda) / float64(m)
		delta0 := 1 / ((c - 1) * meanInv)
		for i := 0; i < n-(tEff); i++ {
			delta[i] = delta0
		}
		for j := 0; j < m; j++ {
			delta[start+j] = 1 / (invLambda[j] * Atheta2[j])
		}
	}

	// Preserve Trace
	scale := floats.Sum(lambda) / floats.Sum(delta)
	floats.Scale(scale, delta)

	// Compute qis covariance estimator
	diagDelta := mat.NewDiagDense(n, delta)
	var tmp2 mat.Dense
	tmp2.Mul(&U, diagDelta)

	var sigma mat.Dense
	sigma.Mul(&tmp2, U.T())

	rm := sigma.RawMatrix()
	out := make([]float64, len(rm.Data))
	copy(out, rm.Data)

	cm := S.RawSymmetric()
	covMatrix := make([]float64, len(cm.Data))
	copy(covMatrix, cm.Data)

	maxAbsDiff := MaxAbsDiff(covMatrix, out)

	return &sigma, maxAbsDiff, nil
}
